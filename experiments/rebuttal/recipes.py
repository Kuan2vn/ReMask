"""Declared contrasts. Priorities concern evidence value, never observed scores."""
from copy import deepcopy

ARMS = {
    'F': {},
    'UA-noH': {'alpha': 0., 'only_a': True},
    'UA-noD': {'beta': 0., 'only_a': True},
    'TC-neither': {'parent': 'F', 'temporal': False, 'classes': False},
    'TC-Tonly': {'parent': 'F', 'classes': False},
    'TC-Conly': {'parent': 'F', 'temporal': False},
    'G-generic-med': {'probe': 'generic'},
    'G-uw-pixel': {'aggregation': 'pixel'},
    'G-none': {'probe': 'none'},
    'S-UA-restart': {'parent': 'F', 'late_ua': True},
    'S-UA-continuous': {'parent': 'F', 'continuous': True},
    'S-20k': {'parent': 'F', 'switch': 20000},
    'UA-neither': {'alpha': 0., 'beta': 0., 'only_a': True},
    'G-generic-pixel': {'probe': 'generic', 'aggregation': 'pixel'},
    'G-uw-mean': {'aggregation': 'mean'},
    'G-confidence': {'probe': 'identity'},
    'G-no-floor': {'floor': 0.},
    'S-27.5k': {'parent': 'S-UA-continuous', 'switch': 27500},
}
PRIORITY = ('F', 'TC-neither', 'TC-Tonly', 'TC-Conly', 'S-UA-restart',
            'S-UA-continuous', 'G-generic-med')
CORE = PRIORITY + ('G-uw-pixel', 'UA-noH', 'UA-noD', 'G-none', 'S-20k')
PROTOCOLS = ('uiis_to_suim', 'usis10k_to_suim')


def phases(arm, protocol, seed=1, smoke=False):
    if arm not in ARMS or protocol not in PROTOCOLS:
        raise ValueError('Unknown arm/protocol')
    is_uiis = protocol == PROTOCOLS[0]
    recipe = dict(probe='underwater', aggregation='median',
                  floor=.8,
                  alpha=1., beta=1.,
                  temporal=True, classes=True, switch=25000, seed=seed,
                  strength=.2,
                  exponent=1. if is_uiis else 1.5)
    recipe.update(deepcopy(ARMS[arm]))
    switch = recipe['switch']
    if smoke:
        switch = {20000: 1, 25000: 2, 27500: 3}[switch]
    total = 4 if smoke else 30000
    a_end = 2 if smoke else 25000
    a_horizon = (4 if smoke else 30000) if (is_uiis or recipe.get('continuous')) else (4 if smoke else 25000)
    a_warmup = (1 if smoke else 6000) if is_uiis else (1 if smoke else 5000)
    a_contrastive = .1 if is_uiis else 0.
    a = dict(recipe, phase='A', steps=a_end, horizon=a_horizon,
             offset=0, lr=2.5e-4, warmup_steps=a_warmup,
             contrastive=a_contrastive, reliability='ua', classes=False, parent=None,
             parent_step=None, full_resume=False)
    if recipe.get('continuous'):
        return [dict(a, steps=total, parent='F', parent_step=a_end, full_resume=True)]
    b = dict(recipe, phase='B', steps=total-switch, horizon=4 if smoke else 20000,
             offset=switch, lr=1e-4, warmup_steps=1 if smoke else 4000,
             contrastive=0., reliability='ua' if recipe.get('late_ua') else 'tc',
             parent=recipe.get('parent', arm), parent_step=switch, full_resume=False)
    if not is_uiis:
        b.update(floor=.6, exponent=1., strength=.35)
    if recipe.get('only_a') or (not is_uiis and arm.startswith('G-')):
        return [a]
    return [b] if recipe.get('parent') else [a, b]


def dependencies(arm):
    parent = ARMS[arm].get('parent')
    return dependencies(parent) + [parent] if parent else []


def ordered_arms(suite='priority', arm=None):
    all_order=CORE+tuple(name for name in ARMS if name not in CORE)
    selected = [arm] if arm else list({'priority': PRIORITY, 'core': CORE, 'all': all_order}[suite])
    ordered = []
    for item in selected:
        for dependency in dependencies(item) + [item]:
            if dependency not in ordered:
                ordered.append(dependency)
    return ordered
