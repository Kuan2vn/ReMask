"""Stable-ST class-balanced image resampling from predicted candidate labels.

Ported from cityscapes_b.py at c447da35598b285eb8868caf4f969aaba1bf39b8.
No target annotation file is read. Preserve the author's500-draw subepochs,
inverse-log class weights, class-pool pointers and skip-singleton rule.
"""
import pickle
from pathlib import Path

import numpy as np


def resample(dataset, cfg, rsc, max_iters):
    with open(Path(cfg.OUTPUT_DIR) / rsc, 'rb') as stream:
        pools, file_classes = pickle.load(stream)
    pools = {cid: list(pools.get(cid, [])) for cid in range(dataset.NUM_CLASS)}
    lookup = {record['name']: record for record in dataset.data_list}
    assert all(Path(name).stem in lookup for names in pools.values() for name in names)
    if not any(len(names) > 1 for names in pools.values()):
        # Tiny correctness subsets may not contain a repeatable class pool.
        # Full-data runs still use the author sampling rule whenever defined.
        return dataset
    eligible = np.asarray([len(pools[cid]) > 1 for cid in range(dataset.NUM_CLASS)])
    indices = dict.fromkeys(range(dataset.NUM_CLASS), 0)
    selected = []
    for _ in range(int(max_iters / 500) + 1):
        distribution = np.zeros(dataset.NUM_CLASS)
        for _ in range(500):
            frequency = distribution / distribution.sum() if distribution.sum() else distribution.copy()
            weights = 1 / np.log(1.01 + frequency)
            # Condition the author rejection sampler on available classes.
            # Otherwise absent classes can consume almost all draws in an8c
            # target and starve the loader before its declared update budget.
            weights[~eligible] = 0
            weights /= weights.sum()
            cid = int(np.random.choice(dataset.NUM_CLASS, p=weights))
            if len(pools[cid]) <= 1:
                continue
            if indices[cid] > len(pools[cid]) - 1:
                np.random.shuffle(pools[cid])
                indices[cid] %= len(pools[cid]) - 1
            name = pools[cid][indices[cid]]
            selected.append(lookup[Path(name).stem])
            indices[cid] += 1
            distribution[file_classes[name]] += 1
    assert selected, 'Class pools were present but resampling produced no examples'
    dataset.data_list = selected
    dataset.img_ids = [record['name'] for record in selected]
    print('Stable-ST predicted-label image resampling:', len(selected), 'records', flush=True)
    return dataset
