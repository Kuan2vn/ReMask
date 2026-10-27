"""Plot committed test-mIoU observations, with separate curves across stage resets."""
import argparse
import json
from pathlib import Path
import uuid


def observations(folder):
    folder = Path(folder)
    if not (folder/'CURRENT.json').exists() or not (folder/'metrics.json').exists():
        return []
    pointer = json.loads((folder/'CURRENT.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'PHASE_CONFIG.json').read_text(encoding='utf-8'))
    phase = config['phase']
    lower = phase['parent_step'] if phase.get('full_resume') else 0
    rows = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    return [dict(local_updates=r['local_updates'], global_updates=r['global_updates'],
                 miou_percent=100*float(r['miou']), adapted=r['local_updates']>lower)
            for r in rows if lower <= r['local_updates'] <= pointer['iteration']]


def draw(series, destination, title):
    import os
    # Colab exports an inline backend unavailable in the isolated training venv.
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    try:
        for name, rows in series:
            ax.plot([r['global_updates'] for r in rows], [r['miou_percent'] for r in rows],
                    marker='o', markersize=3, label=name)
            adapted = [r for r in rows if r['adapted']]
            if adapted:
                peak = max(adapted, key=lambda r:r['miou_percent'])
                ax.scatter([peak['global_updates']], [peak['miou_percent']], marker='*', s=100)
                align_right = peak['global_updates'] >= rows[-1]['global_updates']*.7
                ax.annotate(f"{name} peak {peak['miou_percent']:.2f}% @ {peak['global_updates']}",
                            (peak['global_updates'], peak['miou_percent']), xytext=(-4 if align_right else 4, 10),
                            ha='right' if align_right else 'left', textcoords='offset points', fontsize=8)
        values = [r['miou_percent'] for _, rows in series for r in rows]
        padding = max(.5, (max(values)-min(values))*.15)
        ax.set_ylim(max(0, min(values)-padding), min(100, max(values)+padding))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.set(xlabel='Global optimizer updates (committed evaluations)', ylabel='Test mIoU8 (%)', title=title)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        destination = Path(destination)
        temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
        fig.savefig(temporary, format='png', dpi=150)
        temporary.replace(destination)
    finally:
        plt.close(fig)


def export_arm(arm_folder):
    arm_folder = Path(arm_folder)
    combined = []
    for config_path in sorted(arm_folder.glob('*/PHASE_CONFIG.json')):
        folder = config_path.parent
        rows = observations(folder)
        if not rows:
            continue
        config = json.loads(config_path.read_text(encoding='utf-8'))
        title = f"{config['protocol']} / {config['arm']}"
        if config.get('diagnostic'):
            title += ' [SMOKE ONLY]'
        draw([(folder.name, rows)], folder/'performance.png', title+' / '+folder.name)
        combined.append((folder.name, rows))
    if combined:
        path = arm_folder/'performance.png'
        draw(combined, path, title+' — observed test evaluations')
        print('PERFORMANCE_PNG '+str(path.resolve()), flush=True)
        return path
    return None


def install(control_class, paused_exception):
    original = control_class.boundary
    def render(control):
        import random
        import numpy as np
        import torch
        # Plotting must not consume the augmentation/training random streams.
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:
            export_arm(control.out.parent)
        except Exception as error:
            print(f'[PERFORMANCE WARNING] {type(error).__name__}: {error}; native checkpoint preserved.', flush=True)
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
    def boundary(self, *args, **kwargs):
        try:
            result = original(self, *args, **kwargs)
        except paused_exception:
            render(self)
            raise
        else:
            render(self)
            return result
    control_class.boundary = boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for mode in ('full', 'smoke'):
        for arm_folder in sorted({p.parent.parent for p in (args.output/mode).glob('*/seed*/*/*/PHASE_CONFIG.json')}):
            export_arm(arm_folder)


if __name__ == '__main__':
    main()
