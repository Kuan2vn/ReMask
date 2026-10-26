"""Read-only evidence for global scheduling versus heterogeneous pixel stability."""
import argparse
from collections import deque
from pathlib import Path
import numpy as np
from report import write_csv
from util import read_json

CLASSES = ('water', 'diver', 'plant', 'wreck', 'robot', 'reef', 'fish', 'floor')


def persistent_agreement(history):
    """Agreement at every saved observation AND unchanged teacher label. No GT."""
    teachers = np.stack([h[0] for h in history])
    mismatches = np.stack([h[1] for h in history])
    return (~mismatches).all(0) & (teachers == teachers[-1]).all(0)


def panel_summary(panel):
    support = np.asarray(panel['predicted_class_support'])
    wrong = np.asarray(panel['predicted_class_mismatch'])
    result = dict(mismatch_percent=100*panel['mismatches']/max(panel['pixels'], 1),
                  foreground_mismatch_percent=100*wrong[1:].sum()/support[1:].sum() if support[1:].sum() else None,
                  macro_class_mismatch_percent=100*np.mean(wrong[support>0]/support[support>0]) if (support>0).any() else None,
                  raw_kl=panel['raw_kl'],teacher_drift_percent=100*panel['teacher_drift'] if panel.get('teacher_drift') is not None else None)
    for c, name in enumerate(CLASSES):
        result[name+'_mismatch_percent'] = 100*wrong[c]/support[c] if support[c] else None
        result[name+'_predicted_support'] = int(support[c])
    images = panel.get('per_image', [])
    if images:
        rates = [100*r['mismatches']/max(r['pixels'], 1) for r in images]
        result.update(zip(('image_mismatch_median', 'image_mismatch_p90', 'image_mismatch_p95'),
                          np.quantile(rates, [.5,.9,.95]).tolist()))
    return result


def export(folder, destination, window=3):
    destination.mkdir(parents=True, exist_ok=True)
    config=read_json(folder/'PHASE_CONFIG.json') if (folder/'PHASE_CONFIG.json').exists() else {}
    label=config.get('protocol','unknown protocol')+' / '+folder.parent.name+'/'+folder.name
    if config.get('diagnostic'):label+=' [SMOKE: not a method result]'
    pointer = read_json(folder/'CURRENT.json') if (folder/'CURRENT.json').exists() else None
    metrics = read_json(folder/'metrics.json') if (folder/'metrics.json').exists() else []
    # Ignore evaluations that were not committed in a full checkpoint transaction.
    metrics = [r for r in metrics if pointer and r['local_updates'] <= pointer['iteration']]
    history = deque(maxlen=window); names = None; previous = None
    rows=[]; image_rows=[]; spatial_rows=[]
    for row in metrics:
        panel = row['unlabeled_same_view_panel']
        item = dict(phase=row['phase'],local_updates=row['local_updates'],global_updates=row['global_updates'],
                    images=len(panel['images']),manual_labels_used=False,**panel_summary(panel),
                    persistent_agreement_percent=None,stability_window_observations=window,
                    stability_window_start=None,pixel_maps_available=False)
        for measurement in panel.get('per_image', []):
            image_rows.append(dict(phase=row['phase'],global_updates=row['global_updates'],**measurement,
                                   mismatch_percent=100*measurement['mismatches']/max(measurement['pixels'],1)))
        map_path = folder/panel['pixel_map'] if panel.get('pixel_map') else None
        if map_path and map_path.exists():
            with np.load(map_path, allow_pickle=False) as saved:
                teacher = saved['teacher'].copy(); mismatch = saved['mismatch'].astype(bool)
                current_names = saved['names'].tolist()
            if names is not None and names != current_names:
                raise ValueError('The diagnostic panel changed within a phase')
            names=current_names; item['pixel_maps_available']=True
            if previous is not None and teacher.shape != previous:
                raise ValueError('Spatial diagnostic shape changed within a phase')
            previous=teacher.shape
            # Exclude source-only/reset observations from the persistent-stability window.
            if row['local_updates']>0:
                history.append((teacher,mismatch,row['global_updates']))
            if len(history)==window:
                stable=persistent_agreement(history)
                item['persistent_agreement_percent']=100*float(stable.mean())
                item['stability_window_start']=history[0][2]
                for c,name in enumerate(CLASSES):
                    selected=teacher==c
                    spatial_rows.append(dict(phase=row['phase'],global_updates=row['global_updates'],
                        predicted_class=name,sampled_support=int(selected.sum()),
                        persistent_agreement_percent=100*float(stable[selected].mean()) if selected.any() else None,
                        pixel_stride=8,observations=window,window_start=history[0][2]))
            if row['global_updates'] in (20000,25000) and row['phase']=='A':
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                n=min(3,len(names)); fig,axes=plt.subplots(n,2,squeeze=False,figsize=(9,2.6*n))
                for i in range(n):
                    axes[i,0].imshow(teacher[i],vmin=0,vmax=7,cmap='tab10')
                    axes[i,1].imshow(mismatch[i],vmin=0,vmax=1,cmap='gray_r')
                    axes[i,0].set_title(names[i]+' | teacher class')
                    axes[i,1].set_title('T/S mismatch, stride 8')
                    for ax in axes[i]:ax.axis('off')
                fig.tight_layout();fig.savefig(destination/f'maps_{row["global_updates"]}.png',dpi=150);plt.close(fig)
        else:
            history.clear()
        rows.append(item)
    for name,values in [('over_time.csv',rows),('per_image.csv',image_rows),('pixel_stability_by_class.csv',spatial_rows)]:
        if values:write_csv(destination/name,values)
        else:(destination/name).write_text('',encoding='utf-8')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,10),sharex=True)
        x=[r['global_updates'] for r in rows]
        for key,series_label in [('mismatch_percent','all pixels'),('foreground_mismatch_percent','foreground'),
                          ('macro_class_mismatch_percent','class macro'),('image_mismatch_p90','image p90')]:
            axes[0].plot(x,[r.get(key) if r.get(key) is not None else np.nan for r in rows],label=series_label)
        for name in CLASSES:
            axes[1].plot(x,[r[name+'_mismatch_percent'] if r[name+'_mismatch_percent'] is not None else np.nan for r in rows],label=name)
        axes[2].plot(x,[r['persistent_agreement_percent'] if r['persistent_agreement_percent'] is not None else np.nan for r in rows],label=f'{window}-observation stable-agreement proxy')
        for ax in axes:
            if min(x)<=25000<=max(x):ax.axvline(25000,color='gray',linestyle=':')
            ax.set_ylabel('Percent');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=4)
        axes[2].set_ylim(0,100)
        if not any(r['persistent_agreement_percent'] is not None for r in rows):
            axes[2].text(.5,.5,f'Needs {window} saved post-initialization observations with maps',ha='center',transform=axes[2].transAxes)
        axes[2].set_xlabel('Optimizer updates (observed evaluations only)')
        fig.suptitle(label+' — fixed unlabeled panel')
        fig.tight_layout();fig.savefig(destination/'dynamics.png',dpi=150);plt.close(fig)
    latest=rows[-1] if rows else None
    text = f'''# Stage-transition evidence: {label}

Reviewer RgDe asks why a synchronized UA→TC schedule is reasonable when images,
classes and regions may stabilize at different times. QNjz asks for evidence
behind the 25k/5k split. These exports measure that heterogeneity; they do not
change ReMask into an adaptive pixel-wise switching method.

- Committed observations: {len(rows)}. Latest global update: {latest['global_updates'] if latest else 'not measured'}.
- Fixed unlabeled images: {latest['images'] if latest else 'not measured'}; deterministic selection seed 17.
- Latest same-view T/S mismatch: {round(latest['mismatch_percent'],4) if latest else 'not measured'}%.
- Spatial maps available at {sum(r['pixel_maps_available'] for r in rows)} observations, stride 8.
- Exact 20k observation: {any(r['global_updates']==20000 for r in rows)}; exact 25k observation: {any(r['global_updates']==25000 for r in rows)}.

`over_time.csv` separates all-pixel, foreground, predicted-class macro and per-class
mismatch and per-image quantiles. Denominators are predicted teacher classes,
not true classes; absent classes are missing, never zero. `per_image.csv` retains
the distribution instead of treating pixels as independent experiments.

Persistent agreement requires teacher/student agreement at all {window} consecutive
saved observations and an unchanged teacher label at the sampled pixel. This is a
label-free stability proxy, not correctness, not a Stage-2 assignment, and not
proof that pixel-adaptive switching is inferior. No manual target labels enter
these diagnostics. Source/reset update 0 is excluded from stability windows.

Read phases separately: Stage B resets the teacher and its temporal history.
Agreement at source/reset initialization can be high by construction. Check
`quality_over_time.csv` for the separate labeled evaluation of reliability
ranking, and `mismatch_5k_windows.csv` for saved-snapshot averages, not continuous
measurement of every update. Label-free panel agreement cannot establish that
25k is optimal or that continuing Stage A cannot improve performance; use the
matched continuation and alternate-split controls for those questions.

If class/image curves differ, report that directly: a global curriculum can
coexist with locally varying reliability weights. The result may support a
practical shared schedule, while a direct comparison would be needed to claim
superiority over an adaptive pixel schedule. Do not assume a 5% threshold or a
favorable trend before observing data. Partial runs are partial evidence.
'''
    (destination/'SUPERVISOR_REPORT.md').write_text(text,encoding='utf-8')
    print(destination,':',len(rows),'committed observations')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Same output root as run.py')
    parser.add_argument('--arm',default='F')
    args=parser.parse_args()
    found=False
    for mode in ('full','smoke'):
        for config in sorted((args.output/mode).glob(f'*/seed*/{args.arm}/*/PHASE_CONFIG.json')):
            folder=config.parent
            destination=args.output/mode/'dynamics'/folder.relative_to(args.output/mode)
            export(folder,destination);found=True
    if not found:print('No saved phase yet; run the F cell first. No results have been fabricated.')


if __name__=='__main__':main()
