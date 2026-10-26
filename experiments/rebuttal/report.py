"""Export every available arm/phase, selected checkpoints, and fixed-panel windows."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from metrics import overlap,paired_interval
from util import read_json,atomic_json


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Same output root passed to run.py')
    args=parser.parse_args()
    for mode in ('full','smoke'):
        root=args.output/mode
        if not root.exists():continue
        rows=[];quality_rows=[];windows=[];endpoints={};mass_rows=[]
        for config in sorted(root.glob('*/seed*/*/*/PHASE_CONFIG.json')):
            folder=config.parent;signature=read_json(config);p=signature['phase']
            metrics=read_json(folder/'metrics.json') if (folder/'metrics.json').exists() else []
            status=read_json(folder/'RUN_STATE.json').get('status') if (folder/'RUN_STATE.json').exists() else 'not_started'
            resource=read_json(folder/'resources.json') if (folder/'resources.json').exists() else {}
            lower=p['parent_step'] if p['full_resume'] else 0
            eligible=[r for r in metrics if r['local_updates']>lower]
            prefix=dict(protocol=signature['protocol'],seed=p['seed'],arm=signature['arm'],phase=p['phase'],status=status)
            if (folder/'training_diagnostics.jsonl').exists():
                for line in (folder/'training_diagnostics.jsonl').read_text().splitlines():
                    sample=json.loads(line)
                    if 'weighted_ce_class_mass' in sample:
                        mass_rows.append(dict(prefix,global_updates=sample['global_updates'],
                            **{f'class{c}_mass':value for c,value in enumerate(sample['weighted_ce_class_mass'])}))
            if not eligible:
                rows.append(dict(prefix,role='no_adapted_evaluation'));continue
            choices={'endpoint' if status=='complete' else 'last_measured':eligible[-1],
                     'peak':max(eligible,key=lambda r:r['miou'])}
            for role,row in choices.items():
                output=dict(prefix,role=role,local_updates=row['local_updates'],global_updates=row['global_updates'],
                            miou_percent=100*row['miou'],fg_miou_percent=100*row['fg_miou'],
                            boundary_iou_percent=100*row['boundary_iou'],fg_boundary_iou_percent=100*row['fg_boundary_iou'],
                            incremental_wall_seconds=resource.get('wall_seconds'),
                            incremental_training_seconds=resource.get('training_seconds'),
                            incremental_evaluation_seconds=resource.get('evaluation_seconds'),
                            incremental_candidate_seconds=resource.get('candidate_seconds'),
                            incremental_checkpoint_seconds=resource.get('checkpoint_seconds'),
                            peak_allocated_mb=resource.get('peak_allocated_bytes',0)/1024**2,
                            deployment_parameters=resource.get('deployment_parameters'),
                            inference_ms=row['inference_ms_median'],
                            gpu_segments='|'.join(sorted({s['gpu'] for s in resource.get('segments',[])})),
                            cost_scope='full observed phase;include parent prefix once for full-pipeline cost')
                for c,value in enumerate(row['class_iou']):output[f'iou_class{c}']=100*value
                for c,value in enumerate(row['class_accuracy']):output[f'accuracy_class{c}']=100*value
                for c,value in enumerate(row['class_dice']):output[f'dice_class{c}']=100*value
                rows.append(output)
            if status=='complete':endpoints[(prefix['protocol'],prefix['seed'],prefix['arm'],prefix['phase'])]=eligible[-1]
            for row in eligible:
                for score in sorted({x['score'] for x in row['quality']}):
                    q=[x for x in row['quality'] if x['score']==score]
                    quality_rows.append(dict(prefix,global_updates=row['global_updates'],score=score,
                        mean_image_aurc=float(np.mean([x['aurc'] for x in q])),
                        mean_image_precision75=float(np.mean([x['precision_at_75'] for x in q]))))
            intervals={}
            for row in metrics:
                if row['local_updates']<=lower:continue
                end=((row['global_updates']-1)//5000+1)*5000
                intervals.setdefault(end,[]).append(row['unlabeled_same_view_panel'])
            for end,panels in sorted(intervals.items()):
                pixels=sum(p['pixels'] for p in panels);mismatches=sum(p['mismatches'] for p in panels)
                support=np.asarray([p['predicted_class_support'] for p in panels]).sum(0)
                wrong=np.asarray([p['predicted_class_mismatch'] for p in panels]).sum(0)
                row=dict(prefix,window_start=end-5000,window_end=end,snapshots=len(panels),
                         sampled_snapshot_mean_mismatch_percent=100*mismatches/max(pixels,1),
                         foreground_mismatch_percent=100*wrong[1:].sum()/support[1:].sum() if support[1:].sum() else None,
                         predicted_class_macro_mismatch_percent=100*float(np.mean(wrong[support>0]/support[support>0])) if (support>0).any() else None,
                         definition='fixed_unlabeled_panel;sum mismatches/sum pixels at saved evaluations;not every optimizer step')
                for c in range(8):
                    row[f'class{c}_mismatch_percent']=100*wrong[c]/support[c] if support[c] else None
                    row[f'class{c}_support']=int(support[c])
                windows.append(row)
        write_csv(root/'results.csv',rows)
        write_csv(root/'quality_over_time.csv',quality_rows)
        write_csv(root/'mismatch_5k_windows.csv',windows)
        write_csv(root/'supervision_mass.csv',mass_rows)
        contrasts=[]
        for key,full in endpoints.items():
            protocol,seed,arm,phase=key
            if arm!='F' or phase!='B':continue
            for other in ('TC-neither','TC-Tonly','TC-Conly','S-UA-restart','G-generic-med','G-uw-pixel','G-none'):
                comparison=endpoints.get((protocol,seed,other,'B'))
                if comparison is None:continue
                names=[r['image'] for r in full['per_image']]
                if names!=[r['image'] for r in comparison['per_image']]:raise ValueError('Paired evaluation images differ')
                a=np.array([r['confusion'] for r in full['per_image']])
                b=np.array([r['confusion'] for r in comparison['per_image']])
                rng=np.random.default_rng(1)
                delta=[]
                for _ in range(2000):
                    indices=rng.integers(0,len(a),len(a))
                    delta.append(100*(overlap(a[indices].sum(0))['miou']-overlap(b[indices].sum(0))['miou']))
                contrasts.append(dict(protocol=protocol,seed=seed,comparison='F-minus-'+other,
                                      endpoint_delta_pp=100*(full['miou']-comparison['miou']),
                                      paired_image_bootstrap_ci95_pp=np.quantile(delta,[.025,.975]).tolist(),
                                      inference_scope='conditional on trained endpoints;not training-seed uncertainty'))
        atomic_json(root/'paired_endpoint_contrasts.json',contrasts)
        print('REPORT',root,'phase-results:',len(rows),'paired contrasts:',len(contrasts),flush=True)


if __name__=='__main__':main()
