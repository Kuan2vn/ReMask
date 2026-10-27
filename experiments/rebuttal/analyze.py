"""Frozen-checkpoint UW/generic reliability comparison with image-level pairing."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE/'native'))
sys.path.append(str(REPO/'baselines/underwater'))
import trainer
from underwater_baselines.common import Segmenter,decode_mask
from underwater_baselines.assets import source_checkpoint
from underwater_baselines.launch import validate_data
from metrics import quality,paired_interval
from probes import calibration,generic_probe
from recipes import PROTOCOLS
from run import dataset_signature
from util import atomic_json,digest,read_json,tree_hash


def summarize(rows,output,signature,by_condition=True):
    grouped={}
    for row in rows:
        key=(row['severity'],row['probe'],row['aggregation'],row['context'])
        grouped.setdefault(key,{}).setdefault(row['image'],[]).append(row['quality'])
    summary=[]
    for key,images in grouped.items():
        means={image:{metric:float(np.mean([r[metric] for r in draws if r.get(metric) is not None]))
                      for metric in ('aurc','precision_at_75','error_rate')}
               for image,draws in images.items()}
        summary.append(dict(severity=key[0],probe=key[1],aggregation=key[2],context=key[3],
                            images=len(images),mean_image_aurc=float(np.mean([v['aurc'] for v in means.values()])),
                            mean_image_precision75=float(np.mean([v['precision_at_75'] for v in means.values()]))))
    contrasts=[]
    for severity in sorted({row['severity'] for row in rows}):
        for agg in ('pixel','median','mean'):
            for context in ('gate','base_x_gate'):
                keys=[(severity,probe,agg,context) for probe in ('underwater','generic')]
                if not all(k in grouped for k in keys):continue
                paired=sorted(set(grouped[keys[0]])&set(grouped[keys[1]]))
                differences=[]
                for name in paired:
                    values=[np.mean([draw['aurc'] for draw in grouped[k][name]]) for k in keys]
                    differences.append(values[0]-values[1])
                contrasts.append(dict(severity=severity,aggregation=agg,context=context,
                                      contrast='UW-minus-generic mean-image AURC;negative favors UW',
                                      **paired_interval(differences)))
    atomic_json(output/'summary.json',dict(signature=signature,summary=summary,paired_contrasts=contrasts,
                uncertainty='paired image bootstrap;draws averaged within image;conditional on checkpoint'))
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    if by_condition:
        names=sorted({group for row in rows for group in row.get('strata',[]) if group!='all'})
        for group in names:
            subset=[row for row in rows if group in row.get('strata',[])]
            if len({row['image'] for row in subset})<2:continue
            destination=output/'strata'/group;destination.mkdir(parents=True,exist_ok=True)
            summarize(subset,destination,dict(signature,stratum=group),False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=(*PROTOCOLS,'all'),default='all')
    parser.add_argument('--data',type=Path,default=REPO/'data/SUIM')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/rebuttal')
    parser.add_argument('--checkpoint',type=Path,help='Optional fixed source/UA checkpoint;requires single protocol')
    parser.add_argument('--base',choices=['confidence','ua'],default='confidence')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--draws',type=int,default=3)
    parser.add_argument('--severities',type=float,nargs='+',default=[.5,1.,1.5])
    parser.add_argument('--quality-stride',type=int,default=8)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.checkpoint and args.protocol=='all':parser.error('--checkpoint requires a single protocol')
    if args.base=='ua' and not args.checkpoint:parser.error('UA base requires a real paired teacher/student checkpoint')
    if args.draws<1 or args.quality_stride<1 or any(not 0<s<=2 for s in args.severities):parser.error('Invalid draws/stride/severity')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    data=validate_data(args.data,args.smoke)
    signature_data=dataset_signature(data)
    for protocol in PROTOCOLS if args.protocol=='all' else (args.protocol,):
        source=source_checkpoint(protocol,REPO/'assets/baselines')
        checkpoint=args.checkpoint or Path(source['path'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=Segmenter().to(args.device).eval()
        teacher_fea=saved.get('teacher_feature_extractor',saved['feature_extractor'])
        teacher_cls=saved.get('teacher_classifier',saved['classifier'])
        trainer._ensure_classifier_buffers_from_state(model.head,teacher_cls)
        trainer._load_module_state(model.feature,teacher_fea,strict=True)
        trainer._load_module_state(model.head,teacher_cls,strict=True)
        student=stats=None
        if args.base=='ua':
            if not all(k in saved for k in ('student_feature_extractor','student_classifier')):raise ValueError('Checkpoint has no actual student;cannot reconstruct UA')
            stats=saved.get('training_state',{}).get('uncertainty_stats')
            if not stats or any(v is None for v in stats.values()):raise ValueError('No saved training normalization stats')
            stats={k:v.to(args.device) for k,v in stats.items()}
            student=Segmenter().to(args.device).eval()
            trainer._ensure_classifier_buffers_from_state(student.head,saved['student_classifier'])
            trainer._load_module_state(student.feature,saved['student_feature_extractor'],strict=True)
            trainer._load_module_state(student.head,saved['student_classifier'],strict=True)
        del saved
        strength=.2 if protocol==PROTOCOLS[0] else .35
        output=args.output.resolve()/('smoke' if args.smoke else 'full')/'frozen'/protocol
        output.mkdir(parents=True,exist_ok=True)
        calib=calibration(output/'calibration.json',data,signature_data,strength,trainer._underwater_reliability_perturb,args.smoke)
        from conditions import describe,thresholds,strata
        cutoffs=thresholds([data/'train_val/images'/name for name in calib['images']])
        signature=dict(protocol=protocol,checkpoint_sha256=digest(checkpoint),source_sha256=source['sha256'],
                       dataset_sha256=signature_data,code_sha256=tree_hash(HERE),base=args.base,
                       draws=args.draws,severities=args.severities,quality_stride=args.quality_stride,diagnostic=args.smoke,
                       appearance_thresholds=cutoffs,condition_scope='unlabeled appearance proxies;all groups reported;overlapping')
        if (output/'CONFIG.json').exists() and read_json(output/'CONFIG.json')!=signature:raise ValueError('Frozen output has different checkpoint/settings;choose new --output')
        atomic_json(output/'CONFIG.json',signature)
        files=sorted((data/'TEST/images').glob('*.jpg'))[:2 if args.smoke else 110]
        per_image=output/'images';per_image.mkdir(exist_ok=True)
        for index,path in enumerate(files):
            result_file=per_image/(path.stem+'.json')
            if result_file.exists():continue
            with Image.open(path) as image:rgb=TF.to_tensor(image.convert('RGB').resize((128,96) if args.smoke else (1280,640),Image.Resampling.BICUBIC)).unsqueeze(0).to(args.device)
            masks=[p for ext in ('.bmp','.png') if (p:=data/'TEST/masks'/(path.stem+ext)).exists()]
            with Image.open(masks[0]) as image:y=decode_mask(image)
            normalize=lambda x:TF.normalize(x,[.485,.456,.406],[.229,.224,.225])
            rows=[]
            with torch.no_grad():
                logits=model(normalize(rgb));prob=logits.softmax(1)
                orig=F.interpolate(logits,y.shape,mode='bilinear',align_corners=True).argmax(1)[0].cpu().numpy()
                base=prob.max(1,keepdim=True)[0]
                if student is not None:
                    sp=student(normalize(rgb)).softmax(1)
                    h=-(prob*prob.clamp_min(1e-8).log()).sum(1,keepdim=True)
                    d=(prob*(prob.clamp_min(1e-8).log()-sp.clamp_min(1e-8).log())).sum(1,keepdim=True)
                    h=trainer._normalize_uncertainty(h,stats['entropy_mean'],stats['entropy_var'],1e-6)
                    d=trainer._normalize_uncertainty(d,stats['kl_mean'],stats['kl_var'],1e-6)
                    base=torch.exp(-h-d)
                pseudo=prob.argmax(1)
                stride=args.quality_stride
                valid=(y[::stride,::stride]>=0)&(y[::stride,::stride]<8)
                pred=orig[::stride,::stride]; correct=pred==y[::stride,::stride]
                raw_base=F.interpolate(base,y.shape,mode='nearest')[0,0,::stride,::stride].cpu().numpy()
                baseline_quality=quality(raw_base[valid],correct[valid])
                if baseline_quality:
                    rows.append(dict(image=path.name,severity=0.,draw=0,probe='base',aggregation='pixel',
                                     context='base',quality=baseline_quality))
                for severity in args.severities:
                    for draw in range(args.draws):
                        for family in ('underwater','generic','identity'):
                            devices=[torch.cuda.current_device()] if args.device=='cuda' else []
                            with torch.random.fork_rng(devices=devices):
                                torch.manual_seed(100000+index*101+draw)
                                transformed=(trainer._underwater_reliability_perturb(rgb,strength*severity) if family=='underwater' else
                                             generic_probe(rgb,calib['generic_amplitude']*severity) if family=='generic' else rgb)
                                pp=model(normalize(transformed)).softmax(1) if family!='identity' else prob
                            for aggregation in ('pixel','median','mean'):
                                stability,_,_=trainer._compute_underwater_stability(prob,pp,pseudo,255,'pixel',4,8)
                                if aggregation=='median': stability,_=trainer._segmentwise_median_stability(stability,pseudo,255,4,8)
                                if aggregation=='mean':
                                    from scipy import ndimage
                                    import types
                                    globals_copy=dict(trainer._segmentwise_median_stability.__globals__)
                                    class Mean:
                                        def __getattr__(self,name):return ndimage.mean if name=='median' else getattr(ndimage,name)
                                    globals_copy['ndimage']=Mean()
                                    function=types.FunctionType(trainer._segmentwise_median_stability.__code__,globals_copy)
                                    stability,_=function(stability,pseudo,255,4,8)
                                floor=.8 if protocol==PROTOCOLS[0] else .6
                                gate=floor+(1.-floor)*stability
                                for context,score in [('gate',gate),('base_x_gate',base*gate.pow(1.))]:
                                    score=F.interpolate(score,y.shape,mode='nearest')[0,0,::stride,::stride].cpu().numpy()
                                    result=quality(score[valid],correct[valid])
                                    if result is None:continue
                                    rows.append(dict(image=path.name,severity=severity,draw=draw,probe=family,
                                                aggregation=aggregation,context=context,quality=result,
                                                by_predicted_class={str(c):quality(score[valid&(pred==c)],correct[valid&(pred==c)]) for c in range(8)}))
            appearance=describe(path)
            for row in rows:
                row['appearance']=appearance;row['strata']=strata(appearance,cutoffs)
            atomic_json(result_file,rows)
            print('FROZEN',protocol,index+1,'/',len(files),path.name,flush=True)
        all_rows=[row for path in sorted(per_image.glob('*.json')) for row in read_json(path)]
        summarize(all_rows,output,signature)
        atomic_json(output/'complete.json',dict(images=len(files),diagnostic=args.smoke,signature=signature))


if __name__=='__main__':main()
