"""Checkpoint transactions, fixed-panel diagnostics and original-size evaluation."""
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from metrics import confusion, overlap, boundary_counts, quality
from util import atomic_json, bank_hash, copy_banks, digest, read_json, sync


class SessionPaused(Exception):
    pass


class Control:
    def __init__(self, task, trainer):
        self.task, self.trainer = task, trainer
        self.phase, self.out = task['phase'], Path(task['output'])
        self.device = task['device']
        self.started = self.last_wall = time.monotonic()
        self.deadline = self.started + task['seconds']
        self.metrics, self.shadow, self.panel_history = [], {}, {}
        self.stats = dict(entropy_mean=None, entropy_var=None, kl_mean=None, kl_var=None)
        self.resources = dict(wall_seconds=0., training_seconds=0., evaluation_seconds=0.,
                              candidate_seconds=0., checkpoint_seconds=0., incremental_optimizer_updates=0)
        self.best, self.bank_iteration, self.seen_step = -1., 0, 0
        self.pending, self.last_candidate_time = None, 0.
        self.segment_start_updates=0

    def batch_start(self):
        sync(self.device)
        self.batch_started=time.perf_counter()

    @torch.no_grad()
    def observe(self, state):
        self.seen_step=state['iteration']+1
        t,s=state['teacher_prob'].detach(),state['student_prob'].detach()
        entropy=-(t*t.clamp_min(1e-8).log()).sum(1,keepdim=True)
        kl=(t*(t.clamp_min(1e-8).log()-s.clamp_min(1e-8).log())).sum(1,keepdim=True)
        for key,value in [('entropy',entropy),('kl',kl)]:
            self.trainer._update_running_scalar_stats(self.stats,key,value,.99,1e-6)
        if self.seen_step%20==0 or self.task['smoke']:
            labels=t.argmax(1)
            mismatch=labels!=s.argmax(1)
            row=dict(local_updates=self.seen_step, global_updates=self.phase['offset']+self.seen_step,
                     raw_kl=float(kl.mean()), entropy=float(entropy.mean()),
                     mismatches=int(mismatch.sum()), pixels=mismatch.numel(),
                     predicted_class_support=torch.bincount(labels.flatten(),minlength=8).cpu().tolist(),
                     predicted_class_mismatch=torch.bincount(labels[mismatch],minlength=8).cpu().tolist(),
                     series='training_views_before_mixing_weight_override')
            final_labels=state['pseudo_label'].detach()
            valid=(final_labels>=0)&(final_labels<8)
            mass=torch.bincount(final_labels[valid],weights=state['reliability'].detach().squeeze(1)[valid],minlength=8)
            row['weighted_ce_class_mass']=(mass/mass.sum().clamp_min(1e-8)).cpu().tolist()
            with (self.out/'training_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
        sync(self.device)
        self.resources['training_seconds']+=time.perf_counter()-self.batch_started
        self.resources['incremental_optimizer_updates']+=1

    def candidate(self, function, *args, **kwargs):
        feature,classifier=args[1:3]
        modes=feature.training,classifier.training
        sync(self.device)
        start=time.perf_counter()
        try:
            value=function(*args,**kwargs)
        finally:
            feature.train(modes[0]); classifier.train(modes[1])
        sync(self.device)
        self.resources['candidate_seconds']+=time.perf_counter()-start
        self.bank_iteration=self.seen_step
        return value

    def initial(self, state):
        checkpoint=state.get('checkpoint') or {}
        saved=checkpoint.get('experiment_control')
        if self.task.get('resume_own') and saved:
            self.metrics=saved['metrics']
            self.shadow=saved['shadow']
            self.panel_history=saved.get('panel_history',{})
            self.stats={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in saved['stats'].items()}
            self.bank_iteration=saved['bank_iteration']
            pointer=read_json(self.out/'CURRENT.json')
            self.resources=pointer['resources']
            self.best=saved['best']
            lower=self.phase['parent_step'] if self.phase['full_resume'] else 0
            eligible=[row for row in self.metrics if row['local_updates']>lower]
            if eligible:
                self.recover_best(max(eligible,key=lambda row:row['miou'])['local_updates'])
            diagnostic_file=self.out/'training_diagnostics.jsonl'
            if diagnostic_file.exists():
                committed=[line for line in diagnostic_file.read_text().splitlines()
                           if json.loads(line)['local_updates']<=state['iteration']]
                diagnostic_file.write_text('\n'.join(committed)+'\n',encoding='utf-8')
        elif self.phase.get('full_resume') and saved:
            self.metrics=saved['metrics']
            self.shadow=saved['shadow']
            self.panel_history=saved.get('panel_history',{})
            self.stats={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in saved['stats'].items()}
            self.bank_iteration=saved['bank_iteration']
            self.best=-1.  # Select only measured continuation checkpoints for its best.
        if not self.metrics:
            self.evaluate(state['iteration'],state)
        self.seen_step=state['iteration']
        self.segment_start_updates=self.resources['incremental_optimizer_updates']
        self.resources.setdefault('segments',[]).append(dict(
            device=self.device,gpu=torch.cuda.get_device_name(self.device) if str(self.device).startswith('cuda') else 'CPU',
            torch_version=torch.__version__,cuda_version=torch.version.cuda,
            activation_checkpointing=self.task['activation_checkpointing'],start_local_updates=self.seen_step))
        self.resources['deployment_parameters']=sum(p.numel() for m in (state['teacher_fea'],state['teacher_cls']) for p in m.parameters())

    def recover_best(self,iteration):
        for name in ('best.pth','best_previous.pth'):
            path=self.out/name
            if not path.exists():continue
            saved=torch.load(path,map_location='cpu',weights_only=False)
            matches=saved.get('iteration')==iteration
            del saved
            if matches:
                if name!='best.pth':
                    temporary=self.out/'best.recovered.pth'
                    shutil.copyfile(path,temporary)
                    temporary.replace(self.out/'best.pth')
                return
        raise ValueError('Best model does not match the committed checkpoint metrics;preserve this output for inspection')

    def image(self,path):
        size=tuple(self.task['evaluation_size'])
        with Image.open(path) as image:
            x=TF.to_tensor(image.convert('RGB').resize(size,Image.Resampling.BICUBIC))
        return TF.normalize(x,[.485,.456,.406],[.229,.224,.225]).unsqueeze(0).to(self.device)

    def logits(self,feature,classifier,x,size):
        return classifier(feature(x)[1],size)

    def scores(self,prob,student,previous,prior):
        confidence,labels=prob.max(1,keepdim=True)
        result={'confidence':confidence}
        if all(v is not None for v in self.stats.values()):
            h=-(prob*prob.clamp_min(1e-8).log()).sum(1,keepdim=True)
            d=(prob*(prob.clamp_min(1e-8).log()-student.clamp_min(1e-8).log())).sum(1,keepdim=True)
            h=self.trainer._normalize_uncertainty(h,self.stats['entropy_mean'],self.stats['entropy_var'],1e-6)
            d=self.trainer._normalize_uncertainty(d,self.stats['kl_mean'],self.stats['kl_var'],1e-6)
            result['UA']=torch.exp(-h-d)
        if previous is not None:
            label,conf=previous
            agreement=(labels==label.to(self.device)).float()*.5*(confidence+conf.to(self.device).float())
            result['T']=.5+.5*agreement
            p=prior.detach().float().clamp_min(1e-6)
            p=p/p.sum()
            weights=((1/8)/p).pow(.25).clamp(.7,1.3)
            result['TC']=result['T']*weights[labels]
        return result

    @torch.no_grad()
    def panel(self,state):
        feature,cls=state['feature_extractor'],state['classifier']
        teacher,teacher_cls=state['teacher_fea'],state['teacher_cls']
        files=sorted((Path(self.task['data'])/'train_val/images').glob('*.jpg'))
        rng=np.random.default_rng(17)
        selected=sorted(rng.choice(len(files),min(self.task['panel_images'],len(files)),replace=False).tolist())
        support=np.zeros(8,dtype='int64'); wrong=support.copy()
        drift=drift_pixels=0; raw_kl=0.; pixels=0
        for i in selected:
            x=self.image(files[i])
            t=self.logits(teacher,teacher_cls,x,x.shape[-2:]).softmax(1)
            s=self.logits(feature,cls,x,x.shape[-2:]).softmax(1)
            labels=t.argmax(1); mismatch=labels!=s.argmax(1)
            support+=torch.bincount(labels.flatten(),minlength=8).cpu().numpy()
            wrong+=torch.bincount(labels[mismatch],minlength=8).cpu().numpy()
            raw_kl+=float((t*(t.clamp_min(1e-8).log()-s.clamp_min(1e-8).log())).sum())
            pixels+=labels.numel()
            sampled=labels[:,::8,::8].cpu().to(torch.uint8)
            old=self.panel_history.get(files[i].name)
            if old is not None:
                drift+=int((old!=sampled).sum()); drift_pixels+=sampled.numel()
            self.panel_history[files[i].name]=sampled
        return dict(images=[files[i].name for i in selected], pixels=int(support.sum()),
                    mismatches=int(wrong.sum()), predicted_class_support=support.tolist(),
                    predicted_class_mismatch=wrong.tolist(), raw_kl=raw_kl/max(pixels,1),
                    teacher_drift=drift/max(drift_pixels,1) if drift_pixels else None,
                    teacher_drift_pixel_stride=8, manual_labels_used=False)

    @torch.no_grad()
    def evaluate(self,iteration,state):
        start=time.perf_counter()
        feature,cls=state['feature_extractor'],state['classifier']
        teacher,teacher_cls=state['teacher_fea'],state['teacher_cls']
        modules=[feature,cls,teacher,teacher_cls]
        modes=[m.training for m in modules]
        for module in modules: module.eval()
        rng_state=torch.get_rng_state()
        cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        from core.datasets.suim import SUIM_COLOR_TO_TRAIN_ID
        from metrics import quality
        matrix=np.zeros((8,8),dtype='int64'); boundaries=np.zeros((8,2),dtype='int64')
        images=[]; quality_rows=[]; latencies=[]
        root=Path(self.task['data'])/'TEST'
        try:
            for image in sorted((root/'images').glob('*.jpg')):
                masks=[p for ext in ('.bmp','.png') if (p:=root/'masks'/(image.stem+ext)).exists()]
                if len(masks)!=1: raise ValueError(f'Missing/ambiguous mask: {image.name}')
                with Image.open(masks[0]) as mask: y=np.asarray(mask).copy()
                if y.ndim==3:
                    target=np.full(y.shape[:2],255,dtype='int64')
                    for color,c in SUIM_COLOR_TO_TRAIN_ID.items(): target[(y[...,:3]==color).all(-1)]=c
                    y=target
                x=self.image(image)
                sync(self.device); tick=time.perf_counter()
                logits=self.logits(teacher,teacher_cls,x,y.shape)
                sync(self.device); latencies.append(time.perf_counter()-tick)
                t=logits.softmax(1)
                s=self.logits(feature,cls,x,y.shape).softmax(1)
                pred=t.argmax(1)[0].cpu().numpy()
                cm=confusion(pred,y); matrix+=cm
                bc=boundary_counts(pred,y); boundaries+=bc
                images.append(dict(image=image.name,confusion=cm.tolist(),boundary_counts=bc.tolist()))
                previous=self.shadow.get(image.name)
                scores=self.scores(t,s,previous,state['running_conf'])
                stride=self.task['quality_stride']
                valid=(y[::stride,::stride]>=0)&(y[::stride,::stride]<8)
                sampled_prediction=pred[::stride,::stride]
                correct=(sampled_prediction==y[::stride,::stride])
                for name,score in scores.items():
                    sampled=score[0,0,::stride,::stride].cpu().numpy()
                    measurement=quality(sampled[valid],correct[valid])
                    if measurement is None: continue
                    row=dict(image=image.name,score=name,**measurement)
                    row['by_predicted_class']={str(c):quality(sampled[valid&(sampled_prediction==c)],correct[valid&(sampled_prediction==c)]) for c in range(8)}
                    quality_rows.append(row)
                conf,label=t.max(1,keepdim=True)
                self.shadow[image.name]=(label.cpu().to(torch.uint8),conf.cpu().half())
            panel=self.panel(state)
        finally:
            for module,mode in zip(modules,modes): module.train(mode)
            torch.set_rng_state(rng_state)
            if cuda_state: torch.cuda.set_rng_state_all(cuda_state)
        b=np.divide(boundaries[:,0],boundaries[:,1],out=np.zeros(8,dtype=float),where=boundaries[:,1]>0)
        row=dict(local_updates=iteration,global_updates=self.phase['offset']+iteration,
                 phase=self.phase['phase'],**overlap(matrix),class_boundary_iou=b.tolist(),
                 boundary_iou=float(b.mean()),fg_boundary_iou=float(b[1:].mean()),
                 boundary_ratio=.02,boundary_counts=boundaries.tolist(),per_image=images,
                 quality=quality_rows,quality_estimand='mean_per_image;tie_group_expectation',
                 quality_pixel_stride=self.task['quality_stride'],temporal_diagnostic='causal_previous_eval_same_phase',
                 unlabeled_same_view_panel=panel,
                 ema_age_updates=(iteration-1)%12 if iteration else 0,
                 inference_ms_median=float(1000*np.median(latencies[5:] or latencies)),
                 inference_setup=dict(model='teacher',batch=1,input_size=self.task['evaluation_size'],
                                      output='original_mask_size',warmup_images_excluded=5,
                                      gpu=torch.cuda.get_device_name(self.device) if str(self.device).startswith('cuda') else 'CPU'),
                 source_only=iteration==0 and self.phase['phase']=='A',labeled_checkpoint_selection=True)
        self.metrics.append(row)
        self.resources['evaluation_seconds']+=time.perf_counter()-start
        if iteration>0 and row['miou']>self.best:
            self.best=row['miou']
            temporary=self.out/'best.partial.pth'
            if (self.out/'best.pth').exists():
                (self.out/'best.pth').replace(self.out/'best_previous.pth')
            torch.save(dict(iteration=iteration,global_updates=row['global_updates'],
                            feature_extractor=teacher.state_dict(),classifier=teacher_cls.state_dict(),
                            source_sha256=self.task['source']['sha256']),temporary)
            temporary.replace(self.out/'best.pth')
        print('REBUTTAL_EVAL',json.dumps({k:row[k] for k in ('local_updates','global_updates','miou','fg_miou','fg_boundary_iou')}),flush=True)

    def save(self,payload,filename):
        start=time.perf_counter()
        payload['experiment_control']=dict(metrics=self.metrics,shadow=self.shadow,panel_history=self.panel_history,
                                           stats={k:v.cpu() if torch.is_tensor(v) else v for k,v in self.stats.items()},
                                           bank_iteration=self.bank_iteration,best=self.best)
        payload['experiment_signature']=self.task['signature']
        path=Path(filename); path.parent.mkdir(exist_ok=True,parents=True)
        temporary=path.with_name(path.name+'.partial')
        torch.save(payload,temporary)
        temporary.replace(path)
        self.resources['checkpoint_seconds']+=time.perf_counter()-start

    def boundary(self,iteration,filename,state):
        start=time.perf_counter()
        banks=self.out/'banks'/f'{self.bank_iteration:06d}'
        if not banks.exists():
            temporary=banks.with_name(banks.name+'.partial')
            temporary.mkdir(parents=True,exist_ok=True)
            copy_banks(self.out,temporary)
            atomic_json(temporary/'BANKS.json',dict(sha256=bank_hash(temporary)))
            temporary.replace(banks)
        banks_sha=read_json(banks/'BANKS.json')['sha256']
        checkpoint=Path(filename)
        completed=iteration>=self.phase['steps']
        pause=not completed and (time.monotonic()>=self.deadline or (self.task['smoke'] and iteration==1))
        old=read_json(self.out/'CURRENT.json') if (self.out/'CURRENT.json').exists() else None
        checkpoint_sha=digest(checkpoint)
        self.resources['checkpoint_seconds']+=time.perf_counter()-start
        now=time.monotonic()
        self.resources['wall_seconds']+=now-self.last_wall
        self.last_wall=now
        if str(self.device).startswith('cuda'):
            self.resources['peak_allocated_bytes']=max(self.resources.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated(self.device))
            self.resources['peak_reserved_bytes']=max(self.resources.get('peak_reserved_bytes',0),torch.cuda.max_memory_reserved(self.device))
            self.resources['gpu']=torch.cuda.get_device_name(self.device)
        self.resources['segments'][-1].update(wall_seconds=now-self.started,
                incremental_optimizer_updates=self.resources['incremental_optimizer_updates']-self.segment_start_updates)
        pointer=dict(iteration=iteration,checkpoint=checkpoint.name,sha256=checkpoint_sha,
                     banks=str(banks.relative_to(self.out)),banks_sha256=banks_sha,
                     signature=self.task['signature'],resources=self.resources)
        atomic_json(self.out/'CURRENT.json',pointer)
        anchors=(1,2,3) if self.task['smoke'] else (20000,25000,27500)
        if self.phase['phase']=='A' and iteration in anchors:
            snapshots=self.out/'snapshots'; snapshots.mkdir(exist_ok=True)
            atomic_json(snapshots/f'{iteration:06d}.json',pointer)
        atomic_json(self.out/'metrics.json',self.metrics)
        atomic_json(self.out/'resources.json',self.resources)
        status='complete' if completed else 'session_paused' if pause else 'training'
        atomic_json(self.out/'RUN_STATE.json',dict(status=status,local_updates=iteration,
                    global_updates=self.phase['offset']+iteration,checkpoint_verified=True,
                    full_state=True,stopping_rule='fixed_budget',diagnostic=self.task['smoke']))
        if completed: atomic_json(self.out/'complete.json',dict(local_updates=iteration,signature=self.task['signature']))
        if old and old['checkpoint']!=checkpoint.name:
            retained={read_json(p)['checkpoint'] for p in (self.out/'snapshots').glob('*.json')}
            previous=self.out/old['checkpoint']
            if previous.name not in retained and previous.parent.resolve()==self.out.resolve():
                previous.unlink(missing_ok=True)
        if pause: raise SessionPaused()
