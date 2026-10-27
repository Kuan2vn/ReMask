"""One isolated native training phase. Started by run.py, never by a Slurm queue."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'native'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    args=parser.parse_args()
    task=json.loads(Path(args.task).read_text())
    import torch
    torch.set_num_threads(4)
    import trainer
    from core.configs import cfg
    from core.utils.logger import setup_logger
    from control import Control,SessionPaused
    from probes import install_probe,checkpoint_residual_blocks
    from util import atomic_json
    p=task['phase']
    cfg.MODEL.NAME='deeplab_resnet101'; cfg.MODEL.NUM_CLASSES=8
    cfg.MODEL.DEVICE=task['device']; cfg.MODEL.FREEZE_BN=True; cfg.MODEL.WEIGHTS=''
    cfg.DATASETS.SOURCE_TRAIN='uiis_train' if task['protocol']=='uiis_to_suim' else 'usis10k_bg_train'
    cfg.DATASETS.TARGET_TRAIN='suim_train'; cfg.DATASETS.TEST='suim_val'
    cfg.DATASETS.TT='suim_TT_train'; cfg.DATASETS.II='suim_II_train'
    cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN=(96,64) if task['smoke'] else (960,480)
    cfg.INPUT.TARGET_INPUT_SIZE_TRAIN=cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN
    cfg.INPUT.INPUT_SIZE_TEST=tuple(task['evaluation_size'])
    cfg.INPUT.INPUT_SCALES_TRAIN=(.75,1.25)
    cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN=.5
    cfg.INPUT.BRIGHTNESS=.5; cfg.INPUT.CONTRAST=.5; cfg.INPUT.SATURATION=.5; cfg.INPUT.HUE=.2
    cfg.DTU.DYNAMIC=False; cfg.DTU.EMA_WEIGHT=.99; cfg.DTU.FIX_ITERATION=12
    cfg.TCR.OPEN=True; cfg.TCR.TOPK_CANDIDATE=.5
    cfg.TCR.UPDATE_FREQUENCY=1 if task.get('smoke_refresh') else 3000
    cfg.METAPL.OPEN=False  # ReMask trainer does not perform SND's bi-level denoising.
    cfg.SOLVER.BATCH_SIZE=4; cfg.SOLVER.MAX_ITER=p['horizon']; cfg.SOLVER.STOP_ITER=p['steps']
    cfg.SOLVER.BASE_LR=p['lr']; cfg.SOLVER.CHECKPOINT_PERIOD=1 if task['smoke'] else 500
    cfg.SEED=task['seed']; cfg.OUTPUT_DIR=task['output']; cfg.resume=task['initial_checkpoint']
    cfg.REMASK.ALPHA=p['alpha']; cfg.REMASK.BETA=p['beta']; cfg.REMASK.TAU=1.
    cfg.REMASK.USE_RELIABILITY=p['reliability']=='ua' or p['temporal']
    cfg.REMASK.RELIABILITY_MODE='entropy_kl' if p['reliability']=='ua' else 'temporal_label'
    cfg.REMASK.FUSE_MODE='none'; cfg.REMASK.USE_CLASS_BALANCE=p['classes']
    cfg.REMASK.CLASS_BALANCE_GAMMA=.25; cfg.REMASK.CLASS_BALANCE_MIN=.7; cfg.REMASK.CLASS_BALANCE_MAX=1.3
    cfg.REMASK.USE_CONTRASTIVE=p['contrastive']>0; cfg.REMASK.LAMBDA=p['contrastive']
    cfg.REMASK.HISTORY_UPDATE_MODE='online'; cfg.REMASK.HISTORY_UPDATE_PERIOD=1
    cfg.REMASK.HISTORY_QUEUE_SIZE=1; cfg.REMASK.HISTORY_STRIDE=1; cfg.REMASK.HISTORY_USE_CONF=True
    cfg.REMASK.HISTORY_MOMENTUM=.95; cfg.REMASK.TEMPORAL_CONF_POWER=0.; cfg.REMASK.TEMPORAL_FLOOR=.5
    cfg.TEST.BATCH_SIZE=1
    cfg.freeze()
    trainer._seed_everything(task['seed'])
    torch.backends.cudnn.benchmark=False
    if task['activation_checkpointing']: checkpoint_residual_blocks()
    install_probe(trainer,p,task['calibration']['generic_amplitude'])
    control=Control(task,trainer); trainer.EXPERIMENT=control
    original_candidate=trainer.run_candidate
    trainer.run_candidate=lambda *a,**kw:control.candidate(original_candidate,*a,**kw)
    if task['smoke']:
        # Small integration runs retain the native copy-paste pipeline.
        original_build=trainer._build_target_loader
        def build(*a,**kw):
            data,loader=original_build(*a,**kw)
            loader=torch.utils.data.DataLoader(data,batch_size=4,shuffle=True,num_workers=0,drop_last=True)
            return data,loader
        trainer._build_target_loader=build
    resume_iter=-1 if task.get('resume_own') else (p['parent_step'] if p['full_resume'] else 0)
    options=SimpleNamespace(resume_iteration=resume_iter,
        early_stop_min_miou=-1.,early_stop_start_iter=0,early_stop_patience_iters=0,early_stop_min_delta=0.,
        use_uwrg=p['probe']!='none',uwrg_mode='pixel' if p['aggregation']=='pixel' else 'segment',
        uwrg_composition='weighted_product',uwrg_floor=p['floor'],uwrg_lambda=p['exponent'],
        uwrg_strength=p['strength'],uwrg_warmup_ratio=.2,uwrg_segment_stride=4,uwrg_min_segment_pixels=8,
        uctr_lambda=.1,uctr_tau_low=.6,uctr_tau_high=.9)
    setup_logger('DTST',task['output'],0)
    atomic_json(Path(task['output'])/'RUN_STATE.json',dict(status='initializing',diagnostic=task['smoke']))
    try:
        trainer.train(cfg,0,False,uwrg_options=options)
    except SessionPaused:
        print('Clean checkpoint pause; repeat the coordinator command to continue.',flush=True)


if __name__=='__main__': main()
