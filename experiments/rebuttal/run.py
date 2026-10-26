"""Run the declared ReMask contrasts with dependency reuse and durable Colab resume."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'baselines/underwater'))
from recipes import ARMS,PROTOCOLS,ordered_arms,phases
from util import atomic_json,copy_banks,digest,read_json,tree_hash,verified_pointer


def dataset_signature(data):
    value=hashlib.sha256()
    for relative in ('train_val/images','TEST/images','TEST/masks'):
        for path in sorted((Path(data)/relative).iterdir()):
            if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.bmp','.png'):
                value.update(path.relative_to(data).as_posix().encode()); value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


@contextlib.contextmanager
def local_lock(output):
    import fcntl
    name=hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir())/f'remask-rebuttal-{name}.lock').open('a+b') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Another coordinator owns this output on this runtime') from error
        yield


def phase_run(args,arm,protocol,p,source,data,data_sha,code_sha,calib,deadline):
    from underwater_baselines.launch import data_view
    base=args.output/protocol/f'seed{args.seed}'
    output=base/arm/p['phase']; output.mkdir(parents=True,exist_ok=True)
    signature=dict(arm=arm,protocol=protocol,phase=p,source_sha256=source['sha256'],dataset_sha256=data_sha,
                   code_sha256=code_sha,diagnostic=args.smoke,activation_checkpointing=not args.no_activation_checkpointing,
                   panel_images=args.panel_images,quality_stride=args.quality_stride,smoke_refresh=args.smoke_refresh,
                   device=args.device)
    config=output/'PHASE_CONFIG.json'
    if config.exists() and read_json(config)!=signature:
        raise ValueError(f'Output belongs to a different recipe/code/data: {output}; choose a new --output')
    atomic_json(config,signature)
    if (output/'complete.json').exists():
        if read_json(output/'complete.json')['signature']!=signature: raise ValueError('Completion signature mismatch')
        print(f'Complete: {protocol}/{arm}/{p["phase"]}',flush=True)
        return True
    # Drive does not support dataset symlinks; only durable artifacts live there.
    runtime_key=hashlib.sha256((str(output.resolve())+str(data.resolve())).encode()).hexdigest()[:24]
    runtime=Path(tempfile.gettempdir())/'remask-rebuttal-runtime'/runtime_key
    runtime.mkdir(parents=True,exist_ok=True)
    view=data_view(data,runtime,args.smoke)
    own=(output/'CURRENT.json').exists()
    checkpoint=Path(source['path'])
    if own:
        pointer,checkpoint,banks=verified_pointer(output)
        if pointer['signature']!=signature: raise ValueError('Saved-state signature mismatch')
        copy_banks(banks,output)
    elif p['parent']:
        parent=base/p['parent']/'A'
        pointer,checkpoint,banks=verified_pointer(parent,f'snapshots/{p["parent_step"]:06d}.json')
        parent_signature=pointer['signature']
        if any(parent_signature[k]!=signature[k] for k in ('source_sha256','dataset_sha256','code_sha256','diagnostic')):
            raise ValueError('Parent checkpoint does not match source/data/code/mode')
        copy_banks(banks,output)
    seconds=max(0.,deadline-time.monotonic())
    if seconds<=0: return False
    task=dict(arm=arm,protocol=protocol,phase=p,source=source,signature=signature,output=str(output.resolve()),
              data=str(view.resolve()),initial_checkpoint=str(checkpoint.resolve()),resume_own=own,
              seed=args.seed,smoke=args.smoke,smoke_refresh=args.smoke_refresh,device=args.device,seconds=seconds,calibration=calib,
              activation_checkpointing=not args.no_activation_checkpointing,
              evaluation_size=[128,96] if args.smoke else [1280,640],
              panel_images=min(args.panel_images,2) if args.smoke else args.panel_images,
              quality_stride=2 if args.smoke else args.quality_stride)
    atomic_json(output/'task.json',task)
    command=[sys.executable,'-u',str(HERE/'worker.py'),'--task',str(output/'task.json')]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for _ in range(3 if args.smoke else 1):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=runtime/'.runtime',env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            process.stdout.close()
            if process.wait(): raise RuntimeError(f'Training failed; preserved output: {output}')
        state=read_json(output/'RUN_STATE.json')
        if state['status']=='complete': return True
        if state['status']!='session_paused': raise RuntimeError(f'Unexpected trainer state: {state}')
        if not args.smoke: return False
        pointer,checkpoint,banks=verified_pointer(output)
        task.update(resume_own=True,initial_checkpoint=str(checkpoint),seconds=max(0.,deadline-time.monotonic()))
        atomic_json(output/'task.json',task)
        copy_banks(banks,output)
    raise RuntimeError('Smoke did not complete after clean process resumes')


def main():
    parser=argparse.ArgumentParser(description='ReMask UA/TC/UWRG ablations; all budgets are real optimizer updates')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--suite',choices=['priority','core','all'],default='priority')
    group.add_argument('--arm',choices=list(ARMS))
    parser.add_argument('--protocol',choices=(*PROTOCOLS,'all'),default='all')
    parser.add_argument('--data',type=Path,default=REPO/'data/SUIM')
    parser.add_argument('--output',type=Path,default=REPO/'outputs/rebuttal')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--session-hours',type=float,default=9.)
    parser.add_argument('--panel-images',type=int,default=32)
    parser.add_argument('--quality-stride',type=int,default=8)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--no-activation-checkpointing',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='8 train/2 eval images;2+2 updates;real process restarts;separate outputs')
    parser.add_argument('--smoke-refresh',action='store_true',help='With --smoke only,also exercise a pseudo-label refresh before pause')
    parser.add_argument('--plan',action='store_true',help='Print resolved phases without downloads or GPU')
    parser.add_argument('--check',action='store_true',help='Verify dependencies/data/source hashes and strict CPU source load')
    args=parser.parse_args()
    if args.smoke_refresh and not args.smoke:parser.error('--smoke-refresh requires --smoke')
    if args.panel_images<1 or args.quality_stride<1 or not 0<args.session_hours<1000: parser.error('Invalid panel/stride/session budget')
    protocols=PROTOCOLS if args.protocol=='all' else (args.protocol,)
    arms=ordered_arms(args.suite,args.arm)
    plan=[dict(protocol=protocol,arm=arm,phases=phases(arm,protocol,args.seed,args.smoke)) for arm in arms for protocol in protocols]
    if args.plan:
        print(json.dumps(plan,indent=2));return
    if args.device=='cpu' and not (args.check or args.smoke): parser.error('Full training requires CUDA;CPU is for check/smoke only')
    import torch
    from underwater_baselines.assets import source_checkpoint
    from underwater_baselines.launch import validate_data
    from underwater_baselines.common import Segmenter
    data=validate_data(args.data,args.smoke)
    if not args.check:
        if sys.platform!='linux' or int(os.environ.get('WORLD_SIZE','1'))!=1: parser.error('Use a single Linux process/GPU')
        if args.device=='cuda':
            if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
            free,total=torch.cuda.mem_get_info()
            print('GPU:',torch.cuda.get_device_name(),'; free GiB:',round(free/1024**3,2),flush=True)
            if not args.smoke and free<12*1024**3: raise RuntimeError('At least12GiB free VRAM required;full-recipe capacity still depends on GPU/runtime')
    deadline=time.monotonic()+args.session_hours*3600
    records={protocol:source_checkpoint(protocol,REPO/'assets/baselines') for protocol in protocols}
    if args.check:
        torch.set_num_threads(4)
        for protocol,record in records.items():
            model=Segmenter(record['path']).eval()
            with torch.no_grad(): result=model(torch.zeros(1,3,32,32))
            if result.shape!=(1,8,32,32) or not torch.isfinite(result).all():raise RuntimeError('Source forward failed')
            print('CHECK PASSED:',protocol,record['sha256']);del model
        return
    args.output=args.output.resolve()/('smoke' if args.smoke else 'full')
    args.output.mkdir(parents=True,exist_ok=True)
    data_sha=dataset_signature(data)
    code_sha=tree_hash(HERE)
    # Calibrate before workers; only original UW helper is imported, never its main().
    sys.path.insert(0,str(HERE/'native'))
    import trainer
    from probes import calibration
    with local_lock(args.output):
        for row in plan:
            protocol,arm=row['protocol'],row['arm']
            strength=.2 if protocol==PROTOCOLS[0] else .35
            calib=calibration(args.output/'calibration'/f'{protocol}.json',data,data_sha,strength,
                              trainer._underwater_reliability_perturb,args.smoke)
            for p in row['phases']:
                if not phase_run(args,arm,protocol,p,records[protocol],data,data_sha,code_sha,calib,deadline):
                    print('SESSION_PAUSED: repeat the same command/output to continue the declared queue.',flush=True);return
        print('SUITE_COMPLETE: every requested phase has a committed completion record.',flush=True)


if __name__=='__main__': main()
