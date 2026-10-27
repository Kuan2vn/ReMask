"""Trace native resume preparation without changing native files or state checks."""
import argparse
import faulthandler
import functools
import json
from pathlib import Path
import sys
import time


def traced(label, function, interval=60):
    @functools.wraps(function)
    def call(*args, **kwargs):
        started = time.monotonic()
        print(f'[PREPARE START] {label}', flush=True)
        # Diagnostics only: prints the waiting stack, never interrupts the operation.
        faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stderr)
        try:
            result = function(*args, **kwargs)
        except BaseException:
            print(f'[PREPARE FAILED] {label}; elapsed={time.monotonic()-started:.1f}s', flush=True)
            raise
        else:
            print(f'[PREPARE DONE] {label}; elapsed={time.monotonic()-started:.1f}s', flush=True)
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()
    return call


def install(run, repo=None):
    run.dataset_signature = traced('hash local dataset', run.dataset_signature)
    original_verify = run.verified_pointer

    def verify(folder, name='CURRENT.json'):
        path = Path(folder)/name
        pointer = json.loads(path.read_text(encoding='utf-8'))
        print(f'[CHECKPOINT CANDIDATE - NOT YET VERIFIED] {path}; '
              f'update={pointer["iteration"]}; checkpoint={pointer["checkpoint"]}; '
              f'banks={pointer["banks"]}', flush=True)
        return traced(f'verify checkpoint and immutable banks: {path}', original_verify)(folder, name)
    run.verified_pointer = verify
    original_copy = run.copy_banks

    def copy(source, destination):
        from local_bank_io import mutable_root
        return traced(f'restore LOCAL working banks: {source} -> {mutable_root(destination)}', original_copy)(source, destination)
    run.copy_banks = copy
    original_phase = run.phase_run

    def phase(args, arm, protocol, recipe, *rest):
        folder = args.output/protocol/f'seed{args.seed}'/arm/recipe['phase']
        if repo is not None:
            from local_bank_io import prepare_phase
            prepare_phase(folder, fresh=not (folder/'CURRENT.json').exists())
        print(f'[PHASE] {protocol}/{arm}/{recipe["phase"]}; output={folder}', flush=True)
        if (folder/'CURRENT.json').exists():
            print('[RESUME] Saved own-state pointer found; full verification follows.', flush=True)
        elif recipe['parent']:
            print(f'[FORK] Requires {recipe["parent"]}/A@{recipe["parent_step"]}; initializes working bank from verified DTST warm bank.', flush=True)
        else:
            print('[FRESH] No own-state pointer; native runner will initialize from source with verified DTST warm candidate bank.', flush=True)
        return original_phase(args, arm, protocol, recipe, *rest)
    run.phase_run = phase
    original_popen = run.subprocess.Popen

    def popen(command, *args, **kwargs):
        worker = any(Path(str(value)).name == 'worker.py' for value in command) if isinstance(command, (list, tuple)) else False
        if worker:
            print('[WORKER LAUNCH] Preparation finished. Launch is not yet optimizer progress.', flush=True)
            if repo is not None:
                task = command[command.index('--task')+1]
                command = [command[0], '-u', str(Path(__file__).with_name('performance_worker.py')),
                           '--repo', str(repo), '--task', task]
        process = original_popen(command, *args, **kwargs)
        if worker:
            print(f'[WORKER PID] {process.pid}; watch new optimizer log lines.', flush=True)
        return process
    run.subprocess.Popen = popen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--existing-dependencies', action='store_true')
    parser.add_argument('--reference-output', type=Path)
    parser.add_argument('--continuation-output', type=Path)
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(args.repo/'experiments/rebuttal'))
    import run
    from recipes import ordered_arms, phases, PROTOCOLS
    from local_bank_io import install_coordinator
    from util import paper_stage_a_checkpoint
    install_coordinator(run)
    requested = remaining[remaining.index('--arm')+1] if '--arm' in remaining else None
    suite = remaining[remaining.index('--suite')+1] if '--suite' in remaining else 'priority'
    print('[SCHEDULE]', ' -> '.join(ordered_arms(suite, requested)), flush=True)
    if args.existing_dependencies and requested:
        def value(option, default):
            return remaining[remaining.index(option)+1] if option in remaining else default
        output = Path(value('--output', str(args.repo/'outputs/rebuttal')))
        mode = 'smoke' if '--smoke' in remaining else 'full'
        seed = int(value('--seed', '1'))
        protocol = value('--protocol', 'all')
        protocols = PROTOCOLS if protocol == 'all' else (protocol,)
        missing = []
        for protocol in protocols:
            base = output/mode/protocol/f'seed{seed}'
            for recipe in phases(requested, protocol, seed, mode=='smoke'):
                parent = recipe['parent']
                if parent and parent != requested and not (base/requested/recipe['phase']/'CURRENT.json').exists():
                    snapshot = base/parent/'A/snapshots'/f"{recipe['parent_step']:06d}.json"
                    if not snapshot.exists():
                        reference = args.reference_output if parent == 'F' else args.continuation_output
                        if reference is not None and reference.resolve() != output.resolve():
                            source = reference/mode/protocol/f'seed{seed}'/parent/'A'
                            if (source/'snapshots'/snapshot.name).exists():
                                from anchor_import import import_anchor
                                traced(f'import immutable {parent}/A@{recipe["parent_step"]}', import_anchor)(
                                    source, snapshot.parent.parent, recipe['parent_step'], protocol, seed, mode=='smoke')
                    if not snapshot.exists() and parent == 'F' and recipe['parent_step'] == 25000 and mode != 'smoke':
                        if paper_stage_a_checkpoint(protocol, repo=args.repo) is not None:
                            print(f'[PAPER STAGE-A READY] Will seed {protocol}/F/A@25000 from remask_model_iter024999.pth', flush=True)
                            continue
                    if not snapshot.exists():
                        missing.append(snapshot)
        if missing:
            for path in missing:
                print('[WAITING_DEPENDENCY]', path, flush=True)
            print('[NO TRAINING STARTED] Run F explicitly to continue F; return to this arm after its anchors exist.', flush=True)
            return
        # Own state and parent hashes still pass through unmodified native checks.
        run.ordered_arms = lambda suite, arm: [arm]
        print('[SCHEDULE] Requested arm only; own resume or existing parent anchors, verified by native runner.', flush=True)
    else:
        print('[SCHEDULE] Native dependency order; unfinished dependencies run first.', flush=True)
    install(run, args.repo)
    sys.argv = [str(args.repo/'experiments/rebuttal/run.py'), *remaining]
    run.main()


if __name__ == '__main__':
    main()
