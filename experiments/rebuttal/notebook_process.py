"""Stream one notebook command and stop its own process group on interruption."""
import os
from pathlib import Path
import signal
import subprocess
import time


def stop_group(process):
    # start_new_session below makes this group exclusive to this notebook command.
    for sig, grace in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+grace
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            continue
        break
    process.wait(timeout=5)


def stream_command(command, cwd, log, env=None, on_line=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Log:', log, flush=True)
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
                if on_line is not None:
                    on_line(line)
            if process.wait():
                raise RuntimeError(f'Subprocess failed ({process.returncode}); see {log}')
        except BaseException:
            print('[CELL STOP] Stopping this command and its child workers. '
                  'Resume uses the latest committed checkpoint.', flush=True)
            stop_group(process)
            raise
        finally:
            process.stdout.close()
