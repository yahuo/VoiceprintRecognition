"""Synthetic local protocol peer; never imports or downloads model packages."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import moss
sys.modules['moss'] = moss
from app.services.moss_worker import RequestCommands

commands = RequestCommands(sys.stdin)
protocol = os.fdopen(int(os.environ['MOSS_RESULT_FD']), 'w', buffering=1)

def send(value):
    protocol.write(json.dumps(value) + '\n')

time.sleep(float(os.environ.get('FAKE_MOSS_START_DELAY', '0')))
send({'ready': True})
was_cancelled = False
while True:
    item = commands.requests.get()
    if item is None:
        break
    if isinstance(item, Exception):
        raise item
    request, cancelled = item
    assert os.path.exists(request['audio_path'])
    mode = 'ok' if was_cancelled else os.environ.get('FAKE_MOSS_MODE', 'ok')
    if mode.startswith('stream'):
        prefix = '[0][S01]synthetic[1]\n[1][S02]'
        send({'type': 'delta', 'text': prefix})
        if mode == 'stream_timeout':
            for _ in range(200):
                if cancelled.wait(.02):
                    break
                send({'type': 'delta', 'text': ' '})
        else:
            gate = Path(request['audio_path']).with_suffix('.continue')
            deadline = time.monotonic() + 3
            while not gate.exists() and not cancelled.is_set() and time.monotonic() < deadline:
                time.sleep(.005)
            if not cancelled.is_set():
                assert gate.exists(), 'real segment callback did not arrive before completion'
                send({'type': 'delta', 'text': 'tail[2]'})
                raw = prefix + ('changed[2]' if mode == 'stream_changed' else 'tail[2]')
                send({'type': 'result', 'raw': raw,
                      'finish_reason': 'length' if mode == 'stream_truncated' else 'stop',
                      'eos_reached': mode != 'stream_truncated'})
    elif mode == 'timeout':
        cancelled.wait(30)
    elif mode == 'cancel_hang':
        time.sleep(30)
    elif mode == 'exit':
        sys.exit(7)
    elif mode == 'malformed':
        protocol.write('not json\n')
    elif mode == 'partial':
        send({'raw': '[0][S01]text[1]', 'finish_reason': 'length', 'eos_reached': False})
    else:
        send({'type': 'result', 'raw': '[0][S01]synthetic[1]', 'finish_reason': 'stop', 'eos_reached': True})
    if cancelled.is_set():
        send({'type': 'cancelled'})
        was_cancelled = True
    commands.finish(request['request_id'])
