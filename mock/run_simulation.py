#!/usr/bin/env python3
"""
Orchestrator to run the HTTP mock server and the HTTP-driven fake bot together.

Usage:
  python mock/run_simulation.py --host 127.0.0.1 --port 9000 --ticks 12 --seed 42

It will:
 - start mock/mock_roostoo_server.py as a background process
 - wait until /v3/exchangeInfo responds (with timeout)
 - run mock/http_fake_bot.py pointing at the mock server
 - terminate the mock server when done
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def wait_for_server(base_url: str, timeout: float = 8.0) -> bool:
    import requests

    start = time.time()
    url = f"{base_url}/v3/exchangeInfo"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                return True
        except Exception:
            time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--ticks', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    server_py = ROOT / 'mock_roostoo_server.py'
    bot_py = ROOT / 'http_fake_bot.py'
    host = args.host
    port = args.port
    base_url = f'http://{host}:{port}'

    # Start server
    env = dict(**sys.environ) if hasattr(sys, 'environ') else None
    # Ensure project root is importable for subprocesses
    project_root = str(Path(__file__).resolve().parent.parent)
    if env is None:
        env = dict(PATH=sys.executable)
    env['PYTHONPATH'] = project_root + (':' + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
    proc = subprocess.Popen([sys.executable, str(server_py), '--host', host, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    print('Started mock server, pid=', proc.pid)

    try:
        ready = wait_for_server(base_url, timeout=8.0)
        if not ready:
            print('Mock server did not become ready in time')
            proc.terminate()
            proc.wait(timeout=2)
            return

        # Run the HTTP fake bot
        cmd = [sys.executable, str(bot_py), '--mock-url', base_url, '--ticks', str(args.ticks), '--seed', str(args.seed)]
        ret = subprocess.run(cmd, env=env)
        print('Fake bot exit code:', ret.returncode)

    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


if __name__ == '__main__':
    main()


