#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite that starts the mock Roostoo server and initializes Portfolio against it.

Run: python test/test_portfolio_init_with_mock.py
"""
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.api.roostoo import Roostoo
from bot.portfolio.portfolio import Portfolio


def write_temp_config(base_url: str, cfg_path: Path):
    data = {
        'roostoo': {
            'base_url': base_url,
            'api_key': 'test',
            'secret_key': 'test'
        }
    }
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)


def main():
    server_py = ROOT / 'scripts' / 'mock_roostoo_server.py'
    host = '127.0.0.1'
    port = 9000
    base_url = f'http://{host}:{port}'

    # Start mock server
    proc = subprocess.Popen([sys.executable, str(server_py), '--host', host, '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('Started mock server, pid=', proc.pid)

    # prepare temporary config
    cfg_dir = ROOT / 'bot' / 'config'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / 'roostoo.mock.yaml'
    write_temp_config(base_url, cfg_path)

    try:
        # wait until the server responds
        import requests

        timeout = 5.0
        start = time.time()
        url = f"{base_url}/v3/exchangeInfo"
        ready = False
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=1.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)

        if not ready:
            print('Mock server did not become ready in time')
            return

        # instantiate client with mock config path
        client = Roostoo(config_path=cfg_path)
        print('Calling get_exchange_info()...')
        info = client.get_exchange_info()
        print('exchange_info:', info)

        p = Portfolio(execution_module=None)
        ok = p.initialize_from_exchange_info(roostoo_client=client, exchange_info=info, fallback_balance=100.0)
        print('init ok:', ok)
        print('balance:', p.account_balance)
        print('tradable count:', len(p.tradable_pairs))

    finally:
        # cleanup
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
        if cfg_path.exists():
            cfg_path.unlink()


if __name__ == '__main__':
    main()


