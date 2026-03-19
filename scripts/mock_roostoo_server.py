#!/usr/bin/env python3
"""
Lightweight mock Roostoo API server for local testing.

Endpoints implemented (minimal):
 - GET /v3/exchangeInfo
 - GET /v3/ticker
 - GET /v3/balance
 - POST /v3/place_order
 - POST /v3/query_order
 - POST /v3/cancel_order

Run:
    python scripts/mock_roostoo_server.py --host 127.0.0.1 --port 8000

The server is stateful in-memory and intended for local simulation only.
"""
import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import time

ORDERS = {}
NEXT_ORDER_ID = 1
LOCK = threading.Lock()


def _now_ms():
    return int(time.time() * 1000)


def _make_commission(side: str, coin: str, qty: float, price: float, role: str):
    if role == 'TAKER':
        pct = 0.0012
    else:
        pct = 0.0008
    if side.upper() == 'BUY':
        coin_c = coin
        charge = qty * pct
    else:
        coin_c = 'USD'
        charge = qty * price * pct
    return coin_c, float(charge), pct


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def finish(self):
        try:
            super().finish()
        finally:
            # Force connection teardown so a keep-alive probe cannot monopolize
            # the single mock server worker and block subsequent clients.
            self.close_connection = True

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == '/v3/exchangeInfo':
            payload = {
                'IsRunning': True,
                'InitialWallet': {'USD': 50000},
                'TradePairs': {
                    'BTC/USD': {'Coin': 'BTC', 'CoinFullName': 'Bitcoin', 'Unit': 'USD', 'UnitFullName': 'US Dollar', 'CanTrade': True, 'PricePrecision': 2, 'AmountPrecision': 6, 'MiniOrder': 1},
                    'ETH/USD': {'Coin': 'ETH', 'CoinFullName': 'Ethereum', 'Unit': 'USD', 'UnitFullName': 'US Dollar', 'CanTrade': True, 'PricePrecision': 2, 'AmountPrecision': 4, 'MiniOrder': 1},
                    'XRP/USD': {'Coin': 'XRP', 'CoinFullName': 'XRP', 'Unit': 'USD', 'UnitFullName': 'US Dollar', 'CanTrade': True, 'PricePrecision': 4, 'AmountPrecision': 1, 'MiniOrder': 1},
                }
            }
            self._send_json(payload)
            return

        if path == '/v3/ticker':
            data = {'MaxBid': 100.0, 'MinAsk': 101.0, 'LastPrice': 100.5}
            self._send_json({'Success': True, 'Data': data})
            return

        if path == '/v3/balance':
            self._send_json({'Success': True, 'SpotWallet': {'USD': {'Free': 50000, 'Lock': 0}}})
            return

        self._send_json({'error': 'not_found'}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else ''
        params = parse_qs(body)

        # POST handlers
        if path == '/v3/place_order':
            pair = params.get('pair', [''])[0]
            side = params.get('side', ['BUY'])[0]
            typ = params.get('type', ['MARKET'])[0]
            try:
                qty = float(params.get('quantity', ['0'])[0])
            except Exception:
                qty = 0.0
            try:
                price = float(params.get('price', ['0'])[0]) if 'price' in params else 0.0
            except Exception:
                price = 0.0

            coin = pair.split('/')[0] if '/' in pair else pair

            global NEXT_ORDER_ID
            with LOCK:
                oid = NEXT_ORDER_ID
                NEXT_ORDER_ID += 1

            # decide mode
            mode_roll = random.random()
            role = 'TAKER' if typ == 'MARKET' else 'MAKER'
            if mode_roll < 0.4 or typ == 'MARKET':
                status = 'FILLED'
                filled = qty
            else:
                status = 'PENDING'
                filled = round(qty * random.uniform(0.0, 0.4), 8)

            comm_coin, comm_value, comm_pct = _make_commission(side, coin, filled, price or 0.0, role)

            order = {
                'OrderID': oid,
                'Status': status,
                'Role': role,
                'Pair': pair,
                'Side': side,
                'Type': typ,
                'Price': price,
                'Quantity': qty,
                'FilledQuantity': filled,
                'FilledAverPrice': price if price else 0.0,
                'CommissionCoin': comm_coin,
                'CommissionChargeValue': comm_value,
                'CommissionPercent': comm_pct,
                'CreateTimestamp': _now_ms(),
                'FinishTimestamp': _now_ms() if status == 'FILLED' else 0,
            }

            if status == 'PENDING':
                with LOCK:
                    ORDERS[str(oid)] = order

            self._send_json({'Success': True, 'OrderDetail': order})
            return

        if path == '/v3/query_order':
            # If order_id provided, return that; otherwise return OrderMatched for all pending orders
            order_id = params.get('order_id', [None])[0]
            matched = []
            with LOCK:
                if order_id:
                    o = ORDERS.get(str(order_id))
                    if o:
                        matched.append(o)
                else:
                    # advance state for each pending order
                    for k, o in list(ORDERS.items()):
                        remaining = o['Quantity'] - o['FilledQuantity']
                        if remaining <= 1e-8:
                            o['Status'] = 'FILLED'
                            o['FilledQuantity'] = o['Quantity']
                            o['FinishTimestamp'] = _now_ms()
                            matched.append(o)
                            del ORDERS[k]
                        else:
                            step = round(min(remaining, o['Quantity'] * random.uniform(0.08, 0.45)), 8)
                            o['FilledQuantity'] = round(o['FilledQuantity'] + step, 8)
                            if o['FilledQuantity'] >= o['Quantity'] - 1e-8:
                                o['Status'] = 'FILLED'
                                o['FilledQuantity'] = o['Quantity']
                                o['FinishTimestamp'] = _now_ms()
                                matched.append(o)
                                del ORDERS[k]
                            else:
                                o['Status'] = 'PENDING'
                                matched.append(o)

            self._send_json({'Success': True, 'ErrMsg': '', 'OrderMatched': matched})
            return

        if path == '/v3/cancel_order':
            order_id = params.get('order_id', [None])[0]
            if order_id and str(order_id) in ORDERS:
                with LOCK:
                    ORDERS[str(order_id)]['Status'] = 'CANCELED'
                    ORDERS[str(order_id)]['FinishTimestamp'] = _now_ms()
                self._send_json({'Success': True, 'OrderID': order_id, 'Status': 'CANCELED'})
                return
            self._send_json({'Success': False, 'ErrMsg': 'order_not_found'}, status=404)
            return

        self._send_json({'error': 'not_found'}, status=404)


def run_server(host='127.0.0.1', port=8000):
    srv = ThreadingHTTPServer((host, port), MockHandler)
    print(f"Mock Roostoo server listening at http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('Shutting down')
        srv.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    run_server(args.host, args.port)



