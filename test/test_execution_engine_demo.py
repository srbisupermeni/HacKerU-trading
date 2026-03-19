from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine


class FakeRoostooTaker:
    def place_order(self, pair_or_coin, side, quantity, price=None, order_type=None):
        return {
            'Success': True,
            'ErrMsg': '',
            'OrderDetail': {
                'Pair': pair_or_coin if '/' in pair_or_coin else f"{pair_or_coin}/USD",
                'OrderID': 81,
                'Status': 'FILLED',
                'Role': 'TAKER',
                'Side': side.upper(),
                'Type': 'MARKET',
                'Price': price or 0,
                'Quantity': quantity,
                'FilledQuantity': quantity,
            }
        }


class FakeRoostooMaker:
    def place_order(self, pair_or_coin, side, quantity, price=None, order_type=None):
        return {
            'Success': True,
            'ErrMsg': '',
            'OrderDetail': {
                'Pair': pair_or_coin if '/' in pair_or_coin else f"{pair_or_coin}/USD",
                'OrderID': 83,
                'Status': 'PENDING',
                'Role': 'MAKER',
                'Side': side.upper(),
                'Type': 'LIMIT',
                'Price': price or 0,
                'Quantity': quantity,
                'FilledQuantity': 0,
            }
        }


def run_demo():
    print('---- TAKER (FILLED) demo ----')
    p = Portfolio(None)
    engine = ExecutionEngine(p, roostoo_client=FakeRoostooTaker())
    res = engine.execute_order('BTC', 'BUY', 1.0, price=None, order_type=None, strategy_id='s_taker', stop_loss=30000, take_profit=40000)
    print('result:', res)
    print('positions:', p.positions)
    print('strategy_params:', p.strategy_params)
    print('coin_locks keys:', list(p.coin_locks.keys()))

    print('\n---- MAKER (PENDING) demo ----')
    p2 = Portfolio(None)
    engine2 = ExecutionEngine(p2, roostoo_client=FakeRoostooMaker())
    res2 = engine2.execute_order('ETH', 'SELL', 2.0, price=2500, order_type='LIMIT', strategy_id='s_maker', stop_loss=2400, take_profit=2600)
    print('result:', res2)
    print('positions:', p2.positions)
    print('strategy_params:', p2.strategy_params)
    print('coin_locks keys:', list(p2.coin_locks.keys()))


if __name__ == '__main__':
    run_demo()

