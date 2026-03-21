#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整投资组合管理演示

演示新功能：
- 成本基础追踪
- PnL 计算（已实现和未实现）
- 交易历史
- 市场价格更新
"""

import sys
sys.path.insert(0, '/Users/luisng/PycharmProjects/HacKerU-trading')

from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def demo_pnl_and_cost_tracking():
    """演示成本追踪和 PnL 计算"""
    
    print_section("演示 1: 成本基础和 PnL 跟踪")
    
    portfolio = Portfolio(execution_module=None)
    
    # ===== 场景 1: 买入 1.0 BTC @ 50000 =====
    print("📝 步骤 1: 买入 1.0 BTC @ 50000")
    portfolio.acquire_coin('BTC', 'strategy_1')
    portfolio.register_order_execution('BTC', 'strategy_1', 'BUY', 1.0, 50000.0)
    
    pos = portfolio.get_position('BTC')
    print(f"  持仓: {pos['free']} BTC")
    print(f"  总成本: {pos['total_cost']} USDT")
    print(f"  平均成本: {pos['avg_entry_price']} USDT/BTC")
    print(f"  总 PnL: {pos['total_pnl']} USDT")
    
    # ===== 场景 2: 再买入 0.5 BTC @ 51000 =====
    print("\n📝 步骤 2: 再买入 0.5 BTC @ 51000")
    portfolio.acquire_coin('BTC', 'strategy_2')
    portfolio.register_order_execution('BTC', 'strategy_2', 'BUY', 0.5, 51000.0)
    
    pos = portfolio.get_position('BTC')
    print(f"  持仓: {pos['free']} BTC")
    print(f"  总成本: {pos['total_cost']} USDT")
    print(f"  平均成本: {pos['avg_entry_price']:.2f} USDT/BTC")
    print(f"  总 PnL: {pos['total_pnl']} USDT")
    
    # ===== 场景 3: 更新市场价格为 55000 =====
    print("\n📝 步骤 3: 更新市场价格为 55000")
    portfolio.set_market_prices({'BTC': 55000.0})
    
    pos = portfolio.get_position('BTC')
    print(f"  当前价格: {pos['current_price']} USDT")
    print(f"  未实现 PnL: {pos['unrealized_pnl']:.2f} USDT ({pos['unrealized_pnl_pct']:.2f}%)")
    print(f"  总 PnL: {pos['total_pnl']:.2f} USDT")
    
    # ===== 场景 4: 卖出 0.5 BTC @ 55000（实现收益）=====
    print("\n📝 步骤 4: 卖出 0.5 BTC @ 55000（实现盈利）")
    portfolio.acquire_coin('BTC', 'strategy_1')
    portfolio.register_order_execution('BTC', 'strategy_1', 'SELL', 0.5, 55000.0)
    
    pos = portfolio.get_position('BTC')
    print(f"  持仓: {pos['free']} BTC")
    print(f"  已实现 PnL: {pos['realized_pnl']:.2f} USDT")
    print(f"  未实现 PnL: {pos['unrealized_pnl']:.2f} USDT")
    print(f"  总 PnL: {pos['total_pnl']:.2f} USDT")
    
    # ===== 场景 5: 价格涨到 60000 =====
    print("\n📝 步骤 5: 市场价格涨到 60000")
    portfolio.set_market_prices({'BTC': 60000.0})
    
    pos = portfolio.get_position('BTC')
    print(f"  当前价格: {pos['current_price']} USDT")
    print(f"  已实现 PnL: {pos['realized_pnl']:.2f} USDT")
    print(f"  未实现 PnL: {pos['unrealized_pnl']:.2f} USDT")
    print(f"  总 PnL: {pos['total_pnl']:.2f} USDT")
    print(f"  未实现收益率: {pos['unrealized_pnl_pct']:.2f}%")


def demo_transaction_history():
    """演示交易历史"""
    
    print_section("演示 2: 交易历史")
    
    portfolio = Portfolio(execution_module=None)
    
    # 多次交易
    portfolio.acquire_coin('BTC', 's1')
    portfolio.register_order_execution('BTC', 's1', 'BUY', 1.0, 50000.0)
    
    portfolio.acquire_coin('BTC', 's2')
    portfolio.register_order_execution('BTC', 's2', 'BUY', 0.5, 51000.0)
    
    portfolio.acquire_coin('BTC', 's1')
    portfolio.register_order_execution('BTC', 's1', 'SELL', 0.3, 55000.0)
    
    # 获取交易历史
    history = portfolio.get_transaction_history('BTC')
    
    print("📊 BUY 交易:")
    for tx in history['buy_transactions']:
        print(f"  - {tx['quantity']} @ {tx['price']} = {tx['total_cost']} "
              f"(策略: {tx['strategy_id']})")
    
    print("\n📊 SELL 交易:")
    for tx in history['sell_transactions']:
        print(f"  - {tx['quantity']} @ {tx['sell_price']}, "
              f"成本: {tx['avg_cost']}, 利润: {tx['profit']:.2f} ({tx['profit_pct']:.2f}%)")


def demo_pnl_snapshot():
    """演示 PnL 快照"""
    
    print_section("演示 3: PnL 快照（投资组合汇总）")
    
    portfolio = Portfolio(execution_module=None)
    
    # 多币种交易
    print("📝 多币种交易...")
    
    # BTC
    portfolio.acquire_coin('BTC', 's1')
    portfolio.register_order_execution('BTC', 's1', 'BUY', 1.0, 50000.0)
    portfolio.set_market_prices({'BTC': 55000.0})
    
    portfolio.acquire_coin('BTC', 's1')
    portfolio.register_order_execution('BTC', 's1', 'SELL', 0.5, 55000.0)
    
    # ETH
    portfolio.acquire_coin('ETH', 's2')
    portfolio.register_order_execution('ETH', 's2', 'BUY', 10.0, 2000.0)
    portfolio.set_market_prices({'ETH': 2100.0})
    
    # 获取快照
    snapshot = portfolio.get_pnl_snapshot()
    
    print("\n📊 各币种 PnL:")
    for coin, data in snapshot.items():
        if coin == 'portfolio_summary':
            continue
        
        print(f"\n  {coin}:")
        print(f"    持仓量: {data['total_quantity']}")
        print(f"    平均成本: {data['avg_entry_price']:.2f}")
        print(f"    当前价格: {data['current_price']:.2f}")
        print(f"    已实现 PnL: {data['realized_pnl']:.2f}")
        print(f"    未实现 PnL: {data['unrealized_pnl']:.2f}")
        print(f"    总 PnL: {data['total_pnl']:.2f}")
    
    print("\n📈 投资组合汇总:")
    summary = snapshot['portfolio_summary']
    print(f"  已实现 PnL: {summary['total_realized_pnl']:.2f}")
    print(f"  未实现 PnL: {summary['total_unrealized_pnl']:.2f}")
    print(f"  总 PnL: {summary['total_pnl']:.2f}")


def demo_cost_basis():
    """演示成本基础查询"""
    
    print_section("演示 4: 成本基础查询")
    
    portfolio = Portfolio(execution_module=None)
    
    # 买入
    portfolio.acquire_coin('BTC', 's1')
    portfolio.register_order_execution('BTC', 's1', 'BUY', 1.0, 50000.0)
    
    portfolio.acquire_coin('BTC', 's2')
    portfolio.register_order_execution('BTC', 's2', 'BUY', 0.5, 51000.0)
    
    # 查询
    cb = portfolio.get_cost_basis('BTC')
    
    print("📊 BTC 成本基础:")
    print(f"  总持仓: {cb['total_quantity']}")
    print(f"  总成本: {cb['total_cost']}")
    print(f"  平均成本: {cb['avg_entry_price']:.2f}")


def main():
    print("\n" + "="*60)
    print("  完整投资组合管理演示 - Portfolio Phase 2")
    print("="*60)
    
    try:
        demo_pnl_and_cost_tracking()
        demo_transaction_history()
        demo_pnl_snapshot()
        demo_cost_basis()
        
        print_section("✅ 所有演示完成！")
        print("新功能总结:")
        print("  ✓ 成本基础追踪")
        print("  ✓ 平均买入价计算")
        print("  ✓ 已实现 PnL（SELL 时立即计算）")
        print("  ✓ 未实现 PnL（根据市场价格实时计算）")
        print("  ✓ 交易历史（BUY/SELL 完整记录）")
        print("  ✓ PnL 快照（投资组合汇总）")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

