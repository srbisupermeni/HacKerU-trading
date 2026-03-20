# 策略层快速开始指南

## 5 分钟快速上手

### 第 1 步：初始化（云端启动时执行一次）

```python
from bot.execution.execution_engine import ExecutionEngine
from bot.portfolio.portfolio import Portfolio
from bot.api.roostoo import Roostoo

portfolio = Portfolio(execution_module=None)
roostoo = Roostoo()
engine = ExecutionEngine(portfolio, roostoo)
```

### 第 2 步：下单（策略的核心）

```python
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=1.0,
    price=50000.0,
    strategy_id='my_strategy'
)

if result['success']:
    print("✓ 下单成功")
else:
    print(f"✗ 下单失败: {result['message']}")
```

### 第 3 步：查看持仓

```python
pos = portfolio.get_position('BTC')
print(f"持仓: {pos['free']} BTC")
```

### 第 4 步：轮询线程（后台运行）

```python
import threading
import time

def poller_loop():
    while True:
        try:
            response = roostoo.query_order()
            engine.process_query_response(response)
            time.sleep(3)
        except Exception as e:
            print(f"轮询错误: {e}")
            time.sleep(3)

poller = threading.Thread(target=poller_loop, daemon=True)
poller.start()
```

---

## 常见场景

### 场景 1：币种被占用

```python
result = engine.execute_order(coin='BTC', ...)
if result['error'] == 'coin_locked':
    # 等待 5 秒后重试
    time.sleep(5)
    result = engine.execute_order(coin='BTC', ...)
```

### 场景 2：订单部分成交

```python
result = engine.execute_order(coin='ETH', quantity=10.0, ...)
if result['remaining_qty'] > 0:
    print(f"已成交 {result['filled_qty']}, 还有 {result['remaining_qty']} 待成交")
    # 无需干预，后台轮询会自动跟踪
```

### 场景 3：查看所有待处理订单

```python
pending = engine.get_pending_orders_snapshot()
for key, orders in pending.items():
    print(f"{key}: {len(orders)} 个订单")
```

---

## 核心规则（必须遵守）

✅ **必须做**：
- 每个策略定义唯一的 `strategy_id`
- 检查 `result['success']` 的返回值
- 处理 `coin_locked` 错误

❌ **禁止做**：
- 不要直接调用 `acquire_coin()` 或 `release_coin()`
- 不要修改 `positions` 数据结构
- 不要忽视错误返回值

---

## 完整例子

```python
import time
from bot.execution.execution_engine import ExecutionEngine
from bot.portfolio.portfolio import Portfolio
from bot.api.roostoo import Roostoo

# 初始化
portfolio = Portfolio(execution_module=None)
roostoo = Roostoo()
engine = ExecutionEngine(portfolio, roostoo)

# 轮询线程
import threading


def poller_loop():
    while True:
        try:
            response = roostoo.query_order()
            engine.process_query_response(response)
            time.sleep(3)
        except:
            time.sleep(3)


poller = threading.Thread(target=poller_loop, daemon=True)
poller.start()

# 策略主循环
while True:
    # 你的策略逻辑...
    signal = analyze_market()

    if signal == 'BUY':
        result = engine.execute_order(
            coin='BTC',
            side='BUY',
            quantity=1.0,
            price=50000.0,
            strategy_id='my_strategy'
        )

        if result['success']:
            print(f"BUY: {result['filled_qty']} BTC")
        else:
            print(f"Failed: {result['message']}")

    elif signal == 'SELL':
        result = engine.execute_order(
            coin='BTC',
            side='SELL',
            quantity=0.5,
            price=55000.0,
            strategy_id='my_strategy'
        )

        if result['success']:
            print(f"SELL: {result['filled_qty']} BTC")

    time.sleep(1)
```

---

## API 参考

### execute_order()

```python
engine.execute_order(
    coin='BTC',                    # 币种
    side='BUY',                    # BUY 或 SELL
    quantity=1.0,                  # 数量
    price=50000.0,                 # 价格（可选，None 时使用 MARKET）
    strategy_id='my_strategy'      # 策略 ID
)
```

返回：`{'success': bool, 'filled_qty': float, 'remaining_qty': float, ...}`

### get_position()

```python
portfolio.get_position('BTC')  # 返回 {'free': float, 'locked': float}
```

### get_pending_orders_snapshot()

```python
engine.get_pending_orders_snapshot()  # 返回待处理订单列表
```

---

## 详细文档

- 完整 API：见 `STRATEGY_LAYER_API.md`
- 规划升级：见 `PORTFOLIO_UPGRADE_PLAN.md`
- 项目状态：见 `PROJECT_STATUS.md`

---

**就这么简单！祝你交易顺利！** 🚀

