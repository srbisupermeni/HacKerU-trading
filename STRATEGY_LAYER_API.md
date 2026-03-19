# Portfolio 和 ExecutionEngine API 文档

## 概述

这是为策略层提供的 API 文档。策略层通过这两个主要接口与执行和持仓管理系统交互。

- **Portfolio**: 管理持仓、币种锁、策略参数
- **ExecutionEngine**: 下单、订单管理、成交反馈处理

## 快速开始

```python
from bot.execution.execution_engine import ExecutionEngine
from bot.portfolio.portfolio import Portfolio
from bot.execution.roostoo import Roostoo

# 1. 初始化（云端启动时执行一次）
portfolio = Portfolio(execution_module=None)
roostoo_client = Roostoo()  # 连接到交易所
engine = ExecutionEngine(portfolio, roostoo_client)

# 2. 下单（策略主要调用此方法）
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=1.0,
    price=50000.0,
    strategy_id='my_strategy',
    stop_loss=45000.0,
    take_profit=60000.0
)

# 3. 检查结果
if result['success']:
    print(f"下单成功，订单 ID: {result['order_id']}")
    if result.get('remaining_qty', 0) > 0:
        print(f"部分成交，剩余 {result['remaining_qty']}")
else:
    print(f"下单失败: {result.get('message')}")

# 4. 后台轮询（云端长期运行）
# 参考下面的"轮询线程"部分
```

---

## ExecutionEngine API

### 初始化

```python
engine = ExecutionEngine(portfolio, roostoo_client=None)
```

**参数**:
- `portfolio` (Portfolio): 持仓管理对象
- `roostoo_client` (Roostoo, 可选): 交易所客户端，默认自动创建

---

### execute_order - 下单（主要接口）

```python
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=1.0,
    price=50000.0,
    order_type='LIMIT',
    strategy_id='strategy_1',
    stop_loss=45000.0,
    take_profit=60000.0
)
```

**参数**:
- `coin` (str): 币种代码，例如 'BTC', 'ETH', 'DOGE'
- `side` (str): 方向，'BUY' 或 'SELL'
- `quantity` (float): 数量
- `price` (float, 可选): 价格
  - 如果不提供，自动使用 MARKET 订单
  - 如果提供，自动使用 LIMIT 订单
- `order_type` (str, 可选): 订单类型，'MARKET' 或 'LIMIT'
  - 通常不需要指定，系统根据 price 自动判断
- `strategy_id` (str): 策略 ID（必须），例如 'ma_crossover', 'rsi_bot'
- `stop_loss` (float, 可选): 止损价格
- `take_profit` (float, 可选): 止盈价格

**返回值** (dict):

成功时：
```python
{
    'success': True,
    'order_id': '123456789',          # 交易所订单 ID
    'status': 'FILLED',               # 订单状态
    'filled_qty': 1.0,                # 已成交数量
    'remaining_qty': 0.0,             # 未成交数量
    'queued': False,                  # 是否在待处理队列
    'raw': {...}                      # 交易所原始响应
}
```

部分成交时：
```python
{
    'success': True,
    'order_id': '123456789',
    'status': 'PENDING',
    'filled_qty': 0.5,                # 已成交
    'remaining_qty': 0.5,             # 还未成交
    'queued': True,                   # 订单在待处理队列
    'raw': {...}
}
```

失败时：
```python
{
    'success': False,
    'error': 'coin_locked',           # 错误代码
    'message': 'Coin BTC is locked by another strategy',
    'raw': {...}
}
```

**常见错误**:
- `coin_locked`: 币种被其他策略占用，请等待或换币种
- `client_exception`: 与交易所通信出错
- `order_failed`: 交易所拒绝下单
- `post_registration_failed`: 成交登记失败

**使用示例**:

```python
# 下单 BTC
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=0.5,
    price=50000.0,
    strategy_id='my_strategy'
)

# 检查是否成功
if not result['success']:
    print(f"下单失败: {result['message']}")
    return

# 检查是否完全成交
if result['remaining_qty'] <= 0:
    print(f"订单完全成交: {result['filled_qty']} BTC")
else:
    print(f"订单部分成交: {result['filled_qty']} BTC")
    print(f"还有 {result['remaining_qty']} BTC 待成交")
    print("订单已进入待处理队列，后台会继续跟踪")
```

---

### get_pending_orders_snapshot - 查看待处理订单

```python
pending = engine.get_pending_orders_snapshot()
```

**返回值** (dict):
```python
{
    'BTC:strategy_1': [
        {
            'order_id': '123',
            'side': 'BUY',
            'original_quantity': 1.0,
            'filled_quantity': 0.5,
            'remaining_quantity': 0.5,
            'price': 50000.0,
            'created_at': 1647864000000,
            'last_update': 1647864030000
        }
    ],
    'ETH:strategy_2': [
        {...}
    ]
}
```

**使用示例**:

```python
# 查看所有待处理订单
pending = engine.get_pending_orders_snapshot()

for queue_key, orders in pending.items():
    coin, strategy = queue_key.split(':')
    print(f"{coin} ({strategy}):")
    for order in orders:
        progress = (order['filled_quantity'] / order['original_quantity']) * 100
        print(f"  订单 {order['order_id']}: "
              f"{progress:.1f}% 成交 "
              f"({order['filled_quantity']}/{order['original_quantity']})")
```

---

### process_query_response - 处理订单查询结果

```python
results = engine.process_query_response(response)
```

**参数**:
- `response` (dict): 交易所返回的查询响应

**返回值** (list):
```python
[
    {
        'order_id': '123456',
        'status': 'FILLED',
        'processed': True,
        'type': 'full_fill',              # 完全成交
        'filled_qty': 1.0,
        'remaining_qty': 0.0
    },
    {
        'order_id': '123457',
        'status': 'PENDING',
        'processed': True,
        'type': 'partial_fill',           # 部分成交
        'filled_qty': 0.5,
        'remaining_qty': 0.5
    }
]
```

**使用示例** (后台轮询线程):

```python
import threading
import time

def poller_thread():
    """后台轮询线程，定期查询订单状态"""
    while True:
        try:
            # 查询所有订单
            response = roostoo_client.query_order()
            
            # 处理查询结果
            results = engine.process_query_response(response)
            
            # 记录处理结果
            for result in results:
                if result['type'] == 'full_fill':
                    print(f"✓ 订单 {result['order_id']} 完全成交")
                elif result['type'] == 'partial_fill':
                    print(f"⚡ 订单 {result['order_id']} 进展: "
                          f"已成交 {result['filled_qty']}, "
                          f"剩余 {result['remaining_qty']}")
                elif result['type'] == 'canceled':
                    print(f"✗ 订单 {result['order_id']} 被取消")
            
            time.sleep(3)  # 每 3 秒查询一次
            
        except Exception as e:
            print(f"轮询错误: {e}")
            time.sleep(3)

# 启动轮询线程
poller = threading.Thread(target=poller_thread, daemon=True)
poller.start()
```

---

## Portfolio API

### 初始化

```python
portfolio = Portfolio(execution_module=None)
```

**参数**:
- `execution_module` (可选): ExecutionEngine 引用，通常传 None

---

### get_position - 查看持仓

```python
position = portfolio.get_position('BTC')
```

**参数**:
- `coin` (str): 币种代码

**返回值** (dict):
```python
{
    'free': 1.5,      # 可用数量
    'locked': 0.5     # 锁定数量（待成交订单）
}
```

**使用示例**:

```python
pos = portfolio.get_position('BTC')
print(f"可用 BTC: {pos['free']}")
print(f"锁定 BTC: {pos['locked']}")
print(f"总额: {pos['free'] + pos['locked']}")
```

---

### acquire_coin - 获取币种锁

```python
acquired = portfolio.acquire_coin('BTC', 'my_strategy')
```

**参数**:
- `coin` (str): 币种代码
- `strategy_id` (str): 策略 ID

**返回值** (bool):
- `True`: 成功获取锁，可以下单
- `False`: 币种被其他策略占用

**说明**:
- 非阻塞式调用，立即返回
- 通常 **不需要** 直接调用，execute_order 会自动处理

**使用示例** (仅供参考):

```python
if portfolio.acquire_coin('BTC', 'strategy_1'):
    print("成功获取 BTC 的锁")
    # 继续下单...
else:
    print("BTC 被其他策略占用，请稍后重试")
```

---

### release_coin - 释放币种锁

```python
released = portfolio.release_coin('BTC', strategy_id='my_strategy')
```

**参数**:
- `coin` (str): 币种代码
- `strategy_id` (str, 可选): 策略 ID，建议提供
- `force` (bool, 可选): 强制释放，仅供系统使用

**返回值** (bool):
- `True`: 成功释放
- `False`: 释放失败

**说明**:
- 通常 **不需要** 直接调用，execute_order 会自动处理
- 仅在特殊情况下需要手动调用

---

### set_strategy_params - 设置止损/止盈

```python
portfolio.set_strategy_params(
    coin='BTC',
    strategy_id='my_strategy',
    stop_loss=45000.0,
    take_profit=60000.0
)
```

**参数**:
- `coin` (str): 币种代码
- `strategy_id` (str): 策略 ID
- `stop_loss` (float): 止损价格
- `take_profit` (float): 止盈价格

**说明**:
- 通常在 execute_order 中自动设置
- 无需单独调用

---

### get_strategy_params - 查看止损/止盈

```python
params = portfolio.get_strategy_params('BTC')
```

**参数**:
- `coin` (str): 币种代码

**返回值** (dict):
```python
{
    'strategy_id': 'my_strategy',
    'stop_loss': 45000.0,
    'take_profit': 60000.0
}
```

---

## 完整工作流示例

### 场景 1: 下单 BTC，立即成交

```python
# 下单
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=1.0,
    price=50000.0,
    strategy_id='ma_crossover',
    stop_loss=45000.0,
    take_profit=60000.0
)

# 检查结果
if result['success']:
    if result['remaining_qty'] <= 0:
        print("✓ 订单完全成交")
        pos = portfolio.get_position('BTC')
        print(f"现在拥有 {pos['free']} BTC")
    else:
        print("⚡ 订单部分成交，进入待处理队列")
else:
    print(f"✗ 下单失败: {result['message']}")
```

### 场景 2: 下单后部分成交，等待后续成交

```python
# 下单（部分成交）
result = engine.execute_order(
    coin='ETH',
    side='BUY',
    quantity=10.0,
    price=2000.0,
    strategy_id='rsi_bot'
)

# 订单进入待处理队列
pending = engine.get_pending_orders_snapshot()
print(f"待处理订单数: {len(pending)}")

# 后台轮询会继续跟踪，无需策略层干预
```

### 场景 3: 检查持仓

```python
# 查看所有币种持仓
coins = ['BTC', 'ETH', 'DOGE']
for coin in coins:
    pos = portfolio.get_position(coin)
    total = pos['free'] + pos['locked']
    if total > 0:
        print(f"{coin}: 可用 {pos['free']}, 锁定 {pos['locked']}")
```

---

## 常见问题

### Q: 下单时币种被锁定怎么办？
**A**: 返回 `coin_locked` 错误。这表示其他策略正在交易该币种。请：
1. 等待几秒后重试
2. 或换其他币种

### Q: 订单部分成交后，剩余部分何时成交？
**A**: 后台轮询线程会定期查询订单状态。当剩余部分成交时，系统会自动更新持仓和待处理队列，无需策略层干预。

### Q: 我需要自己实现轮询线程吗？
**A**: 是的，需要在云端启动一个后台线程定期调用 `process_query_response()`，参考上面的"轮询线程"示例。

### Q: 止损/止盈如何触发？
**A**: 目前止损/止盈参数已记录，但触发机制需在策略层或监控层实现（参考待实现的 `_monitor_loop` 方法）。

### Q: 可以同时对不同币种下单吗？
**A**: 可以。锁定机制是按币种的，不同币种的订单不会相互影响。

### Q: 可以同时对同一币种下单吗？
**A**: 不可以。同一币种同一时间只能有一个策略的订单，避免持仓混乱。

---

## 线程安全性

✅ **完全线程安全**

所有接口都是线程安全的，可以：
- 在策略线程中调用 `execute_order()`
- 在轮询线程中调用 `process_query_response()`
- 同时在多个线程中查询 `get_position()` 等

内部使用锁保护共享状态，无需策略层担心并发问题。

---

## 错误处理

所有接口都返回结果（不抛异常），策略层应检查返回值：

```python
result = engine.execute_order(...)

if not result['success']:
    error_code = result.get('error')
    error_msg = result.get('message')
    print(f"错误 [{error_code}]: {error_msg}")
    
    # 根据错误类型处理
    if error_code == 'coin_locked':
        # 等待重试
        time.sleep(5)
    elif error_code == 'client_exception':
        # 网络错误，等待网络恢复
        time.sleep(10)
```

---

## 性能建议

1. **下单频率**: 正常情况下无限制，但建议避免高频重复下单
2. **轮询频率**: 建议每 2-5 秒查询一次（根据交易所 API 限制）
3. **查询持仓**: 可频繁调用，成本很低

---

## 总结

策略层需要实现的基本流程：

```python
# 1. 初始化（启动时）
engine = ExecutionEngine(portfolio, roostoo_client)

# 2. 下单（策略信号出现时）
result = engine.execute_order(...)

# 3. 轮询（后台长期运行）
while True:
    response = roostoo_client.query_order()
    engine.process_query_response(response)
    time.sleep(3)

# 4. 查询持仓（需要时）
position = portfolio.get_position('BTC')
```

其他细节都由 Portfolio 和 ExecutionEngine 自动处理。

---

**文档版本**: 1.0
**最后更新**: 2026-03-19
**状态**: ✓ 生产就绪

