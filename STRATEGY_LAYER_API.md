# 策略层 API 文档

策略层仅需关注两个核心接口：**ExecutionEngine** 和 **Portfolio**。

---

## ExecutionEngine - 下单和订单管理

### execute_order() - 下单

```python
result = engine.execute_order(
    coin='BTC',           # 币种
    side='BUY',           # 方向（BUY 或 SELL）
    quantity=1.0,         # 数量
    price=50000.0,        # 价格（可选）
    strategy_id='my_strategy',  # 策略 ID
    stop_loss=45000.0,    # 止损（可选）
    take_profit=60000.0   # 止盈（可选）
)
```

**返回值**：
```python
# 下单成功
{
    'success': True,
    'order_id': '123456',
    'filled_qty': 1.0,        # 已成交
    'remaining_qty': 0.0      # 未成交
}

# 币种被占用
{
    'success': False,
    'error': 'coin_locked',
    'message': 'Coin BTC is locked by another strategy'
}

# 其他错误
{
    'success': False,
    'error': 'order_failed',
    'message': '...'
}
```

**说明**：
- 如果 `remaining_qty > 0`，订单进入待处理队列，后台自动跟踪
- 如果币种被占用（`coin_locked`），请稍后重试或换币种
- 此方法自动获取锁，无需手动处理

---

## Portfolio - 持仓查询

### get_position() - 查看持仓

```python
position = portfolio.get_position('BTC')
```

**返回值**：
```python
{
    'free': 1.0,              # 可用数量
    'locked': 0.0,            # 锁定数量
    'total': 1.0,
    
    'avg_entry_price': 50000.0,  # 平均买入价
    'total_cost': 50000.0,        # 总成本
    
    'unrealized_pnl': 5000.0,     # 未实现收益
    'realized_pnl': 1000.0,       # 已实现收益
    'total_pnl': 6000.0           # 总收益
}
```

---

## 工作流示例

### 场景 1：下单，立即成交

```python
result = engine.execute_order(
    coin='BTC',
    side='BUY',
    quantity=1.0,
    price=50000.0,
    strategy_id='strategy_1'
)

if result['success']:
    # 成功下单
    if result['remaining_qty'] == 0:
        print("订单完全成交")
        pos = portfolio.get_position('BTC')
        print(f"现在持仓: {pos['free']} BTC")
else:
    # 失败处理
    if result['error'] == 'coin_locked':
        # 币种被占用，等待后重试
        time.sleep(5)
    else:
        # 其他错误
        print(f"下单失败: {result['message']}")
```

### 场景 2：下单，部分成交

```python
result = engine.execute_order(
    coin='ETH',
    side='BUY',
    quantity=10.0,
    price=2000.0,
    strategy_id='strategy_1'
)

if result['success'] and result['remaining_qty'] > 0:
    # 订单部分成交，进入待处理队列
    # 后台轮询会继续跟踪，无需策略干预
    print(f"订单已成交 {result['filled_qty']}, 还有 {result['remaining_qty']} 待成交")
```

### 场景 3：查看持仓和收益

```python
pos = portfolio.get_position('BTC')

print(f"持仓: {pos['free']} BTC")
print(f"平均成本: {pos['avg_entry_price']}")
print(f"未实现收益: {pos['unrealized_pnl']} (即 {pos['unrealized_pnl_pct']:.2f}%)")
print(f"已实现收益: {pos['realized_pnl']}")
```

---

## 常见场景处理

### 币种被占用，我应该怎么做？

```python
# 方法 1: 等待重试
for _ in range(10):
    result = engine.execute_order(...)
    if result['success']:
        break
    time.sleep(1)

# 方法 2: 换币种
if not engine.execute_order(coin='BTC', ...):
    engine.execute_order(coin='ETH', ...)

# 方法 3: 查看是谁占用了
pending = engine.get_pending_orders_snapshot()
print(f"待处理订单: {pending}")
```

### 订单部分成交，如何跟踪？

```python
result = engine.execute_order(...)

if result['remaining_qty'] > 0:
    # 订单已进入待处理队列
    # 后台轮询会自动更新成交状态
    # 你只需继续监控持仓
    
    while True:
        pos = portfolio.get_position('BTC')
        if pos['free'] >= expected_amount:  # 当成交完成时
            break
        time.sleep(1)
```

---

## 常见错误和解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `coin_locked` | 币种被其他策略占用 | 等待或换币种 |
| `order_failed` | 交易所拒绝 | 检查价格、数量是否合理 |
| `client_exception` | 网络错误 | 检查网络，重试 |

---

## 核心规则

✅ **必须遵守**：
- 每个策略必须有唯一的 `strategy_id`
- 下单前检查返回的 `success` 字段
- 同一币种同时只有一个策略在交易

❌ **禁止操作**：
- 不要直接调用 `acquire_coin()` 或 `release_coin()`（execute_order 自动处理）
- 不要忽略 `coin_locked` 错误
- 不要手动修改 positions 或 cost_basis

---

**就这么简单！** 只需关注 `execute_order()` 和 `get_position()`，其他都由系统自动处理。


