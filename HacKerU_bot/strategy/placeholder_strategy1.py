"""
PlaceholderStrategy1

简体中文说明：
这是一个占位（示例）策略类，演示如何通过 `ExecutionEngine` 构建本地订单字典并返回。

用法要点：
- 简单方式：直接从 `ExecutionEngine` 导入 `create_order` 并传入 `engine` 实例：
    from bot.execution.execution_engine import ExecutionEngine
    create_order = ExecutionEngine.create_order
    create_order(engine, coin, side, quantity, ...)
- `main()` 应返回由 `create_order(...)` 创建的订单字典（或 None）。
"""


from bot.execution.execution_engine import ExecutionEngine


# 为了简单，直接将类方法作为函数别名导出，调用时需传入 engine 实例作为第一个参数：
# from bot.execution.execution_engine import ExecutionEngine
# create_order = ExecutionEngine.create_order
create_order = ExecutionEngine.create_order


class PlaceholderStrategy1:
    """示例策略类：通过 ExecutionEngine.create_order 构建订单并返回。

    说明（简体中文）：
    - 不再使用外部注入的 `self.create_order` 或 `emit_order` 标志。
    - 在构造时接收一个 `engine` 对象（预期为 `ExecutionEngine` 实例或任何
      提供 `create_order(...)` 方法的对象）。策略在 `main()` 中直接调用
      `self.engine.create_order(...)` 来构建订单字典。

    参数：
    - name: 策略名称，默认 'placeholder1'
    - engine: 必须提供一个拥有 `create_order` 方法的对象（通常为 ExecutionEngine 实例）
    """

    def __init__(self, name: str = 'placeholder1', engine=None):
        # 策略名称
        self.name = name
        # 引擎实例（应提供 create_order 方法）
        self.engine = engine

    def main(self):
        """策略主入口。

        直接使用 `self.engine.create_order(...)` 构建订单并返回。若 `engine` 未设置
        或不包含 `create_order`，则抛出 RuntimeError。
        返回值：create_order 返回的订单字典（通常传入 execute_order 以执行）。
        """

        # 示例下单参数；实际策略可依据市场数据计算这些值
        coin = 'BTC'
        side = 'BUY'
        quantity = 0.01
        order_type = 'MARKET'

        # 通过包装的 create_order 函数创建标准化订单字典（传入 engine 实例）
        order = create_order(self.engine, coin=coin, side=side, quantity=quantity,
                             price=None, order_type=order_type, strategy_id=self.name)

        return order
