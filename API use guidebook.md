# roostoo use guidebook from andy (API，GET，POST)

ipynb link to colab\
https://colab.research.google.com/drive/1NvtY6uWPYINias8CfPuTgwxg7V935It6?usp=sharing

### Q14. Can we use other APIs for market data?
- Ans: Yes.

### Q15. Is Roostoo real-time data different from Binance?

**Ans:** No. Roostoo real-time pricing is streamed from Binance.

### Q16. Does Roostoo API provide OHLCV?
**Ans**: No. It provides ticker snapshot.
For OHLCV, use external data providers in our data source pack resources. 
OHLCV = 开盘价 / 最高价 / 最低价 / 收盘价 / 成交量（K 线数据）
ticker snapshot = 只给你当前最新价格、盘口快照，不是历史 K 线
结论：
不能用 Roostoo 做回测，因为没有历史 K 线。
回测要用官方给的 Data Resource 里的外部数据。

### Q38. Should teams log their trades?

Yes — strongly recommended.

Minimum logging:

- Timestamp
- Symbol
- Side
- Price
- Quantity
- Order ID
- API response

Optional:

- PnL
- Signal reason
- Strategy state

- ### Q37. What’s best practice for repository management?

Recommended structure:

```
bot/
  strategy/
  execution/
  data/
  config/
  logs/
tests/
requirements.txt
Dockerfile
README.md
```

Best practices:

- Use `.env` for API keys
- Add clear README
- Use Git branches (main/dev)
- Tag final submission version
- Keep it reproducible

### Q41. What should be included in the README file of repository?

Ans: Your README should clearly explain how your bot works and how to run it. Judges should be able to understand and reproduce your project easily.

Recommended README Structure (feel free to create your own structure too):

1. **Project** **Overview**
    - Short description of your strategy
    - High-level idea (e.g., momentum, mean reversion, ML-based, arbitrage, etc.)
    - Key features
2. **Architecture**
    - System design diagram (optional but recommended)
    - Components (data module, strategy module, execution module, logging module)
    - Tech stack used
3. **Strategy Explanation**
    - Entry conditions
    - Exit conditions
    - Risk management rules
    - Position sizing logic
    - Any assumptions made
4. **Setup instructions & How to run bot**
