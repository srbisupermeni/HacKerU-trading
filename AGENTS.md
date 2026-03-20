# AGENTS.md — quick guide for AI coding agents

Purpose: give an AI coder (or CI bot) the minimal, actionable knowledge to be immediately productive in this repository.

1) Big-picture architecture (one-paragraph)
- Layers: data fetchers (database/), feature engineering (bot/data/), strategy & utilities (top-level files like `Stop_Loss.py`, `Trading_Volume.py`, `Get_Price.py`), and execution (bot/execution/roostoo.py). Tests and demos live in `test/`.
- Data flows: fetch historical or hot market data via `database/Binance_Vision_fetcher.py` or `database/Binance_fetcher.py` → optional feature engineering with `_add_technical_indicators` or `bot/data/feature_engineering.py` → strategy/decision code (top-level scripts) → execution through `Roostoo` client (`bot/execution/roostoo.py`).
  - Execution layer expanded: there is now a separate execution/portfolio implementation that manages order lifecycle, locks and PnL tracking: `bot/execution/execution_engine.py` (class `ExecutionEngine`) and `bot/portfolio/portfolio.py` (class `Portfolio`). Strategies should call `ExecutionEngine.execute_order(...)` with a `strategy_id` to acquire coin locks rather than directly calling `Roostoo.place_order` in most cases.

2) Key files & entry points (exact symbols to call)
- Execution / exchange client: `bot/execution/roostoo.py` — class `Roostoo`. Demo-run via `python bot/execution/roostoo.py` (the file includes a `if __name__ == "__main__"` demo). Config YAML expected at `bot/config/roostoo.yaml` (keys: `roostoo.base_url`, `roostoo.api_key`, `roostoo.secret_key`). Logs written to `bot/logs/roostoo.log`.
 - Execution engine: `bot/execution/execution_engine.py` — class `ExecutionEngine`. Primary strategy entrypoint for placing orders in the application stack; handles coin-level locking, partial-fill handling, and pending-order queueing. Call `ExecutionEngine.execute_order(coin, side, quantity, price=None, order_type=None, strategy_id=...)`.
 - Portfolio manager: `bot/portfolio/portfolio.py` — class `Portfolio`. Manages coin locks (`acquire_coin`, `release_coin`), positions, cost-basis and PnL tracking (`register_order_execution`, `update_market_prices`), and a background monitoring thread (`start_monitoring`). Prefer using `Portfolio` APIs for bookkeeping rather than mutating state directly.
 - Local mock server for safe testing: `mock/mock_roostoo_server.py` — lightweight HTTP mock of the Roostoo API useful for offline/safe integration tests. Run with `python mock/mock_roostoo_server.py --port 8000`.
- Roostoo procedural demo/test: `test/test roostoo API.py` (contains standalone functions like `check_server_time()`, `place_order(...)`) — runnable with `python "test/test roostoo API.py"`.
- Historical / batch fetch: `database/Binance_Vision_fetcher.py` — class `VisionFetcher` with `get_and_save_range(...)`, `fetch_klines_range(...)`, and helper `_add_technical_indicators(df)`.
- Live/SDK fetch: `database/Binance_fetcher.py` — class `BinanceDataFetcher` with `fetch_recent_klines(...)` and `_add_technical_indicators(...)`.
- CoinGecko: `database/coingecko_api.py` — class `CoinGeckoClient` with `get_today_overview(...)`, `get_today_intraday_data(...)`, and `get_market_depth(...)`.

3) Developer workflows & exact commands
- Install dependencies: `pip install -r requirements.txt` (project root). Requirements include pandas, numpy, python-binance, PyYAML, scikit-learn, xgboost, matplotlib.
- Run Roostoo demo client: `python bot/execution/roostoo.py`.
 - Run the mock Roostoo API server (safe local testing): `python mock/mock_roostoo_server.py --port 8000`.
- Run VisionFetcher demo (download historical data): `python database/Binance_Vision_fetcher.py`.
- Run Binance SDK fetcher demo: `python database/Binance_fetcher.py`.
- Run procedural Roostoo tests/demos: `python "test/test roostoo API.py"`.
 - Run a full execution + portfolio demo (example pattern): import `ExecutionEngine` and `Portfolio` and use `ExecutionEngine.execute_order(...)` (see `bot/execution/execution_engine.py` docstring for usage examples).
- Notes: there is no pytest-based test suite; the `test/` directory contains a runnable demo script (filename includes a space). Use the `python` command to run these scripts directly.

4) Project-specific conventions & gotchas
- Config vs env: code expects YAML config at `bot/config/roostoo.yaml` (see Roostoo docstring). The repository's guidebook recommends `.env`, but the code uses YAML — prefer creating the YAML file for immediate runs.
- Rate limits enforced in clients: Roostoo client enforces 30 requests/min (deque of 30) and CoinGecko client enforces 15 requests/min. Any agent must respect these limits or reuse the client methods (they sleep internally).
- Root detection pattern: both fetcher classes attempt to find the project root by walking up until they find `requirements.txt`. This matters for where data is saved (`database/raw_data`, `database/processed_data`) and for resolving relative paths.
- Timezones & timestamps: Binance fetchers convert timestamps to UTC+8 (Beijing time) in-place — be mindful when comparing timestamps from other sources.
 - Order lifecycle & locking conventions (important): `ExecutionEngine` and `Portfolio` implement an application-level coin lock. Callers must pass a `strategy_id` when placing orders via `ExecutionEngine.execute_order(...)`. `Portfolio.acquire_coin` is non-blocking and returns False if the coin is already locked; `release_coin` validates ownership unless `force=True`. Partial fills are re-acquired internally by the engine; do NOT try to take ownership of a coin that you did not acquire.
- Mixed styles: the repo mixes class-based clients (VisionFetcher, BinanceDataFetcher, CoinGeckoClient, Roostoo) with procedural scripts (test demo). Expect inconsistent naming and some fragile modules (see `Get_Price.py`).
- Buggy/incomplete files: `Get_Price.py` contains undefined variables (`params` and erroneous request usage). Treat it as needing fixes before use.

5) Integration points & external dependencies
- Roostoo API (mock URL defaulted in code): used by `bot/execution/roostoo.py` and `test/test roostoo API.py` (see `BASE_URL` or YAML `base_url`).
 - Local mock Roostoo server included: `mock/mock_roostoo_server.py` provides a simple HTTP server that emulates `/v3/place_order`, `/v3/query_order`, `/v3/cancel_order`, `/v3/exchangeInfo`, `/v3/ticker`, `/v3/balance`. Use it for safe end-to-end testing without hitting real endpoints.
- Binance Vision (data.binance.vision) scraped by `database/Binance_Vision_fetcher.py` (ZIP -> CSV processing).
- Binance REST/SDK: `python-binance` Client used in `database/Binance_fetcher.py` for recent klines.
- CoinGecko public API: `database/coingecko_api.py` for intraday/overview/depth. Remember CoinGecko requires full coin IDs (e.g., "bitcoin" not "BTC").
 - Config file note: this repository already contains `bot/config/roostoo.yaml` with `roostoo.api_key` and `roostoo.secret_key` set. Treat any credentials in the repo as sensitive: do NOT commit new secrets, and rotate/remove the keys before publishing. Agents should prefer running the included mock server or using a `.yaml.template` rather than editing the existing YAML in-place.

6) Quick examples (copyable)
- Instantiate Roostoo and check ticker:
  - python snippet: `from bot.execution.roostoo import Roostoo; c=Roostoo(); print(c.get_ticker('BTC/USD'))`
- Place a market order (demo):
  - python snippet: `from bot.execution.roostoo import Roostoo; c=Roostoo(); c.place_order('BNB/USD','BUY',1)`
 - Place orders via the ExecutionEngine (preferred):
   - python snippet: `from bot.portfolio.portfolio import Portfolio; from bot.execution.execution_engine import ExecutionEngine; p=Portfolio(None); engine=ExecutionEngine(p); engine.execute_order('BTC','BUY',0.01, price=None, strategy_id='strategy_1')`
 - Run the mock server locally and point Roostoo to it (safe test):
   - shell: `python mock/mock_roostoo_server.py --port 8000` and then set `bot/config/roostoo.yaml` base_url to `http://127.0.0.1:8000` or run `Roostoo(config_path=...)` overriding base_url.
- Download Binance Vision range and save processed CSV:
  - python snippet: `from database.Binance_Vision_fetcher import VisionFetcher; v=VisionFetcher(); v.get_and_save_range(symbol='BTCUSDT', interval='1h', start_year=2025, start_month=1, end_year=2025, end_month=3, data_type='monthly')`
- Get CoinGecko intraday prices (note coin id):
  - python snippet: `from database.coingecko_api import CoinGeckoClient; cg=CoinGeckoClient(); d=cg.get_today_intraday_data('bitcoin'); print(d['prices'][-1])`

7) Where to look first when debugging or extending
- Execution & API wiring: `bot/execution/roostoo.py` (logging, config, signed headers, rate limits).
- Fetch pipelines & saved data: `database/Binance_Vision_fetcher.py` and output folders `database/raw_data` and `database/processed_data`.
 - Execution engine and order lifecycle: `bot/execution/execution_engine.py` (pending-order queueing, `process_query_response`) and bookkeeping: `bot/portfolio/portfolio.py` (coin locks, cost basis, realized/unrealized PnL). These are the best places to inspect how orders are tracked and how filling/cancelling is handled.
- Quick tests and examples: `test/test roostoo API.py` (procedural examples you can run to reproduce API calls).

8) What an AI agent should NOT change without human review
- Any code that sends real orders (functions `place_order`, `cancel_order`, `place_order` demos). Treat order placement code as sensitive. Use dry-run or mock servers.
- YAML credentials or any code that writes secrets into repo.
 - The repository currently contains `bot/config/roostoo.yaml` with example/real-looking API keys and `test/test roostoo API.py` also contains hard-coded `API_KEY`/`SECRET_KEY` constants. Do NOT alter these files to insert new secrets; prefer using environment variables, a `bot/config/roostoo.yaml.template`, or the included mock server for testing.

9) Missing/absent conventions (so agent can propose fixes)
- No CI/test harness (no pytest tests). The agent can add unit tests but must follow existing procedural demo patterns.
- No linter/formatter specified in `requirements.txt` — adopt project style (simple, small diffs) and prefer not to reformat unrelated files.

If you want, I can now:
- create a `bot/config/roostoo.yaml.template` with the expected YAML keys, or
- fix `Get_Price.py` to be usable, or
- add a small safe-mode wrapper that mocks `Roostoo.place_order` for offline testing.

---
Discovered: no existing AGENT/AGENTS/Copilot instructions files were found in the repo. This AGENTS.md is intentionally compact and code-reference rich so automated agents can act safely and quickly.

