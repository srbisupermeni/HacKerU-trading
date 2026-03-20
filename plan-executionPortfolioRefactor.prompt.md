## Plan: Simple autonomous bot architecture

- [ ] Keep one `Roostoo` client file, split internally into `RoostooGET` and `RoostooPOST`.
- [ ] Make `Portfolio` store only `positions + cost_basis + PnL`, and overwrite it from exchange sync.
- [ ] Refactor `ExecutionEngine` into a thin coordinator for order requests, pending tracking, and fill reconciliation.
- [ ] Add a simple central runtime that pauses everything on sync failure and resumes only after healthy sync.
- [ ] Keep strategy-facing APIs compatible for now, while removing messy coin-lock behavior from the critical path.

## Proposed bot architecture

### 1) Runtime layer
This is the top-level control loop, likely starting in `main.py` for now.

Responsibilities:
- run periodic exchange sync
- run order placement queue
- run order query queue
- pause all trading on sync failure
- resume only after state is healthy again

Priority rules:
1. Exit orders: stop loss / take profit
2. Entry orders: new strategy opens
3. Status queries: LIMIT order polling

Simple rule:
- if one queue has work, use the available request budget there
- if both queues have work, prioritize order placement over queries
- if sync fails, hard stop all activities

### 2) API layer: single client file, split internally
Keep one file, `bot/api/roostoo.py`, but organize it as:

- `RoostooBase`
  - shared config loading
  - timestamp
  - signature generation
  - request helper
  - logging
  - common retry/rate-limit utilities

- `RoostooGET`
  - `serverTime`
  - `exchangeInfo`
  - `ticker`
  - `balance`
  - `pending_count`

- `RoostooPOST`
  - `place_order`
  - `query_order`
  - `cancel_order`

Why this is better:
- GET and POST have different rate limits
- place/query/cancel are all POST, so they need a dedicated rate-limited path
- easier to manage one file than two disconnected clients

### 3) Execution layer
`ExecutionEngine` should remain the strategy-facing entry point, but become simpler.

Current good API to preserve:
- `execute_order(...)`
- `process_query_response(...)`
- `get_pending_orders_snapshot()`

Responsibilities after refactor:
- accept strategy requests
- validate/normalize request
- enqueue order placement
- track pending order metadata
- poll and reconcile query responses
- only update `Portfolio` when an order is truly filled

Important simplification:
- partial fills stay only in pending order metadata
- `Portfolio` is not updated on partial fill
- only true fills update `Portfolio`

That is the simplest working model.

### 4) Portfolio layer
`Portfolio` should become a local account cache and accounting book.

Store only:
- `positions`
- `cost_basis`
- `pnl_tracking`

Derive from these:
- average entry price
- unrealized PnL
- realized PnL
- PnL %

Source of truth:
- exchange account state on sync

So on successful sync:
- overwrite portfolio positions from exchange
- keep your own cost basis / PnL bookkeeping separately
- do not let stale local state drift

### 5) Pending order model
Use one pending-order metadata store inside `ExecutionEngine`.

Track:
- order id
- coin
- side
- original quantity
- filled quantity
- remaining quantity
- price
- timestamps
- strategy id
- raw exchange response

Do not create multiple local order books.

### 6) Locking / strategy ownership
Per latest direction:
- coin-lock logic is on hold/removal path
- strategy team decides ownership
- keep compatibility methods only where needed

### 7) Restart recovery
On boot:
1. sync exchange state
2. query pending orders immediately if exchange is healthy
3. rebuild local pending metadata
4. restore portfolio bookkeeping from persisted `positions + cost_basis + PnL`
5. only then allow trading

For day one persistence, save only:
- `positions`
- `cost_basis`
- `pnl_tracking`

That matches the simple-and-stable goal.

## Minimal compatibility-preserving refactor path
In order:
1. Split `Roostoo` internally into GET/POST child classes.
2. Add a simple central runtime around current `main.py`.
3. Simplify `ExecutionEngine` queueing and full-fill-only bookkeeping.
4. Keep `Portfolio` as exchange-overwritten cache + accounting.
5. Defer SQLite persistence to a later phase.
6. Defer/remove lock ownership in execution/portfolio until strategy ownership is fully finalized.

