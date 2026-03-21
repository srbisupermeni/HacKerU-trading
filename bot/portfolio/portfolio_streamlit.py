"""
Streamlit-based Portfolio Monitor

This module provides two things:

- save_portfolio_state(portfolio, file_path): write a JSON snapshot of the passed
  Portfolio instance to disk (atomic write). Intended to be called from user code
  (e.g. from `main`) after portfolio state changes.

- A Streamlit app (when run via `streamlit run bot/portfolio/portfolio_streamlit.py`) that
  reads the JSON snapshot file and displays portfolio tables and PnL summaries.

Notes:
- The module purposefully does not import Portfolio to avoid circular imports. It
  expects an object that implements the public Portfolio API used in the repository
  (attributes/methods read in `save_portfolio_state`).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

try:
    import streamlit as st
    import pandas as pd
except Exception:
    # If streamlit or pandas are not available, the module still exposes save_portfolio_state
    st = None
    pd = None


def _serialize_portfolio(portfolio) -> Dict[str, Any]:
    """Create a JSON-serializable snapshot dict from a Portfolio-like object."""
    snap = {
        'timestamp': time.time(),
        'account_currency': getattr(portfolio, 'account_currency', None),
        'account_balance': getattr(portfolio, 'account_balance', None),
        'positions': getattr(portfolio, 'positions', {}),
        'cost_basis': getattr(portfolio, 'cost_basis', {}),
        'market_prices': getattr(portfolio, 'market_prices', {}),
        'pnl_tracking': getattr(portfolio, 'pnl_tracking', {}),
    }

    # Try to include helper summaries when available
    try:
        snap['pnl_snapshot'] = portfolio.get_pnl_snapshot()
    except Exception:
        snap['pnl_snapshot'] = {}

    try:
        snap['transactions'] = portfolio.get_transaction_history()
    except Exception:
        snap['transactions'] = {}

    return snap


def save_portfolio_state(portfolio, file_path: Optional[str] = None) -> str:
    """Write portfolio snapshot to a JSON file atomically.

    Returns the path written.
    """
    if file_path is None:
        # default to workspace-level temp file
        file_path = os.path.join(os.path.dirname(__file__), 'portfolio_state.json')

    tmp_path = f"{file_path}.tmp"
    snap = _serialize_portfolio(portfolio)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)

    # atomic replace
    os.replace(tmp_path, file_path)
    return file_path


def _read_state(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _render_table(title: str, data: Any):
    if st is None or pd is None:
        return
    st.subheader(title)
    try:
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index')
        else:
            df = pd.DataFrame(data)
        st.dataframe(df)
    except Exception:
        st.write(data)


def main():
    if st is None:
        raise RuntimeError('streamlit is not installed; run `pip install streamlit pandas`')

    st.set_page_config(page_title='Portfolio Monitor', layout='wide')
    st.title('Portfolio Monitor')

    default_path = os.path.join(os.path.dirname(__file__), 'portfolio_state.json')
    file_path = st.sidebar.text_input('State file path', value=default_path)
    refresh = st.sidebar.button('Refresh now')

    # Auto-refresh interval selector (ms)
    interval = st.sidebar.selectbox('Auto-refresh interval', options=[0, 2000, 5000, 10000], index=2)
    if interval and interval > 0:
        rerun = getattr(st, 'experimental_rerun', None)
        if callable(rerun):
            rerun()
        else:
            st.sidebar.info('Auto-refresh requires Streamlit with `experimental_rerun` support.')

    state = _read_state(file_path)
    if not state:
        st.info(f'No state found at {file_path}. Use save_portfolio_state(portfolio, file_path) to publish snapshots.')
        return

    st.sidebar.markdown(f"Last snapshot: {time.ctime(state.get('timestamp', 0))}")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header('Positions')
        _render_table('Positions', state.get('positions', {}))

        st.header('Cost Basis')
        _render_table('Cost Basis', state.get('cost_basis', {}))

    with col2:
        st.header('PnL Snapshot')
        _render_table('PnL Snapshot', state.get('pnl_snapshot', {}))

        st.header('Market Prices')
        _render_table('Market Prices', state.get('market_prices', {}))

    st.header('Recent Transactions')
    tx = state.get('transactions', {})
    if isinstance(tx, dict):
        for coin, rec in tx.items():
            st.subheader(coin)
            _render_table(f'{coin} transactions', rec)
    else:
        st.write(tx)


if __name__ == '__main__':
    # Allows running `python bot/portfolio/portfolio_streamlit.py` for a quick check (without streamlit server)
    try:
        main()
    except RuntimeError as e:
        print(e)

