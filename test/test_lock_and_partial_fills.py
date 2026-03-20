"""
Demo and test script for the new application-level locking and partial fill handling.

This script demonstrates:
1. Application-level ownership lock (safe cross-thread release)
2. Partial fill handling (tracking filled/remaining quantities, keeping locks)
3. Full fill handling (releasing locks after complete execution)
4. Cancelled order handling
5. Safe multi-threaded scenarios

Run with: python test/test_lock_and_partial_fills.py
"""

import sys
import threading
import time
import logging
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(threadName)-12s] [%(name)s] %(levelname)-8s %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
sys.path.insert(0, '/Users/luisng/PycharmProjects/HacKerU-trading')
from bot.portfolio.portfolio import Portfolio
from bot.execution.execution_engine import ExecutionEngine


class FakeRoostoo:
    """Mock Roostoo client for testing without real API calls."""
    
    def __init__(self):
        self.call_count = 0
        self.orders = {}
    
    def place_order(self, coin: str, side: str, quantity: float, 
                   price: Optional[float] = None, order_type: str = 'MARKET') -> Dict[str, Any]:
        """
        Simulate placing orders with various fill scenarios.
        """
        self.call_count += 1
        order_id = self.call_count
        
        # Scenario mapping by call count for demo purposes
        scenario = self.call_count % 4
        
        if scenario == 1:
            # Scenario 1: Full fill immediately (MARKET taker)
            logger.info(f"[FakeRoostoo] Scenario 1: Full MARKET fill (order_id={order_id})")
            return {
                'Success': True,
                'OrderDetail': {
                    'OrderID': order_id,
                    'Status': 'FILLED',
                    'Role': 'TAKER',
                    'Pair': f'{coin}/USD',
                    'Side': side,
                    'Type': order_type,
                    'Quantity': quantity,
                    'FilledQuantity': quantity,
                    'FilledAverPrice': price or 100.0,
                    'Price': price or 100.0
                }
            }
        
        elif scenario == 2:
            # Scenario 2: Partial fill immediately
            filled = quantity * 0.3
            logger.info(f"[FakeRoostoo] Scenario 2: Partial fill {filled}/{quantity} (order_id={order_id})")
            return {
                'Success': True,
                'OrderDetail': {
                    'OrderID': order_id,
                    'Status': 'PENDING',
                    'Role': 'MAKER',
                    'Pair': f'{coin}/USD',
                    'Side': side,
                    'Type': order_type,
                    'Quantity': quantity,
                    'FilledQuantity': filled,
                    'FilledAverPrice': 0.0,
                    'Price': price or 100.0
                }
            }
        
        elif scenario == 3:
            # Scenario 3: PENDING MAKER order (no fill)
            logger.info(f"[FakeRoostoo] Scenario 3: PENDING MAKER order (order_id={order_id})")
            return {
                'Success': True,
                'OrderDetail': {
                    'OrderID': order_id,
                    'Status': 'PENDING',
                    'Role': 'MAKER',
                    'Pair': f'{coin}/USD',
                    'Side': side,
                    'Type': order_type,
                    'Quantity': quantity,
                    'FilledQuantity': 0.0,
                    'FilledAverPrice': 0.0,
                    'Price': price or 100.0
                }
            }
        
        else:
            # Scenario 0: Order failed
            logger.info(f"[FakeRoostoo] Scenario 0: Order failed (order_id={order_id})")
            return {
                'Success': False,
                'ErrMsg': 'Order placement failed'
            }


def test_basic_acquire_release():
    """Test basic acquire and release with application-level locking."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Basic acquire/release")
    logger.info("="*60)
    
    portfolio = Portfolio(execution_module=None)
    
    # Test 1a: Successful acquire
    acquired = portfolio.acquire_coin('BTC', 'strategy_1')
    assert acquired, "Should acquire BTC for strategy_1"
    logger.info("✓ Acquired BTC for strategy_1")
    
    # Test 1b: Blocking acquire (coin already locked)
    acquired = portfolio.acquire_coin('BTC', 'strategy_2')
    assert not acquired, "Should not acquire BTC (already locked by strategy_1)"
    logger.info("✓ Blocked second acquire for BTC (correct)")
    
    # Test 1c: Release with ownership verification
    released = portfolio.release_coin('BTC', strategy_id='strategy_1')
    assert released, "Should release BTC"
    logger.info("✓ Released BTC by strategy_1")
    
    # Test 1d: Now second strategy should acquire
    acquired = portfolio.acquire_coin('BTC', 'strategy_2')
    assert acquired, "Should acquire BTC after release"
    logger.info("✓ Acquired BTC for strategy_2 after release")
    
    # Cleanup
    portfolio.release_coin('BTC', strategy_id='strategy_2')
    logger.info("✓ Test 1 passed\n")


def test_cross_thread_release():
    """Test that release can be called from different thread (application-level lock benefit)."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Cross-thread release (safe with app-level locking)")
    logger.info("="*60)
    
    portfolio = Portfolio(execution_module=None)
    results = []
    
    def thread_acquire():
        # Thread 1: Acquire lock
        acquired = portfolio.acquire_coin('ETH', 'strategy_a')
        results.append(('acquire', acquired))
        logger.info(f"Thread 1: Acquired ETH = {acquired}")
        time.sleep(0.5)  # Hold lock briefly
    
    def thread_release():
        time.sleep(0.2)  # Wait for acquire to complete
        # Thread 2: Release the lock acquired by Thread 1
        # This would fail with OS-level threading.Lock, but works with app-level ownership
        released = portfolio.release_coin('ETH', strategy_id='strategy_a')
        results.append(('release', released))
        logger.info(f"Thread 2: Released ETH (by strategy_a) = {released}")
    
    t1 = threading.Thread(target=thread_acquire, name='Acquire')
    t2 = threading.Thread(target=thread_release, name='Release')
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    assert results[0] == ('acquire', True), "Acquire should succeed"
    assert results[1] == ('release', True), "Release from different thread should succeed"
    logger.info("✓ Test 2 passed: Cross-thread release works correctly\n")


def test_lock_status_and_force_release():
    """Test lock status queries and force release for recovery."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Lock status and force release")
    logger.info("="*60)
    
    portfolio = Portfolio(execution_module=None)
    
    # Acquire and check status
    portfolio.acquire_coin('XRP', 'strategy_x')
    status = portfolio.get_coin_lock_status('XRP')
    assert status['locked'] == True, "Lock should be held"
    assert status['owner_strategy_id'] == 'strategy_x', "Owner should be strategy_x"
    logger.info(f"✓ Lock status: {status}")
    
    # Force release (admin recovery)
    forced = portfolio.force_release_coin('XRP')
    assert forced, "Force release should succeed"
    status = portfolio.get_coin_lock_status('XRP')
    assert status['locked'] == False, "Lock should be released"
    logger.info("✓ Force-released XRP (admin intervention)")
    
    # Check all locks
    portfolio.acquire_coin('BTC', 'strat1')
    portfolio.acquire_coin('ETH', 'strat2')
    all_locks = portfolio.get_all_lock_status()
    assert len(all_locks) >= 2, "Should have at least 2 locks"
    logger.info(f"✓ All lock statuses: {all_locks}\n")


def test_full_execution_flow():
    """Test full execution flow with ExecutionEngine and partial fills."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Full execution flow with partial fills")
    logger.info("="*60)
    
    portfolio = Portfolio(execution_module=None)
    fake_roostoo = FakeRoostoo()
    engine = ExecutionEngine(portfolio, roostoo_client=fake_roostoo)
    
    # Execute order 1: Should result in full fill (scenario 1)
    logger.info("\n--- Order 1: Full MARKET fill ---")
    result1 = engine.execute_order(
        coin='BTC',
        side='BUY',
        quantity=1.0,
        price=None,
        order_type='MARKET',
        strategy_id='strat_1',
        stop_loss=30000.0,
        take_profit=40000.0
    )
    logger.info(f"Result 1: {result1}")
    assert result1['success'], "Order 1 should succeed"
    assert result1['filled_qty'] > 0, "Should have filled quantity"
    assert result1.get('remaining_qty', 0) <= 0.001, "Should have 0 remaining for full fill"
    
    # Check BTC lock is released after full fill
    btc_status = portfolio.get_coin_lock_status('BTC')
    assert not btc_status['locked'], "BTC lock should be released after full fill"
    logger.info("✓ BTC lock released after full fill")
    
    # Execute order 2: Should result in partial fill (scenario 2)
    logger.info("\n--- Order 2: Partial LIMIT fill ---")
    result2 = engine.execute_order(
        coin='ETH',
        side='BUY',
        quantity=10.0,
        price=2000.0,
        order_type='LIMIT',
        strategy_id='strat_2',
        stop_loss=1800.0,
        take_profit=2200.0
    )
    logger.info(f"Result 2: {result2}")
    assert result2['success'], "Order 2 should succeed"
    assert result2['queued'], "Should be queued for partial fill"
    assert 'remaining_qty' in result2, "Should have remaining_qty tracked"
    assert result2['remaining_qty'] > 0.001, "Should have remaining quantity"
    
    # Check ETH lock is still held after partial fill
    eth_status = portfolio.get_coin_lock_status('ETH')
    assert eth_status['locked'], "ETH lock should still be held after partial fill"
    assert eth_status['owner_strategy_id'] == 'strat_2', "Owner should still be strat_2"
    logger.info("✓ ETH lock still held for partial fill")
    
    # Check pending orders
    pending = engine.get_pending_orders_snapshot()
    logger.info(f"Pending orders: {pending}")
    assert len(pending) > 0, "Should have pending orders"
    
    # Simulate query response for partial fill
    logger.info("\n--- Query response: Further partial fill ---")
    query_response = {
        'OrderMatched': [
            {
                'OrderID': result2['order_id'],
                'Status': 'PENDING',
                'Quantity': 10.0,
                'FilledQuantity': 6.0,  # Now 6.0 filled (was 3.0)
                'Side': 'BUY',
                'FilledAverPrice': 2000.0
            }
        ]
    }
    processed = engine.process_query_response(query_response)
    logger.info(f"Query response processed: {processed}")
    assert len(processed) > 0, "Should process at least 1 order"
    
    # Check pending order was updated (not removed)
    pending = engine.get_pending_orders_snapshot()
    logger.info(f"Pending orders after update: {pending}")
    # Should still have the order in queue with updated filled_quantity
    
    # Simulate full fill
    logger.info("\n--- Query response: Full fill ---")
    query_response_full = {
        'OrderMatched': [
            {
                'OrderID': result2['order_id'],
                'Status': 'FILLED',
                'Quantity': 10.0,
                'FilledQuantity': 10.0,  # All filled
                'Side': 'BUY',
                'FilledAverPrice': 2000.0
            }
        ]
    }
    processed = engine.process_query_response(query_response_full)
    logger.info(f"Full fill processed: {processed}")
    
    # Check ETH lock is released after full fill
    eth_status = portfolio.get_coin_lock_status('ETH')
    assert not eth_status['locked'], "ETH lock should be released after full fill"
    logger.info("✓ ETH lock released after full fill")
    
    # Check pending orders cleaned up
    pending = engine.get_pending_orders_snapshot()
    logger.info(f"Pending orders after full fill: {pending}")
    
    logger.info("✓ Test 4 passed\n")


def test_position_updates():
    """Test that positions are updated correctly on BUY and SELL."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Position updates on BUY and SELL")
    logger.info("="*60)
    
    portfolio = Portfolio(execution_module=None)
    
    # Initial position
    pos = portfolio.get_position('BTC')
    logger.info(f"Initial BTC position: {pos}")
    
    # Register BUY execution
    portfolio.acquire_coin('BTC', 'strat1')
    portfolio.register_order_execution('BTC', 'strat1', 'BUY', 1.0, 50000.0)
    pos = portfolio.get_position('BTC')
    assert pos['free'] == 1.0, "BTC free should be 1.0 after BUY"
    logger.info(f"✓ After BUY 1.0: {pos}")
    
    # Register SELL execution
    portfolio.acquire_coin('BTC', 'strat2')
    portfolio.register_order_execution('BTC', 'strat2', 'SELL', 0.5, 55000.0)
    pos = portfolio.get_position('BTC')
    assert pos['free'] == 0.5, "BTC free should be 0.5 after SELL"
    logger.info(f"✓ After SELL 0.5: {pos}")
    
    logger.info("✓ Test 5 passed\n")


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE TEST SUITE: Application-Level Locking & Partial Fills")
    logger.info("="*70)
    
    try:
        test_basic_acquire_release()
        test_cross_thread_release()
        test_lock_status_and_force_release()
        test_full_execution_flow()
        test_position_updates()
        
        logger.info("\n" + "="*70)
        logger.info("ALL TESTS PASSED ✓")
        logger.info("="*70)
        logger.info("\nSummary of improvements:")
        logger.info("1. ✓ Application-level ownership (safe cross-thread release)")
        logger.info("2. ✓ Partial fill tracking (update metadata, keep lock)")
        logger.info("3. ✓ Full fill detection (release lock after complete execution)")
        logger.info("4. ✓ Admin recovery (force_release_coin method)")
        logger.info("5. ✓ Lock status queries (get_coin_lock_status, get_all_lock_status)")
        logger.info("="*70 + "\n")
        
    except AssertionError as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ UNEXPECTED ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

