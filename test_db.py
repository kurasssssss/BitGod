import pytest
from BITGOT_ETAP1_foundation import BITGOTDatabase, BotState, BotTier, BotStatus, MarketType, Regime, OmegaError, ErrorCat, Severity, HealAction, HealResult, BotTrade, TradingSignal, Action, SignalState, CFG, BotGenome, PairInfo
import sqlite3
import time
from unittest.mock import MagicMock
import logging

def test_database_flush_error_handling():
    """
    Test error handling during the batch flush process of BITGOTDatabase.
    Simulates a database error (e.g. constraint failure) during executemany
    and verifies that the error is raised correctly or handled according to the design.
    """
    # Setup mock logger to avoid AttributeError during initialization
    BITGOTDatabase._log = logging.getLogger("test_logger")
    db = BITGOTDatabase(":memory:")

    # Enqueue a dummy trade
    trade = BotTrade(
        id="t1_error_test",
        bot_id=999,
        symbol="BTCUSDT",
        market_type=MarketType.FUTURES_USDT.value,
        side=Action.BUY.value,
        entry_price=50000.0,
        exit_price=51000.0,
        qty=0.1,
        notional=5000.0,
        leverage=10,
        pnl=100.0,
        pnl_pct=2.0,
        roi_pct=20.0,
        fees=2.5,
        duration_ms=5000,
        exit_reason="tp",
        signal_conf=0.95,
        regime=Regime.MARKUP.value,
        tier=BotTier.SCOUT.value,
        portfolio_at=2000.0,
        entry_ts=time.time() - 100,
        exit_ts=time.time()
    )

    db.q_trade(trade)

    # Verify the item is in the queue
    assert len(db._trade_q) == 1

    # Mock the database connection's executemany to simulate an error
    mock_conn = MagicMock()
    mock_conn.executemany.side_effect = sqlite3.Error("UNIQUE constraint failed: trades.id")

    # Store the original connection
    original_conn = db._conn
    db._conn = mock_conn

    # Execute flush and ensure it raises the sqlite3.Error
    with pytest.raises(sqlite3.Error, match="UNIQUE constraint failed: trades.id"):
        db._flush()

    # Check that the queue was cleared before the error occurred
    # (as per current code design where clear() is called before executemany)
    assert len(db._trade_q) == 0

    # Restore the original connection
    db._conn = original_conn


def test_database_flush_success():
    """
    Test successful batch flush process of BITGOTDatabase.
    """
    BITGOTDatabase._log = logging.getLogger("test_logger")
    db = BITGOTDatabase(":memory:")

    trade = BotTrade(
        id="t2_success",
        bot_id=1000,
        symbol="ETHUSDT",
        market_type=MarketType.FUTURES_USDT.value,
        side=Action.SELL.value,
        entry_price=3000.0,
        exit_price=2900.0,
        qty=1.0,
        notional=3000.0,
        leverage=5,
        pnl=100.0,
        pnl_pct=3.33,
        roi_pct=16.65,
        fees=1.5,
        duration_ms=10000,
        exit_reason="sl",
        signal_conf=0.8,
        regime=Regime.MARKDOWN.value,
        tier=BotTier.APEX.value,
        portfolio_at=5000.0,
        entry_ts=time.time() - 200,
        exit_ts=time.time()
    )

    db.q_trade(trade)
    assert len(db._trade_q) == 1

    # Execute flush and ensure it does not raise
    db._flush()

    # Queue should be empty
    assert len(db._trade_q) == 0

    # Ensure the trade was saved in the database
    cursor = db._conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id=?", ("t2_success",))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "t2_success"
