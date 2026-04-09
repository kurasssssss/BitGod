import sys
from unittest.mock import patch, MagicMock

# Mock out external dependencies to allow importing BITGOT_ETAP1_foundation
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()

import pytest
from BITGOT_ETAP1_foundation import BotState, BotStatus, BotTier, MarketType, CFG

@pytest.fixture
def base_bot():
    """Returns a BotState object with default valid trading conditions."""
    bot = BotState(bot_id=1, symbol="BTCUSDT", market_type=MarketType.FUTURES_USDT)
    bot.status = BotStatus.SCANNING
    bot.paused_until = 0.0
    bot.error_count = 0
    return bot

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_success(mock_ts, base_bot):
    """Test happy path where all conditions to trade are met."""
    base_bot.paused_until = 50.0 # Past
    assert base_bot.can_trade is True

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_status_dead(mock_ts, base_bot):
    """Test failure when bot status is DEAD."""
    base_bot.status = BotStatus.DEAD
    assert base_bot.can_trade is False

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_status_paused(mock_ts, base_bot):
    """Test failure when bot status is PAUSED."""
    base_bot.status = BotStatus.PAUSED
    assert base_bot.can_trade is False

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_status_healing(mock_ts, base_bot):
    """Test failure when bot status is HEALING."""
    base_bot.status = BotStatus.HEALING
    assert base_bot.can_trade is False

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_paused_until_future(mock_ts, base_bot):
    """Test failure when bot is paused until a time in the future."""
    base_bot.paused_until = 150.0 # Future time compared to mock_ts return value
    assert base_bot.can_trade is False

@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_paused_until_exact(mock_ts, base_bot):
    """Test boundary condition where paused_until is exactly current time."""
    base_bot.paused_until = 100.0 # Exact time (should pass based on >=)
    assert base_bot.can_trade is True

@patch.object(CFG, 'max_bot_errors', 5)
@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_error_count_exceeded(mock_ts, base_bot):
    """Test failure when error count meets or exceeds the maximum configured limit."""
    base_bot.error_count = 5
    assert base_bot.can_trade is False

    base_bot.error_count = 6
    assert base_bot.can_trade is False

@patch.object(CFG, 'max_bot_errors', 5)
@patch('BITGOT_ETAP1_foundation._TS', return_value=100.0)
def test_can_trade_error_count_just_below(mock_ts, base_bot):
    """Test boundary condition where error count is just below the maximum limit."""
    base_bot.error_count = 4
    assert base_bot.can_trade is True
