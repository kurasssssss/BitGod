import sys
from unittest.mock import MagicMock

# Mocking missing dependencies before importing the module under test
sys.modules["numpy"] = MagicMock()
sys.modules["ccxt"] = MagicMock()
sys.modules["ccxt.async_support"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["fastapi"] = MagicMock()
sys.modules["uvicorn"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["websockets"] = MagicMock()

import pytest
from BITGOT_ETAP1_foundation import BotTrade

def test_bot_trade_won_positive_pnl():
    """Test that won property returns True when pnl is positive."""
    trade = BotTrade(pnl=10.0)
    assert trade.won is True

def test_bot_trade_won_negative_pnl():
    """Test that won property returns False when pnl is negative."""
    trade = BotTrade(pnl=-5.0)
    assert trade.won is False

def test_bot_trade_won_zero_pnl():
    """Test that won property returns False when pnl is zero."""
    trade = BotTrade(pnl=0.0)
    assert trade.won is False
