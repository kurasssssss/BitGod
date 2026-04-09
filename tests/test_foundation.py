import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies that might not be installed
_MOCK_MODULES = ['numpy', 'ccxt', 'pandas', 'scipy', 'torch', 'aiohttp', 'websockets', 'rich', 'colorama', 'requests']
for mod in _MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest
from BITGOT_ETAP1_foundation import Action

def test_action_to_side_long():
    """Test that bullish actions map to 'long' side."""
    assert Action.STRONG_BUY.to_side() == "long"
    assert Action.BUY.to_side() == "long"

def test_action_to_side_short():
    """Test that bearish actions map to 'short' side."""
    assert Action.STRONG_SELL.to_side() == "short"
    assert Action.SELL.to_side() == "short"

def test_action_to_side_flat():
    """Test that hold action maps to 'flat' side."""
    assert Action.HOLD.to_side() == "flat"

def test_action_is_bullish():
    """Test is_bullish returns True only for bullish actions."""
    assert Action.STRONG_BUY.is_bullish() is True
    assert Action.BUY.is_bullish() is True
    assert Action.HOLD.is_bullish() is False
    assert Action.SELL.is_bullish() is False
    assert Action.STRONG_SELL.is_bullish() is False

def test_action_is_bearish():
    """Test is_bearish returns True only for bearish actions."""
    assert Action.STRONG_BUY.is_bearish() is False
    assert Action.BUY.is_bearish() is False
    assert Action.HOLD.is_bearish() is False
    assert Action.SELL.is_bearish() is True
    assert Action.STRONG_SELL.is_bearish() is True

def test_action_is_strong():
    """Test is_strong returns True only for strong actions."""
    assert Action.STRONG_BUY.is_strong() is True
    assert Action.BUY.is_strong() is False
    assert Action.HOLD.is_strong() is False
    assert Action.SELL.is_strong() is False
    assert Action.STRONG_SELL.is_strong() is True
