import pytest
import sys
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="module", autouse=True)
def mock_dependencies():
    """Mock external dependencies for tests to avoid ImportErrors."""
    mocks = {
        'numpy': MagicMock(),
        'pandas': MagicMock(),
        'torch': MagicMock(),
        'ccxt': MagicMock(),
        'ccxt.async_support': MagicMock(),
        'aiohttp': MagicMock(),
        'fastapi': MagicMock(),
        'uvicorn': MagicMock(),
        'psutil': MagicMock(),
        'websockets': MagicMock()
    }
    with patch.dict('sys.modules', mocks):
        yield

def test_action_is_bullish():
    from BITGOT_ETAP1_foundation import Action
    assert Action.STRONG_BUY.is_bullish() is True
    assert Action.BUY.is_bullish() is True

    assert Action.HOLD.is_bullish() is False
    assert Action.SELL.is_bullish() is False
    assert Action.STRONG_SELL.is_bullish() is False

def test_action_is_bearish():
    from BITGOT_ETAP1_foundation import Action
    assert Action.STRONG_SELL.is_bearish() is True
    assert Action.SELL.is_bearish() is True

    assert Action.HOLD.is_bearish() is False
    assert Action.BUY.is_bearish() is False
    assert Action.STRONG_BUY.is_bearish() is False

def test_action_is_strong():
    from BITGOT_ETAP1_foundation import Action
    assert Action.STRONG_BUY.is_strong() is True
    assert Action.STRONG_SELL.is_strong() is True

    assert Action.HOLD.is_strong() is False
    assert Action.BUY.is_strong() is False
    assert Action.SELL.is_strong() is False

def test_action_to_side():
    from BITGOT_ETAP1_foundation import Action
    assert Action.STRONG_BUY.to_side() == "long"
    assert Action.BUY.to_side() == "long"

    assert Action.STRONG_SELL.to_side() == "short"
    assert Action.SELL.to_side() == "short"

    assert Action.HOLD.to_side() == "flat"
