import pytest
import sys
from unittest.mock import MagicMock

# Mock third-party modules that might be missing in the environment,
# as recommended by project memory guidelines.
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['websockets'] = MagicMock()

try:
    from BITGOT_ETAP1_foundation import Action
except ImportError:
    pytest.skip("Could not import BITGOT_ETAP1_foundation", allow_module_level=True)

def test_action_is_bearish():
    # Test that STRONG_SELL and SELL return True
    assert Action.STRONG_SELL.is_bearish() is True
    assert Action.SELL.is_bearish() is True

    # Test that other actions return False
    assert Action.STRONG_BUY.is_bearish() is False
    assert Action.BUY.is_bearish() is False
    assert Action.HOLD.is_bearish() is False
