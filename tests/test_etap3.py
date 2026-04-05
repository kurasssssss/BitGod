import unittest
from unittest.mock import MagicMock
import sys

sys.modules['numpy'] = MagicMock()
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.distributions'] = MagicMock()

from BITGOT_ETAP3_execution import LivePosition

class TestETAP3(unittest.TestCase):
    def test_live_position_ts(self):
        pos = LivePosition(
            symbol="BTC/USDT",
            side="long",
            margin=10.0,
            notional=100.0,
            leverage=10,
            entry_price=50000.0,
            open_ts=1600000000000,
            sl_price=49000.0,
            tp_price=52000.0,
            bot_id=1,
            qty=0.002
        )
        self.assertEqual(pos.open_ts, 1600000000000)
        self.assertFalse(hasattr(pos, 'entry_ts'))

if __name__ == '__main__':
    unittest.main()
