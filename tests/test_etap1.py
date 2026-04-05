import unittest
from unittest.mock import MagicMock
import sys

sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.distributions'] = MagicMock()

from BITGOT_ETAP1_foundation import BotTier, BITGOTConfig

class TestETAP1(unittest.TestCase):
    def test_bot_tier_max_leverage(self):
        self.assertEqual(BotTier.APEX.max_leverage(), 100)
        self.assertEqual(BotTier.ELITE.max_leverage(), 75)
        self.assertEqual(BotTier.STANDARD.max_leverage(), 50)
        self.assertEqual(BotTier.SCOUT.max_leverage(), 25)

    def test_bot_tier_from_wr(self):
        self.assertEqual(BotTier.from_wr(0.85), BotTier.APEX)
        self.assertEqual(BotTier.from_wr(0.76), BotTier.ELITE)
        self.assertEqual(BotTier.from_wr(0.61), BotTier.STANDARD)
        self.assertEqual(BotTier.from_wr(0.51), BotTier.SCOUT)

    def test_config_math_realism(self):
        cfg = BITGOTConfig()
        self.assertEqual(cfg.n_bots, 100)
        self.assertEqual(cfg.start_capital, 3000.0)
        self.assertEqual(cfg.base_position_pct, 1.0)
        self.assertEqual(cfg.max_position_pct, 5.0)

if __name__ == '__main__':
    unittest.main()
