import unittest
import sys
import os

# Add parent directory to path to import BITGOT_ETAP1_foundation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BITGOT_ETAP1_foundation import CapitalEngine, CFG

class TestCapitalEngine(unittest.TestCase):
    def setUp(self):
        # We can create a fresh CapitalEngine for each test
        self.engine = CapitalEngine(start_capital=3000.0)

    def test_position_size_usd_defaults(self):
        # Default behavior:
        # CFG.base_position_pct = 0.0333
        # portfolio = 3000
        # base = 3000 * 0.0333 / 100 = 0.999
        # Tier NANO base_margin_usd = 0.50
        # base = max(0.999, 0.50) = 0.999

        # Kelly: edge = 0.8 * 0.5 - (1-0.8) * (1-0.5) = 0.40 - 0.10 = 0.30
        # kelly = 0.30 / 0.8 * 0.25 = 0.09375
        # clipped kelly = np.clip(0.09375, 0.5, 1.5) = 0.5

        # conf_mult = 1.0 + (0.80 - 0.80) * 2.0 = 1.0
        # clipped conf_mult = np.clip(1.0, 0.8, 1.5) = 1.0

        # size = 0.999 * 0.5 * 1.0 = 0.4995
        # min_position_usd = 1.0
        # max_size = 3000 * 0.10 / 100 = 3.0 (CFG.max_position_pct = 0.10)
        # clipped size = np.clip(0.4995, 1.0, 3.0) = 1.0

        size = self.engine.position_size_usd()
        self.assertAlmostEqual(size, 1.0, places=4)

    def test_position_size_usd_high_confidence_high_wr(self):
        # Test with high portfolio, confidence and win rate
        # Portfolio = 100_000, confidence = 0.95, win_rate = 0.90
        # CFG.base_position_pct = 0.0333
        # base = 100_000 * 0.0333 / 100 = 33.3
        # Tier XLARGE base_margin_usd = 50.0
        # base = max(33.3, 50.0) = 50.0

        # Kelly: edge = 0.95 * 0.9 - (1-0.95) * (1-0.9) = 0.855 - 0.05 * 0.1 = 0.85
        # kelly = 0.85 / 0.95 * 0.25 = 0.22368
        # clipped kelly = np.clip(0.22368, 0.5, 1.5) = 0.5

        # conf_mult = 1.0 + (0.95 - 0.80) * 2.0 = 1.3
        # clipped conf_mult = np.clip(1.3, 0.8, 1.5) = 1.3

        # size = 50.0 * 0.5 * 1.3 = 32.5
        # max_size = 100_000 * 0.10 / 100 = 100.0
        # clipped size = np.clip(32.5, 1.0, 100.0) = 32.5

        size = self.engine.position_size_usd(portfolio=100_000.0, confidence=0.95, win_rate=0.90)
        self.assertAlmostEqual(size, 32.5, places=4)

    def test_position_size_usd_clip_max(self):
        # Test clipping to max position size
        # To hit max position size easily, we can temporarily change CFG
        original_max_pct = CFG.max_position_pct
        try:
            # Set max position pct very low to force clipping
            CFG.max_position_pct = 0.005 # 0.005% of 10_000 = 0.5
            # We also need to change min position so max clip works correctly
            original_min_pos = CFG.min_position_usd
            CFG.min_position_usd = 0.1

            size = self.engine.position_size_usd(portfolio=10_000.0, confidence=0.95, win_rate=0.90)
            self.assertAlmostEqual(size, 0.5, places=4)
        finally:
            CFG.max_position_pct = original_max_pct
            CFG.min_position_usd = original_min_pos

    def test_position_size_usd_clip_min(self):
        # Test clipping to min position size
        # Min position size is normally 1.0
        # With portfolio 100, base = 0.0333 (Tier Nano base is 0.5) -> base = 0.5
        # Kelly clip is 0.5, conf clip is 1.0 -> size 0.25
        # Min is 1.0, so it should clip to 1.0. Max is 100 * 0.1/100 = 0.1... wait, clip max is evaluated last!
        # If max_size < min_size, np.clip behavior: np.clip(size, min, max) -> it will return max if max < min in some numpy versions, but actually np.clip(a, a_min, a_max) clips a to a_min, then a_max. So if a < a_min, it becomes a_min. Then if a_min > a_max, it becomes a_max. Let's trace it.
        # min_position = 1.0, max_position = 100 * 0.1 / 100 = 0.1.
        # np.clip(0.25, 1.0, 0.1).
        # Actually in numpy, if a_min > a_max, the result is a_max.

        # Let's verify standard min clip where max > min
        # Portfolio = 3000, base_position_pct = 0.0333 -> max_size = 3.0.
        # Min is 1.0. Size calculated is ~0.5. Result should be 1.0.
        size = self.engine.position_size_usd(portfolio=3000.0, confidence=0.80, win_rate=0.50)
        self.assertAlmostEqual(size, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
