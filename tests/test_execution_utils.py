
import sys
from unittest.mock import MagicMock

# Mock third-party dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['websockets'] = MagicMock()

# Mock bitgot_e2 because it's missing some imports used in ETAP3
mock_e2 = MagicMock()
# Provide the names that ETAP3 expects to import from bitgot_e2
mock_e2.BotBrain = MagicMock()
mock_e2.SwarmIntelligence = MagicMock()
mock_e2.GlobalMetaPool = MagicMock()
mock_e2.CouncilVerdict = MagicMock()
mock_e2.SignalCouncil = MagicMock()
mock_e2.AdversarialShield = MagicMock()
mock_e2.RLEngineCluster = MagicMock()
mock_e2.NeuralSwarm = MagicMock()
mock_e2.MicroSignalEngine = MagicMock()
mock_e2.TIER_ENGINE_CLASSES = {}

sys.modules['bitgot_e2'] = mock_e2

import unittest
# Import directly from the files to avoid potential symlink issues if any
from BITGOT_ETAP1_foundation import PairInfo, BotTier, MarketType
from BITGOT_ETAP3_execution import pair_max_lev

class TestPairMaxLev(unittest.TestCase):
    def test_pair_max_lev_limit_by_pair(self):
        """Case 1: pair.max_leverage is smaller than pair.tier.max_leverage()"""
        # BotTier.APEX.max_leverage() is 125
        pair = PairInfo(
            symbol="BTC/USDT:USDT",
            base="BTC",
            quote="USDT",
            market_type=MarketType.FUTURES_USDT,
            max_leverage=10,
            tier=BotTier.APEX
        )
        result = pair_max_lev(pair)
        self.assertEqual(result, 10)

    def test_pair_max_lev_limit_by_tier(self):
        """Case 2: pair.tier.max_leverage() is smaller than pair.max_leverage"""
        # BotTier.SCOUT.max_leverage() is 25
        pair = PairInfo(
            symbol="BTC/USDT:USDT",
            base="BTC",
            quote="USDT",
            market_type=MarketType.FUTURES_USDT,
            max_leverage=100,
            tier=BotTier.SCOUT
        )
        result = pair_max_lev(pair)
        self.assertEqual(result, 25)

    def test_pair_max_lev_equal(self):
        """Case 3: Both values are equal"""
        # BotTier.SCOUT.max_leverage() is 25
        pair = PairInfo(
            symbol="BTC/USDT:USDT",
            base="BTC",
            quote="USDT",
            market_type=MarketType.FUTURES_USDT,
            max_leverage=25,
            tier=BotTier.SCOUT
        )
        result = pair_max_lev(pair)
        self.assertEqual(result, 25)

if __name__ == '__main__':
    unittest.main()
