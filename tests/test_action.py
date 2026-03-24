import sys
import unittest
from unittest.mock import MagicMock

class TestAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mocked_modules = ["numpy", "ccxt", "ccxt.async_support", "aiohttp", "fastapi", "uvicorn", "psutil", "websockets"]
        cls.original_modules = {}

        # Save original modules and mock them
        for mod in cls.mocked_modules:
            if mod in sys.modules:
                cls.original_modules[mod] = sys.modules[mod]
            sys.modules[mod] = MagicMock()

    @classmethod
    def tearDownClass(cls):
        # Restore original modules or remove mocked ones
        for mod in cls.mocked_modules:
            if mod in cls.original_modules:
                sys.modules[mod] = cls.original_modules[mod]
            else:
                if mod in sys.modules:
                    del sys.modules[mod]

        # Also clean up BITGOT_ETAP1_foundation so it can be reimported cleanly by other tests
        if "BITGOT_ETAP1_foundation" in sys.modules:
            del sys.modules["BITGOT_ETAP1_foundation"]

    def test_is_strong(self):
        """Test Action.is_strong() correctly identifies strong buy/sell actions."""
        # Import inside the test after mocks are set up
        from BITGOT_ETAP1_foundation import Action

        # Strong actions should return True
        self.assertTrue(Action.STRONG_BUY.is_strong())
        self.assertTrue(Action.STRONG_SELL.is_strong())

        # Other actions should return False
        self.assertFalse(Action.BUY.is_strong())
        self.assertFalse(Action.SELL.is_strong())
        self.assertFalse(Action.HOLD.is_strong())

if __name__ == "__main__":
    unittest.main()
