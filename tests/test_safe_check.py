import sys
import os
import unittest
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies to prevent ImportErrors during test collection
for module in ['numpy', 'pandas', 'ccxt', 'torch', 'aiohttp', 'websockets', 'fastapi', 'uvicorn', 'pydantic', 'scipy', 'statsmodels', 'numba', 'py_compile', 'bitgot_data', 'ccxt.async_support']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

import BITGOT_ETAP1_foundation
from BITGOT_ETAP1_foundation import _safe_check, StateVector

class TestSafeCheck(unittest.TestCase):
    def test_safe_check_success_true(self):
        sv = MagicMock(spec=StateVector)
        fn = lambda x: True
        self.assertTrue(_safe_check(fn, sv))

    def test_safe_check_success_false(self):
        sv = MagicMock(spec=StateVector)
        fn = lambda x: False
        self.assertFalse(_safe_check(fn, sv))

    def test_safe_check_exception(self):
        sv = MagicMock(spec=StateVector)
        def raising_fn(x):
            raise ValueError("Test error")
        self.assertFalse(_safe_check(raising_fn, sv))

    def test_safe_check_exception_key_error(self):
        sv = MagicMock(spec=StateVector)
        def raising_fn(x):
            raise KeyError("Key error")
        self.assertFalse(_safe_check(raising_fn, sv))

if __name__ == '__main__':
    unittest.main()
