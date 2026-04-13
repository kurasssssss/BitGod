import sys
from unittest.mock import MagicMock

# Mock required dependencies before importing BITGOT modules
sys.modules['ccxt'] = MagicMock()
sys.modules['ccxt.async_support'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()
sys.modules['websockets'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['uvicorn'] = MagicMock()
sys.modules['colorama'] = MagicMock()

import pytest
from BITGOT_ETAP1_foundation import GlobalPortfolio

def test_drawdown_pct_normal():
    gp = GlobalPortfolio(peak_capital=1000.0, total_capital=900.0)
    assert gp.drawdown_pct == 0.1

def test_drawdown_pct_zero_peak():
    gp = GlobalPortfolio(peak_capital=0.0, total_capital=0.0)
    assert gp.drawdown_pct == 0.0

def test_drawdown_pct_negative_peak():
    gp = GlobalPortfolio(peak_capital=-100.0, total_capital=-200.0)
    assert gp.drawdown_pct == 0.0

def test_drawdown_pct_zero_drawdown():
    gp = GlobalPortfolio(peak_capital=1000.0, total_capital=1000.0)
    assert gp.drawdown_pct == 0.0

def test_drawdown_pct_negative_total_capital():
    gp = GlobalPortfolio(peak_capital=1000.0, total_capital=-500.0)
    assert gp.drawdown_pct == 1.5

def test_drawdown_pct_total_greater_than_peak():
    gp = GlobalPortfolio(peak_capital=1000.0, total_capital=1100.0)
    assert gp.drawdown_pct == -0.1
