import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BITGOT_ETAP1_foundation import BitgetConnector, BITGOTConfig, CFG

@pytest.fixture
def connector():
    conn = BitgetConnector(cfg=CFG)
    conn._healthy = True
    conn._ex = AsyncMock()
    # Wygaszamy rate limiting dla szybszych testów
    conn._rate_limit = AsyncMock()
    return conn

@pytest.mark.asyncio
async def test_fetch_ticker_unhealthy(connector):
    connector._healthy = False
    res = await connector.fetch_ticker("BTC/USDT")
    assert res == {}

@pytest.mark.asyncio
async def test_fetch_ticker_success(connector):
    mock_ticker = {"symbol": "BTC/USDT", "last": 50000}

    # Kiedy _ex.fetch_ticker zostanie zawaitowane, zwróci mock_ticker
    async def mock_fetch(symbol):
        return mock_ticker

    connector._ex.fetch_ticker.side_effect = mock_fetch

    res = await connector.fetch_ticker("BTC/USDT")
    assert res == mock_ticker
    connector._ex.fetch_ticker.assert_called_once_with("BTC/USDT")

@pytest.mark.asyncio
async def test_fetch_ticker_error_auth(connector):
    async def mock_fetch(symbol):
        raise Exception("Invalid API key provided")

    connector._ex.fetch_ticker.side_effect = mock_fetch

    res = await connector.fetch_ticker("BTC/USDT")
    assert res == {}
    connector._ex.fetch_ticker.assert_called_once_with("BTC/USDT")

@pytest.mark.asyncio
async def test_fetch_ticker_error_rate_limit(connector):
    calls = 0
    async def mock_fetch(symbol):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("Rate limit exceeded")
        return {"symbol": "BTC/USDT", "last": 50000}

    connector._ex.fetch_ticker.side_effect = mock_fetch

    # Przyspieszamy sleep by testy były szybkie
    original_sleep = asyncio.sleep
    async def fast_sleep(wait):
        await original_sleep(0)

    # Patchujemy sleep żeby nie czekać 60s
    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", fast_sleep)
        res = await connector.fetch_ticker("BTC/USDT")

    assert res == {"symbol": "BTC/USDT", "last": 50000}
    assert calls == 2
    assert connector._ex.fetch_ticker.call_count == 2

@pytest.mark.asyncio
async def test_fetch_ticker_error_timeout_retry(connector):
    calls = 0
    async def mock_fetch(symbol):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise Exception("Timeout Error")
        return {"symbol": "BTC/USDT", "last": 50000}

    connector._ex.fetch_ticker.side_effect = mock_fetch

    original_sleep = asyncio.sleep
    async def fast_sleep(wait):
        await original_sleep(0)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", fast_sleep)
        res = await connector.fetch_ticker("BTC/USDT")

    assert res == {"symbol": "BTC/USDT", "last": 50000}
    assert calls == 3
    assert connector._ex.fetch_ticker.call_count == 3

@pytest.mark.asyncio
async def test_fetch_ticker_all_retries_fail(connector):
    async def mock_fetch(symbol):
        raise Exception("Timeout Error")

    connector._ex.fetch_ticker.side_effect = mock_fetch

    original_sleep = asyncio.sleep
    async def fast_sleep(wait):
        await original_sleep(0)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(asyncio, "sleep", fast_sleep)
        res = await connector.fetch_ticker("BTC/USDT")

    assert res == {}
    assert connector._ex.fetch_ticker.call_count == connector.MAX_RETRIES
