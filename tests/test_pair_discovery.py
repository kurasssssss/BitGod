import pytest
from unittest.mock import AsyncMock, MagicMock
from BITGOT_ETAP1_foundation import PairDiscovery, BitgetConnector, BITGOTConfig, MarketType, BotTier

@pytest.fixture
def mock_connector():
    connector = MagicMock(spec=BitgetConnector)
    connector.fetch_all_markets = AsyncMock()
    connector.fetch_tickers = AsyncMock()
    return connector

@pytest.fixture
def config():
    cfg = BITGOTConfig()
    cfg.min_volume_usdt_24h = 1000.0
    cfg.max_spread_pct = 1.0
    cfg.min_vol_1h_pct = 0.01
    return cfg

@pytest.mark.asyncio
async def test_discover_no_markets(mock_connector, config):
    mock_connector.fetch_all_markets.return_value = {}

    discovery = PairDiscovery(connector=mock_connector, cfg=config)
    pairs = await discovery.discover()

    assert len(pairs) == config.n_bots
    # Because there are no markets, it should generate synthetic pairs.
    # The pairs are chosen from a predefined list of COINS.
    coins = [
        "BTC","ETH","SOL","XRP","ADA","DOGE","AVAX","DOT","LINK","LTC",
        "UNI","ATOM","NEAR","APT","ARB","OP","MATIC","FIL","TRX","XLM",
        "SUI","INJ","TIA","WIF","PEPE","ORDI","BLUR","IMX","ENS","SNX",
        "CRV","COMP","AAVE","MKR","YFI","SUSHI","1INCH","GRT","LRC","BAT",
        "MANA","SAND","AXS","ENJ","CHZ","GALA","FLOW","ICP","FTM","HBAR",
        "ALGO","VET","EOS","XTZ","ZIL","KSM","DCR","ZEC","DASH","BCH",
        "ETC","XMR","WAVES","QTUM","ONT","ZRX","BAL","REN","OMG","LRC",
        "STORJ","SKL","NMR","OGN","MLN","CELR","BNT","KNC","LOOM","POLY",
        "RUNE","LUNA","DYDX","PERP","GMX","RBN","LYRA","STG","VELO","CKB",
        "STX","THETA","CHR","REEF","ALICE","TLM","SUPER","FARM","MASK","FET",
        "AGIX","OCEAN","NKN","ORN","TORN","POND","AUDIO","BAND","CTSI","DENT",
        "BETA","HERO","AUCTION","NULS","STMX","UTK","WAN","POLS","DF","DODO",
        "ACM","OG","SANTOS","LAZIO","PORTO","ATM","BAR","CITY","INTER","PSG",
        "JUV","ASR","FLOKI","SHIB","BOME","SLERF","WEN","MEW","SAMO","BONK",
    ]
    for p in pairs:
        base_coin = p.symbol.split("/")[0]
        # Base coin might have a suffix like BTC1, so check if it starts with one of the coins
        assert any(base_coin.startswith(c) for c in coins) or base_coin.startswith("SYNTH")

@pytest.mark.asyncio
async def test_discover_with_markets_and_tickers(mock_connector, config):
    mock_connector.fetch_all_markets.return_value = {
        "BTC/USDT:USDT": {
            "active": True,
            "type": "swap",
            "quote": "USDT",
            "limits": {"amount": {"min": 0.01}, "cost": {"min": 5.0}, "leverage": {"max": 125}},
            "precision": {"price": 2, "amount": 4},
            "base": "BTC",
        },
        "ETH/USDT:USDT": {
            "active": True,
            "type": "swap",
            "quote": "USDT",
            "limits": {"amount": {"min": 0.1}, "cost": {"min": 5.0}, "leverage": {"max": 100}},
            "precision": {"price": 2, "amount": 4},
            "base": "ETH",
        },
        "DOGE/USDT:USDT": { # This will be excluded due to low volume
            "active": True,
            "type": "swap",
            "quote": "USDT",
            "limits": {"amount": {"min": 1.0}, "cost": {"min": 5.0}, "leverage": {"max": 50}},
            "precision": {"price": 4, "amount": 0},
            "base": "DOGE",
        }
    }

    mock_connector.fetch_tickers.return_value = {
        "BTC/USDT:USDT": {
            "last": 50000.0,
            "bid": 49990.0,
            "ask": 50010.0,
            "quoteVolume": 2000000.0, # Meets min_volume
            "high": 51000.0,
            "low": 49000.0, # Meets min_vol_1h_pct -> (51000 - 49000) / 50000 = 0.04 > 0.01
        },
        "ETH/USDT:USDT": {
            "last": 3000.0,
            "bid": 2999.0,
            "ask": 3001.0,
            "quoteVolume": 1500000.0,
            "high": 3100.0,
            "low": 2900.0,
        },
        "DOGE/USDT:USDT": {
            "last": 0.1,
            "bid": 0.099,
            "ask": 0.101,
            "quoteVolume": 500.0, # Below min_volume 1000.0
            "high": 0.11,
            "low": 0.09,
        }
    }

    discovery = PairDiscovery(connector=mock_connector, cfg=config)
    pairs = await discovery.discover()

    # DOGE should be filtered out
    assert len(pairs) == 2
    symbols = [p.symbol for p in pairs]
    assert "BTC/USDT:USDT" in symbols
    assert "ETH/USDT:USDT" in symbols

    # Check that properties are set correctly
    btc_pair = next(p for p in pairs if p.symbol == "BTC/USDT:USDT")
    assert btc_pair.market_type == MarketType.FUTURES_USDT
    assert btc_pair.current_price == 50000.0
    assert btc_pair.volume_24h_usdt == 2000000.0

@pytest.mark.asyncio
async def test_discover_api_error_on_tickers(mock_connector, config):
    mock_connector.fetch_all_markets.return_value = {
        "BTC/USDT:USDT": {
            "active": True,
            "type": "swap",
            "quote": "USDT",
        }
    }

    # Simulate an error when fetching tickers
    mock_connector.fetch_tickers.side_effect = Exception("API error")

    discovery = PairDiscovery(connector=mock_connector, cfg=config)
    pairs = await discovery.discover()

    # If fetch_tickers throws an exception, tickers becomes {} and pairs are skipped
    assert len(pairs) == 0
