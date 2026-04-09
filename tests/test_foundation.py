import pytest
from BITGOT_ETAP1_foundation import BotState, BotTier, BotStatus, MarketType

def test_botstate_wr_zero_trades():
    bot = BotState(
        bot_id=1,
        symbol="BTC/USDT",
        market_type=MarketType.FUTURES_USDT,
        tier=BotTier.SCOUT,
        status=BotStatus.INITIALIZING,
        n_trades=0,
        n_wins=0
    )
    assert bot.wr == 0.0

def test_botstate_wr_with_trades():
    bot = BotState(
        bot_id=1,
        symbol="BTC/USDT",
        market_type=MarketType.FUTURES_USDT,
        tier=BotTier.SCOUT,
        status=BotStatus.INITIALIZING,
        n_trades=10,
        n_wins=5
    )
    assert bot.wr == 0.5
