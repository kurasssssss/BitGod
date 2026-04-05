import asyncio
from typing import Dict, Optional, Tuple
from BITGOT_ETAP1_foundation import (
    Action, StateVector, SignalState, MarketDataCache, FeatureBuilder
)
from .godmind import GodmindController
from .rl_engines import PPOEngine, DQNEngine

class CompositeSignalBuilder:
    """Buduje ostateczny sygnał dla ETAP3"""
    def __init__(self, bot_id: int):
        self.bot_id = bot_id
        self.godmind = GodmindController()

        # Inicjalizacja wybranych silników
        self.engines = {
            "PPO": PPOEngine("SYMBOL", bot_id),
            "DQN": DQNEngine("SYMBOL", bot_id)
        }

        for name, eng in self.engines.items():
            self.godmind.register_engine(name, eng.WEIGHT)

    async def build_signal(self, symbol: str, cache: MarketDataCache) -> Optional[SignalState]:
        sv = FeatureBuilder.build(cache.get(symbol))
        if not sv:
            return None

        preds = {}
        for name, eng in self.engines.items():
            eng.symbol = symbol # Update symbol dynamically for base architecture
            preds[name] = eng.act(sv)

        final_action, conf = self.godmind.aggregate(preds)

        # Proste zarządzanie sygnałem
        if conf > 0.6 and final_action in [Action.BUY, Action.STRONG_BUY]:
            return SignalState(symbol=symbol, action=final_action, confidence=conf, leverage=50)
        elif conf > 0.6 and final_action in [Action.SELL, Action.STRONG_SELL]:
            return SignalState(symbol=symbol, action=final_action, confidence=conf, leverage=50)

        return None

    def feed_reward(self, symbol: str, pnl_pct: float):
        """Metoda dla ETAP3 by zwrócić PNL do silników w ramach meta-learningu."""
        pass
