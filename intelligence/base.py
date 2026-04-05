import threading
import logging
from typing import Tuple
from BITGOT_ETAP1_foundation import Action, StateVector

# Hyperparameters
GAMMA      = 0.995
LR_FAST    = 0.002
LR_SLOW    = 0.0004
LR_META    = 0.0001
BATCH      = 64
BUF_CAP    = 100_000
EPS_START  = 0.15
EPS_MIN    = 0.003
EPS_DECAY  = 0.9999
N_ACT      = 5        # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
S_DIM      = 80       # StateVector dim

class BaseEngine:
    NAME     = "BASE"
    SOUL     = ""
    WEIGHT   = 1.00
    SPEC     = ""

    def __init__(self, symbol: str, bot_id: int):
        self.symbol   = symbol
        self.bot_id   = bot_id
        self.epsilon  = EPS_START
        self.n_acts   = 0
        self.n_wins   = 0
        self.win_rate = 0.5
        self._lock    = threading.Lock()
        self._log     = logging.getLogger(f"E.{self.NAME}.{bot_id:04d}")
        self._build()

    def _build(self):
        pass

    def act(self, sv: StateVector) -> Tuple[Action, float]:
        return Action.HOLD, 0.0

    def learn(self, exp: dict):
        pass

    def save(self):
        pass

    def load(self):
        pass
