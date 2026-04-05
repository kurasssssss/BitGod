from BITGOT_ETAP1_foundation import Action
from .base import N_ACT
import threading
import numpy as np
import logging

class GodmindController:
    """
    Centralny kontroler - agreguje sygnały od wszystkich silników i podejmuje
    ostateczną decyzję na podstawie ich wag.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._log = logging.getLogger("GODMIND")
        self.weights = {}

    def register_engine(self, name: str, init_weight: float):
        with self._lock:
            self.weights[name] = init_weight

    def aggregate(self, predictions: dict) -> tuple[Action, float]:
        """
        predictions: dict {engine_name: (action, confidence)}
        """
        with self._lock:
            votes = np.zeros(N_ACT)
            total_weight = 0.0

            for name, (action, conf) in predictions.items():
                w = self.weights.get(name, 1.0)
                votes[action.value] += conf * w
                total_weight += w

            if total_weight == 0:
                return Action.HOLD, 0.0

            probs = votes / total_weight
            best_idx = int(np.argmax(probs))
            best_conf = float(probs[best_idx])

            return Action(best_idx), best_conf
