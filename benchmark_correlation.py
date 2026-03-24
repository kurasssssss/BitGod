
import time
import asyncio
from typing import Set, Dict, List

class MockSignal:
    def __init__(self, symbol: str):
        self.symbol = symbol

class SignalManagerOriginal:
    CORR_BLOCK_PAIRS = {
        "BTC": {"ETH", "BTC"},
        "ETH": {"BTC", "ETH"},
        "BNB": {"BNB"},
    }
    def __init__(self):
        self._active_symbols = set()
        self._n_blocked = 0

    def _block(self, reason):
        self._n_blocked += 1

    def process_correlation_original(self, sig: MockSignal) -> bool:
        base = sig.symbol.split("/")[0]
        blocked = self.CORR_BLOCK_PAIRS.get(base, set())
        for act_sym in self._active_symbols:
            act_base = act_sym.split("/")[0]
            if act_base in blocked and base in blocked:
                self._block(f"correlation:{act_sym}"); return False
        return True

class SignalManagerOptimized:
    CORR_BLOCK_PAIRS = {
        "BTC": {"ETH", "BTC"},
        "ETH": {"BTC", "ETH"},
        "BNB": {"BNB"},
    }
    def __init__(self):
        self._active_symbols = set()
        self._active_bases = {} # Counter-like dict: base -> count
        self._n_blocked = 0

    def _block(self, reason):
        self._n_blocked += 1

    def add_active_symbol(self, symbol: str):
        self._active_symbols.add(symbol)
        base = symbol.split("/")[0]
        self._active_bases[base] = self._active_bases.get(base, 0) + 1

    def remove_active_symbol(self, symbol: str):
        if symbol in self._active_symbols:
            self._active_symbols.remove(symbol)
            base = symbol.split("/")[0]
            self._active_bases[base] -= 1
            if self._active_bases[base] <= 0:
                del self._active_bases[base]

    def process_correlation_optimized(self, sig: MockSignal) -> bool:
        base = sig.symbol.split("/")[0]
        blocked = self.CORR_BLOCK_PAIRS.get(base, set())
        if not blocked:
            return True

        # Check if any blocked base is currently active
        for b in blocked:
            if b in self._active_bases:
                self._block(f"correlation:{b}")
                return False
        return True

def benchmark():
    n_active = 3000
    n_signals = 10000

    symbols = [f"SYM{i}/USDT" for i in range(n_active)]
    # Add some correlated ones
    symbols[0] = "BTC/USDT"
    symbols[1] = "ETH/USDT"

    sig_btc = MockSignal("BTC/USDT")
    sig_other = MockSignal("XRP/USDT")

    # Original
    sm_orig = SignalManagerOriginal()
    sm_orig._active_symbols = set(symbols)

    t0 = time.perf_counter()
    for _ in range(n_signals):
        sm_orig.process_correlation_original(sig_other)
        sm_orig.process_correlation_original(sig_btc)
    t1 = time.perf_counter()
    print(f"Original took: {t1-t0:.4f}s")

    # Optimized
    sm_opt = SignalManagerOptimized()
    for sym in symbols:
        sm_opt.add_active_symbol(sym)

    t0 = time.perf_counter()
    for _ in range(n_signals):
        sm_opt.process_correlation_optimized(sig_other)
        sm_opt.process_correlation_optimized(sig_btc)
    t1 = time.perf_counter()
    print(f"Optimized took: {t1-t0:.4f}s")

if __name__ == "__main__":
    benchmark()
