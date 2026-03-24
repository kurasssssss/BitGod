
import asyncio
import sys
from collections import Counter
from typing import Set, Dict, List, Optional
from dataclasses import dataclass, field

# Mocking necessary parts from bitgot_e1 and bitgot_e2 as they are imported in E3
class Mock:
    def __getattr__(self, name):
        return Mock()
    def __call__(self, *args, **kwargs):
        return Mock()

sys.modules['bitgot_e1'] = Mock()
sys.modules['bitgot_e2'] = Mock()

# Import the actual classes from E3 if possible, but it might be easier to just
# extract the SignalManager class and its dependencies for a focused test.

# Let's extract the SignalManager class from the modified file to test it.
with open("BITGOT_ETAP3_execution.py", "r") as f:
    lines = f.readlines()

# We need to extract the SignalManager class and anything it depends on.
# Or we can just mock the dependencies.

@dataclass
class TradingSignal:
    bot_id: int
    symbol: str
    side: str
    confidence: float
    margin: float
    leverage: int
    expires_ts: float
    regime: str = ""

@dataclass
class SignalRecord:
    signal: TradingSignal
    received_ts: float = 0.0
    routed_ts: float = 0.0

# Mock other components
class MockPortfolio:
    def check_halt(self): return False, ""
    def open_position(self, margin): pass
    def close_position(self, margin, pnl): pass
    @property
    def snapshot(self):
        class Snap:
            available = 1000000
            active_positions = 0
            total_capital = 1000000
        return Snap()

class MockCapital:
    def current_tier(self, cap):
        class Tier:
            max_concurrent = 1000
        return Tier()

class MockExecutor:
    def __init__(self):
        self.should_succeed = True
    async def execute(self, sig):
        if self.should_succeed:
            return {"pnl": 0}
        return None

class MockExecutorPool:
    def __init__(self):
        self.executor = MockExecutor()
    def get_free_executor(self, bot_id):
        return self.executor

class MockDB:
    def q_signal(self, sig): pass
    def q_trade(self, trade): pass

# Constants that SignalManager expects to be global
MIN_CONFIDENCE = 0.8
def _TS(): return 1000.0

# Paste the modified SignalManager here for testing
from collections import deque, Counter
from contextlib import suppress

class SignalManager:
    CORR_BLOCK_PAIRS  = {
        "BTC": {"ETH", "BTC"},
        "ETH": {"BTC", "ETH"},
        "BNB": {"BNB"},
    }

    def __init__(self, executor_pool, portfolio, capital, db):
        self.executors = executor_pool
        self.port      = portfolio
        self.capital   = capital
        self.db        = db
        self._log      = Mock() # logging.getLogger("BITGOT·SignalMgr")
        self._active_symbols: Set[str] = set()
        self._active_bases: Counter = Counter()
        self._recent_signals: Dict[str, float] = {}
        self._signal_history: deque = deque(maxlen=10_000)
        self._lock = asyncio.Lock()
        self._n_received  = 0; self._n_passed  = 0; self._n_blocked = 0
        self._block_reasons: Counter = Counter()
        self._conf_history: deque = deque(maxlen=500)
        self.DEDUP_WINDOW_S = 30.0

    async def process(self, sig: TradingSignal) -> bool:
        self._n_received += 1
        async with self._lock:
            if _TS() > sig.expires_ts:
                self._block("expired"); return False
            halted, halt_reason = self.port.check_halt()
            if halted:
                self._block(f"portfolio_halt:{halt_reason}"); return False
            if sig.confidence < MIN_CONFIDENCE:
                self._block(f"low_conf:{sig.confidence:.3f}"); return False
            last_ts = self._recent_signals.get(sig.symbol, 0)
            if _TS() - last_ts < self.DEDUP_WINDOW_S:
                self._block("dedup"); return False
            if sig.symbol in self._active_symbols:
                self._block("symbol_active"); return False
            gp = self.port.snapshot
            cap_tier = self.capital.current_tier(gp.total_capital)
            if gp.active_positions >= cap_tier.max_concurrent:
                self._block("max_positions"); return False
            if sig.margin > gp.available * 0.95:
                self._block("insufficient_capital"); return False

            # 8. Correlation guard (optimized)
            base = sig.symbol.split("/")[0]
            blocked = self.CORR_BLOCK_PAIRS.get(base, set())
            if blocked:
                for b in blocked:
                    if b in self._active_bases:
                        self._block(f"correlation:{b}"); return False

            executor = self.executors.get_free_executor(sig.bot_id)
            if executor is None:
                self._block("no_executor"); return False
            self._active_symbols.add(sig.symbol)
            self._active_bases[sig.symbol.split("/")[0]] += 1
            self._recent_signals[sig.symbol] = _TS()
            self.port.open_position(sig.margin)

        try:
            trade = await executor.execute(sig)
            if trade:
                self._n_passed += 1
                self._conf_history.append(sig.confidence)
                self.db.q_signal(sig)
                self._signal_history.append(SignalRecord(signal=sig, routed_ts=_TS()))
                return True
            else:
                async with self._lock:
                    if sig.symbol in self._active_symbols:
                        self._active_symbols.discard(sig.symbol)
                        base = sig.symbol.split("/")[0]
                        self._active_bases[base] -= 1
                        if self._active_bases[base] <= 0: del self._active_bases[base]
                    self.port.close_position(sig.margin, 0.0)
                self._block("execution_failed")
                return False
        except Exception as e:
            async with self._lock:
                if sig.symbol in self._active_symbols:
                    self._active_symbols.discard(sig.symbol)
                    base = sig.symbol.split("/")[0]
                    self._active_bases[base] -= 1
                    if self._active_bases[base] <= 0: del self._active_bases[base]
                self.port.close_position(sig.margin, 0.0)
            return False

    def release_symbol(self, symbol: str):
        with suppress(Exception):
            if symbol in self._active_symbols:
                self._active_symbols.discard(symbol)
                base = symbol.split("/")[0]
                self._active_bases[base] -= 1
                if self._active_bases[base] <= 0:
                    del self._active_bases[base]

    def _block(self, reason: str):
        self._n_blocked += 1
        self._block_reasons[reason] += 1

async def test_correlation():
    sm = SignalManager(MockExecutorPool(), MockPortfolio(), MockCapital(), MockDB())

    # 1. First signal (BTC) should pass
    sig1 = TradingSignal(1, "BTC/USDT", "long", 0.9, 100, 10, 2000.0)
    res1 = await sm.process(sig1)
    assert res1 is True
    assert "BTC/USDT" in sm._active_symbols
    assert sm._active_bases["BTC"] == 1

    # 2. Second signal (ETH) should be blocked because BTC is active and they correlate
    sig2 = TradingSignal(2, "ETH/USDT", "long", 0.9, 100, 10, 2000.0)
    res2 = await sm.process(sig2)
    assert res2 is False
    assert sm._block_reasons["correlation:BTC"] == 1

    # 3. Third signal (XRP) should pass (no correlation)
    sig3 = TradingSignal(3, "XRP/USDT", "long", 0.9, 100, 10, 2000.0)
    res3 = await sm.process(sig3)
    assert res3 is True
    assert "XRP/USDT" in sm._active_symbols
    assert sm._active_bases["XRP"] == 1

    # 4. Release BTC, ETH should now pass
    sm.release_symbol("BTC/USDT")
    assert "BTC/USDT" not in sm._active_symbols
    assert "BTC" not in sm._active_bases

    res4 = await sm.process(sig2)
    assert res4 is True
    assert "ETH/USDT" in sm._active_symbols
    assert sm._active_bases["ETH"] == 1

    # 5. Test failure path (executor returns None)
    sm.executors.executor.should_succeed = False
    sig5 = TradingSignal(5, "LTC/USDT", "long", 0.9, 100, 10, 2000.0)
    res5 = await sm.process(sig5)
    assert res5 is False
    assert "LTC/USDT" not in sm._active_symbols
    assert "LTC" not in sm._active_bases

    print("Functional test passed!")

if __name__ == "__main__":
    asyncio.run(test_correlation())
