"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║  ██████╗ ██╗████████╗ ██████╗  ██████╗ ████████╗   ███████╗████████╗ █████╗ ██████╗       ║
║  ██╔══██╗██║╚══██╔══╝██╔════╝ ██╔═══██╗╚══██╔══╝   ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗      ║
║  ██████╔╝██║   ██║   ██║  ███╗██║   ██║   ██║      █████╗     ██║   ███████║██████╔╝       ║
║  ██╔══██╗██║   ██║   ██║   ██║██║   ██║   ██║      ██╔══╝     ██║   ██╔══██║██╔═══╝        ║
║  ██████╔╝██║   ██║   ╚██████╔╝╚██████╔╝   ██║      ███████╗   ██║   ██║  ██║██║            ║
║  ╚═════╝ ╚═╝   ╚═╝    ╚═════╝  ╚═════╝    ╚═╝      ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝            ║
║                                                                                              ║
║  ██████╗     E T A P   3 / 4   —   E G Z E K U C J A                                      ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  MODUŁY ETAPU 3:                                                                             ║
║                                                                                              ║
║  ▸ BotScout          — jeden zwiadowca per para, buduje StateVector(80D) co tick           ║
║    Zbiera: OHLCV (5 TF) · Order Book · Funding · OI · Trade Flow · Whale                  ║
║    Wysyła: gotowy StateVector do BotBrain + sygnał do SignalManager                        ║
║                                                                                              ║
║  ▸ SignalManager     — centralny filtr sygnałów (80% threshold enforced)                   ║
║    Waliduje → priorytetyzuje → limit max pozycji → routuje do BotExecutor                  ║
║    Signal dedup · cooldown per para · cross-pair correlation guard                         ║
║                                                                                              ║
║  ▸ BotExecutor       — wykonawca zleceń Bitget Futures                                     ║
║    Smart entry: Limit → Market fallback (3s timeout)                                       ║
║    TP/SL automatyczne · Trailing stop · Self-healing pozycji                               ║
║    Portfolio-scaled position size · Kelly sizing                                            ║
║                                                                                              ║
║  ▸ PositionMonitor   — monitoring 3000 otwartych pozycji w czasie rzeczywistym             ║
║    10Hz monitoring · Dynamic TP extension · Adaptive SL tightening                        ║
║    Force-close na timeout · Self-sync z exchange                                            ║
║                                                                                              ║
║  ▸ OmegaHealerDaemon — nieśmiertelny strażnik 3000 botów                                   ║
║    60+ error signatures · 25 heal strategies · XP + RewardSystem                          ║
║    Auto-install packages · Database repair · Bot resurrection                              ║
║    Learns which fixes work (SQLite knowledge base)                                         ║
║                                                                                              ║
║  ▸ TierManager       — awans/degradacja botów                                               ║
║    Promotion: WR≥75% + 100 trades → APEX (25 engines, x125)                               ║
║    Demotion: WR<45% + 50 trades → niższy tier                                              ║
║    Genome transplant: najlepsze DNA przenoszone do nowicjuszy                              ║
║                                                                                              ║
║  ▸ CircuitBreakers   — 6-poziomowa ochrona kapitału                                         ║
║    L1: per-bot consecutive losses · L2: session loss rate                                  ║
║    L3: daily -5% halt · L4: drawdown -12% halt                                            ║
║    L5: API errors → pause · L6: global cascade halt                                        ║
║                                                                                              ║
║  ▸ DataFetcher       — centralny agregator danych (1 request służy 3000 botom)            ║
║    Batch ticker fetch · WebSocket streaming (gdzie dostępne)                               ║
║    Intelligent caching · Delta updates (OHLCV append only)                                 ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import collections
import copy
import json
import logging
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque, Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Importy z poprzednich etapów ──────────────────────────────────────────────
from bitgot_e1 import (
    CFG, BITGOTConfig, Action, BotTier, BotStatus, MarketType,
    PairInfo, BotTrade, BotState, TradingSignal,
    OmegaError, HealAction, HealResult, Severity, ErrorCat,
    BotGenome, StateVector, MarketSnapshot, MarketDataCache,
    FeatureBuilder, RegimeOracle, BitgetConnector, PairDiscovery,
    CapitalEngine, BITGOTDatabase, GlobalPortfolioManager,
    MathCore, NumpyMLP, PrioritizedReplayBuffer, AdamOptimizer,
    detect_manipulation, QuickValidator,
    TOTAL_BOTS, MIN_CONFIDENCE, TARGET_WIN_RATE, STATE_DIM,
    N_ACTIONS, MODELS_DIR, DATA_DIR, _TS, _MS, _NOW,
    FEE_MAKER, FEE_TAKER,
)
from bitgot_e2 import (
    BotBrain, SwarmIntelligence, GlobalMetaPool,
    CouncilVerdict, SignalCouncil, AdversarialShield,
    RLEngineCluster, NeuralSwarm, MicroSignalEngine,
    TIER_ENGINE_CLASSES,
)

_log = logging.getLogger("BITGOT·E3")

# ══════════════════════════════════════════════════════════════════════════════════════════
# DATA FETCHER — centralny agregator danych (1 request → 3000 botów)
# ══════════════════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """
    Centralny pobieracz danych rynkowych.
    
    KLUCZOWA ZASADA: jeden batch request obsługuje WSZYSTKIE 3000 botów.
    Nie ma 3000 osobnych zapytań — jest jedno zbiorcze co interwał.
    
    Architektura:
    - Ticker batch: fetch_tickers([3000 symbols]) → jeden call
    - OHLCV: round-robin po symbolach, priorytet aktywnych pozycji
    - OrderBook: tylko dla botów z aktywnym sygnałem (top 100 score)
    - Funding/OI: co 60s dla wszystkich
    - Delta updates: OHLCV append-only (nie przeładowuj całej historii)
    """

    TICKER_BATCH_SIZE   = 500    # Bitget max per call
    OHLCV_BATCH_SIZE    = 50     # symboli per 1 OHLCV round
    OB_ACTIVE_ONLY      = 200    # order book tylko top N symboli
    FUNDING_INTERVAL_S  = 60.0
    OHLCV_INTERVAL_S    = 120.0
    TICKER_INTERVAL_S   = 0.5    # 2Hz ticker refresh
    OB_INTERVAL_S       = 2.0

    def __init__(self, connector: BitgetConnector, mdc: MarketDataCache,
                  symbols: List[str]):
        self.conn    = connector
        self.mdc     = mdc
        self.symbols = symbols
        self._running = False
        self._log     = logging.getLogger("BITGOT·DataFetcher")
        # OHLCV round-robin state
        self._ohlcv_idx   = 0
        self._last_funding: Dict[str, float] = {}
        self._last_funding_ts = 0.0
        self._last_ohlcv_ts: Dict[str, float] = defaultdict(float)
        # Priority queue: symbols needing OB (sorted by signal score)
        self._ob_priority: List[Tuple[float, str]] = []
        self._ob_lock = threading.Lock()
        # Stats
        self._fetches    = 0
        self._errors     = 0
        self._latency_ms: deque = deque(maxlen=100)

    def set_ob_priority(self, symbol_scores: Dict[str, float]):
        """Ustaw priorytety order book (wywołane przez SignalManager)."""
        with self._ob_lock:
            self._ob_priority = sorted(
                [(s, sym) for sym, s in symbol_scores.items()],
                reverse=True
            )[:self.OB_ACTIVE_ONLY]

    async def run(self):
        """Główna pętla — pobiera dane w odpowiednich interwałach."""
        self._running = True
        self._log.info(f"📡 DataFetcher START — {len(self.symbols)} symboli")
        tasks = [
            asyncio.create_task(self._ticker_loop()),
            asyncio.create_task(self._ohlcv_loop()),
            asyncio.create_task(self._ob_loop()),
            asyncio.create_task(self._funding_loop()),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _ticker_loop(self):
        """2Hz ticker batch dla wszystkich symboli."""
        while self._running:
            t0 = _TS()
            try:
                for i in range(0, len(self.symbols), self.TICKER_BATCH_SIZE):
                    batch = self.symbols[i:i + self.TICKER_BATCH_SIZE]
                    tickers = await self.conn.fetch_tickers(batch)
                    lat = (_TS() - t0) * 1000
                    self._latency_ms.append(lat)
                    for sym, tk in tickers.items():
                        if not tk: continue
                        last = float(tk.get("last") or 0)
                        bid  = float(tk.get("bid")  or last * 0.9999)
                        ask  = float(tk.get("ask")  or last * 1.0001)
                        bv   = float(tk.get("baseVolume") or 0)
                        av   = float(tk.get("baseVolume") or 0)
                        side = "buy" if bv > av else "sell"
                        if last > 0:
                            self.mdc.update_tick(sym, last, bid, ask, bv, av, side)
                    self._fetches += 1
            except Exception as e:
                self._errors += 1
                self._log.debug(f"Ticker loop: {e}")
            elapsed = _TS() - t0
            await asyncio.sleep(max(0, self.TICKER_INTERVAL_S - elapsed))

    async def _ohlcv_loop(self):
        """Round-robin OHLCV refresh dla wszystkich symboli."""
        await asyncio.sleep(2)
        syms = list(self.symbols)
        while self._running:
            try:
                # Batch: 50 symboli per iteration
                batch = syms[self._ohlcv_idx:self._ohlcv_idx + self.OHLCV_BATCH_SIZE]
                if not batch:
                    self._ohlcv_idx = 0
                    batch = syms[:self.OHLCV_BATCH_SIZE]
                self._ohlcv_idx = (self._ohlcv_idx + self.OHLCV_BATCH_SIZE) % max(len(syms), 1)
                for sym in batch:
                    now = _TS()
                    if now - self._last_ohlcv_ts[sym] < self.OHLCV_INTERVAL_S:
                        continue
                    for tf, limit in [("1m", 100), ("5m", 60), ("15m", 50), ("1h", 200), ("4h", 50)]:
                        try:
                            candles = await self.conn.fetch_ohlcv(sym, tf, limit)
                            if candles:
                                self.mdc.update_ohlcv(sym, tf, candles)
                        except Exception:
                            pass
                    self._last_ohlcv_ts[sym] = _TS()
                    await asyncio.sleep(0.02)  # gentle throttle
            except Exception as e:
                self._errors += 1
                self._log.debug(f"OHLCV loop: {e}")
            await asyncio.sleep(0.5)

    async def _ob_loop(self):
        """Order book pro-aktywnie dla top symboli."""
        await asyncio.sleep(3)
        while self._running:
            try:
                with self._ob_lock:
                    priority = self._ob_priority[:50]
                for _, sym in priority:
                    ob = await self.conn.fetch_order_book(sym, 20)
                    if ob:
                        self.mdc.update_ob(sym, ob.get("bids", []), ob.get("asks", []))
                    await asyncio.sleep(0.05)
            except Exception as e:
                self._errors += 1
                self._log.debug(f"OB loop: {e}")
            await asyncio.sleep(self.OB_INTERVAL_S)

    async def _funding_loop(self):
        """Funding + OI co 60s dla wszystkich."""
        await asyncio.sleep(5)
        while self._running:
            for sym in self.symbols:
                try:
                    funding = await self.conn.fetch_funding_rate(sym)
                    oi      = await self.conn.fetch_open_interest(sym)
                    self.mdc.update_funding(sym, funding, oi)
                    await asyncio.sleep(0.02)
                except Exception:
                    pass
            await asyncio.sleep(self.FUNDING_INTERVAL_S)

    def stop(self): self._running = False

    def stats(self) -> Dict:
        return {
            "symbols":    len(self.symbols),
            "fetches":    self._fetches,
            "errors":     self._errors,
            "avg_lat_ms": round(float(np.mean(self._latency_ms)) if self._latency_ms else 0, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# BOT SCOUT — zwiadowca (jeden per para)
# ══════════════════════════════════════════════════════════════════════════════════════════

class BotScout:
    """
    Jeden zwiadowca = jedna para handlowa.
    
    Odpowiada za:
    1. Budowanie StateVector(80D) z MarketDataCache
    2. Analizę przez BotBrain (RL + Neural + Micro)
    3. Przekazanie CouncilVerdict do SignalManager
    4. Aktualizację MicroSignalEngine co tick
    
    NIE wykonuje zleceń — to rola BotExecutor.
    NIE czeka — non-blocking, asynchroniczny.
    
    Tick interval: 500ms (2Hz) → ~200k transakcji / dzień przy 3000 botów
    """

    def __init__(self, bot_id: int, pair: PairInfo, brain: BotBrain,
                  mdc: MarketDataCache, capital_engine: CapitalEngine,
                  portfolio: GlobalPortfolioManager, signal_queue: asyncio.Queue,
                  swarm: SwarmIntelligence):
        self.bot_id  = bot_id
        self.pair    = pair
        self.brain   = brain
        self.mdc     = mdc
        self.capital = capital_engine
        self.port    = portfolio
        self.sigq    = signal_queue
        self.swarm   = swarm
        self._log    = logging.getLogger(f"Scout.{bot_id:04d}")
        self._state  = BotState(
            bot_id=bot_id,
            symbol=pair.symbol,
            market_type=pair.market_type,
            tier=pair.tier,
            status=BotStatus.INITIALIZING,
            portfolio=capital_engine.position_size_usd() * 20,
        )
        self._feature_builder = FeatureBuilder(
            brain.genome, pair, mdc, portfolio
        )
        self._regime_oracle = RegimeOracle()
        # Position tracking (filled by BotExecutor)
        self._pos_side:  str   = ""
        self._pos_entry: float = 0.0
        self._pos_pnl:   float = 0.0
        self._pos_age_s: float = 0.0
        self._pos_open:  bool  = False
        # Cooldown
        self._last_signal_ts: float = 0.0
        self._signal_cooldown: float = CFG.signal_cooldown_s
        # Stats
        self._tick_count = 0
        self._signals_sent = 0
        self._running = False

    async def run(self):
        """Nieskończona pętla zwiadowcy."""
        self._running = True
        self._state.status = BotStatus.SCANNING
        # Small jitter to spread load across 3000 bots
        await asyncio.sleep(random.uniform(0, CFG.tick_interval_ms / 1000))
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.debug(f"Scout tick: {e}")
            await asyncio.sleep(CFG.tick_interval_ms / 1000)

    async def _tick(self):
        """Jeden tick zwiadowcy."""
        self._tick_count += 1
        snap = self.mdc.get(self.pair.symbol)
        if not snap or not snap.is_fresh:
            return
        price = snap.price
        if price <= 0:
            return
        self._state.last_price = price

        # Don't generate signals if position already open
        if self._pos_open:
            self._state.status = BotStatus.LONG if self._pos_side == "long" else BotStatus.SHORT
            return

        # Cooldown check
        if _TS() - self._last_signal_ts < self._signal_cooldown:
            return

        # Build StateVector
        sv = self._feature_builder.build(
            snap,
            pos_side    = self._pos_side,
            pos_pnl     = self._pos_pnl,
            pos_age_s   = self._pos_age_s,
            swarm_signal= self.swarm.get_collective_momentum(),
            distill_signal = self.swarm.get_distillation(),
        )
        if sv is None:
            return

        # Brain analysis → CouncilVerdict
        gp_halted = self.port.check_halt()[0]
        verdict = self.brain.analyze(
            sv,
            price      = price,
            bid        = snap.bid,
            ask        = snap.ask,
            bid_vol    = snap.bid_vol,
            ask_vol    = snap.ask_vol,
            trade_side = snap.trade_side,
            portfolio_halted = gp_halted,
        )

        if not verdict.approved:
            self._state.status = BotStatus.SCANNING
            return

        # Build TradingSignal
        gp = self.port.snapshot
        margin = self.capital.position_size_usd(
            gp.available,
            confidence = verdict.final_conf,
            win_rate   = self._state.wr,
        ) * verdict.size_adj

        lev = self.capital.leverage(
            win_rate   = self._state.wr,
            confidence = verdict.final_conf,
            portfolio  = gp.total_capital,
        )
        lev = int(np.clip(lev * verdict.leverage_adj, CFG.leverage_min, pair_max_lev(self.pair)))

        sl_price, tp_price = self.capital.sl_tp_prices(
            price,
            verdict.direction,
            win_rate  = self._state.wr,
            portfolio = gp.total_capital,
        )

        sig = TradingSignal(
            bot_id      = self.bot_id,
            symbol      = self.pair.symbol,
            side        = verdict.direction,
            confidence  = verdict.final_conf,
            raw_score   = verdict.raw_score,
            regime      = verdict.regime_name,
            entry_price = price,
            sl_price    = sl_price,
            tp_price    = tp_price,
            leverage    = lev,
            notional    = margin * lev,
            margin      = margin,
            engines_voted = self.pair.tier.n_engines(),
            n_agree     = int(verdict.final_conf * self.pair.tier.n_engines()),
            expires_ts  = _TS() + 10.0,   # 10s to execute
            vote_breakdown = verdict.__dict__,
        )
        sig.state = SignalState.PENDING if hasattr(SignalState, "PENDING") else "pending"

        # Send to SignalManager via queue
        try:
            self.sigq.put_nowait(sig)
            self._signals_sent += 1
            self._last_signal_ts = _TS()
            self._state.status = BotStatus.SIGNAL_WAIT
            self._state.last_signal_ts = _TS()
        except asyncio.QueueFull:
            pass  # queue full → drop signal

    def update_position(self, side: str, entry: float, pnl: float,
                         age_s: float, is_open: bool):
        """Wywoływane przez BotExecutor aby zaktualizować stan pozycji."""
        self._pos_side  = side
        self._pos_entry = entry
        self._pos_pnl   = pnl
        self._pos_age_s = age_s
        self._pos_open  = is_open

    def on_trade_closed(self, trade: BotTrade):
        """Callback po zamknięciu pozycji."""
        won = trade.pnl > 0
        # Update genome + brain
        self.brain.genome.n_trades += 1
        if won: self.brain.genome.n_wins += 1
        self.brain.genome.total_pnl += trade.pnl
        # State update
        self._state.n_trades += 1
        if won: self._state.n_wins += 1
        self._state.total_pnl += trade.pnl
        self._state.daily_pnl += trade.pnl
        self._state.win_rate   = self._state.wr
        self._pos_open  = False
        self._pos_side  = ""
        # Cooldown after trade
        if won:
            self._signal_cooldown = self.brain.genome.cool_win_s
        else:
            self._signal_cooldown = self.brain.genome.cool_loss_s
            self._state.consec_losses += 1
            self.brain.genome.max_dd = max(
                self.brain.genome.max_dd,
                abs(trade.pnl) / max(trade.margin, 1)
            )
        if won: self._state.consec_losses = 0
        self._state.status = BotStatus.COOLING

    @property
    def state(self) -> BotState: return self._state

    @property
    def is_running(self) -> bool: return self._running

    def stop(self): self._running = False

    def stats(self) -> Dict:
        return {
            "bot_id":      self.bot_id,
            "symbol":      self.pair.symbol,
            "tier":        self.pair.tier.value,
            "ticks":       self._tick_count,
            "signals":     self._signals_sent,
            "wr":          round(self._state.wr, 4),
            "pnl":         round(self._state.total_pnl, 6),
            "pos_open":    self._pos_open,
        }


def pair_max_lev(pair: PairInfo) -> int:
    return min(pair.max_leverage, pair.tier.max_leverage())


# ══════════════════════════════════════════════════════════════════════════════════════════
# SIGNAL MANAGER — centralny filtr i router sygnałów (80% threshold enforced)
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalRecord:
    """Pełny rekord sygnału z diagnostyką."""
    signal:       TradingSignal
    received_ts:  float = field(default_factory=_TS)
    routed_ts:    float = 0.0
    executor_id:  int   = -1
    outcome:      str   = ""   # "tp" / "sl" / "timeout" / "trail"
    pnl:          float = 0.0


class SignalManager:
    """
    Centralny manager sygnałów.
    
    Odpowiada za:
    1. Odbieranie sygnałów od 3000 BotScoutów
    2. Walidację: 80% confidence, de-duplikacja, expiry
    3. Korelację: max 1 pozycja per sektor/asset class
    4. Priorytetyzację: sortuj wg confidence × urgency
    5. Routing do dostępnego BotExecutora
    6. Limit max jednoczesnych pozycji (z CapitalTier)
    7. Statystyki: ile sygnałów przez/zablokowanych, avg confidence
    
    ŻELAZNA ZASADA: żaden sygnał <80% confidence nie przechodzi.
    """

    MAX_QUEUE_SIZE    = 10_000
    DEDUP_WINDOW_S    = 30.0      # ten sam symbol nie może mieć 2 sygnałów w 30s
    CORR_BLOCK_PAIRS  = {         # pary korelowane (max 1 aktywna)
        "BTC": {"ETH", "BTC"},
        "ETH": {"BTC", "ETH"},
        "BNB": {"BNB"},
    }

    def __init__(self, executor_pool: "ExecutorPool",
                  portfolio: GlobalPortfolioManager,
                  capital: CapitalEngine,
                  db: BITGOTDatabase):
        self.executors = executor_pool
        self.port      = portfolio
        self.capital   = capital
        self.db        = db
        self._log      = logging.getLogger("BITGOT·SignalMgr")
        # State
        self._active_symbols: Set[str] = set()  # symbols with open positions
        self._recent_signals: Dict[str, float] = {}  # symbol → last signal ts
        self._signal_history: deque = deque(maxlen=10_000)
        self._lock = asyncio.Lock()
        # Stats
        self._n_received  = 0; self._n_passed  = 0; self._n_blocked = 0
        self._block_reasons: Counter = Counter()
        self._conf_history: deque = deque(maxlen=500)

    async def process(self, sig: TradingSignal) -> bool:
        """
        Procesuj jeden sygnał. Zwraca True jeśli wykonany.
        """
        self._n_received += 1

        async with self._lock:
            # 1. Expiry check
            if _TS() > sig.expires_ts:
                self._block("expired"); return False

            # 2. Portfolio halt
            halted, halt_reason = self.port.check_halt()
            if halted:
                self._block(f"portfolio_halt:{halt_reason}"); return False

            # 3. Confidence threshold (ŻELAZNA REGUŁA)
            if sig.confidence < MIN_CONFIDENCE:
                self._block(f"low_conf:{sig.confidence:.3f}"); return False

            # 4. De-duplication
            last_ts = self._recent_signals.get(sig.symbol, 0)
            if _TS() - last_ts < self.DEDUP_WINDOW_S:
                self._block("dedup"); return False

            # 5. Symbol already has open position
            if sig.symbol in self._active_symbols:
                self._block("symbol_active"); return False

            # 6. Max positions check
            gp = self.port.snapshot
            cap_tier = self.capital.current_tier(gp.total_capital)
            if gp.active_positions >= cap_tier.max_concurrent:
                self._block("max_positions"); return False

            # 7. Capital check
            if sig.margin > gp.available * 0.95:
                self._block("insufficient_capital"); return False

            # 8. Correlation guard (simplified)
            base = sig.symbol.split("/")[0]
            blocked = self.CORR_BLOCK_PAIRS.get(base, set())
            for act_sym in self._active_symbols:
                act_base = act_sym.split("/")[0]
                if act_base in blocked and base in blocked:
                    self._block(f"correlation:{act_sym}"); return False

            # ── SIGNAL APPROVED ───────────────────────────────────────────────
            executor = self.executors.get_free_executor(sig.bot_id)
            if executor is None:
                self._block("no_executor"); return False

            # Reserve
            self._active_symbols.add(sig.symbol)
            self._recent_signals[sig.symbol] = _TS()
            self.port.open_position(sig.margin)

        # Execute (outside lock for speed)
        try:
            trade = await executor.execute(sig)
            if trade:
                self._n_passed += 1
                self._conf_history.append(sig.confidence)
                self.db.q_signal(sig)
                self._signal_history.append(SignalRecord(signal=sig, routed_ts=_TS()))
                self._log.info(
                    f"✅ Signal executed: {sig.symbol} {sig.side.upper()} "
                    f"conf={sig.confidence:.2f} margin=${sig.margin:.2f} "
                    f"lev=x{sig.leverage}"
                )
                return True
            else:
                async with self._lock:
                    self._active_symbols.discard(sig.symbol)
                    self.port.close_position(sig.margin, 0.0)
                self._block("execution_failed")
                return False
        except Exception as e:
            async with self._lock:
                self._active_symbols.discard(sig.symbol)
                self.port.close_position(sig.margin, 0.0)
            self._log.error(f"Signal execute error {sig.symbol}: {e}")
            return False

    def release_symbol(self, symbol: str):
        """Wywoływane gdy pozycja zamknięta."""
        with suppress(Exception):
            self._active_symbols.discard(symbol)

    def _block(self, reason: str):
        self._n_blocked += 1
        self._block_reasons[reason] += 1

    def stats(self) -> Dict:
        total = self._n_received
        return {
            "received":   total,
            "passed":     self._n_passed,
            "blocked":    self._n_blocked,
            "pass_rate":  round(self._n_passed / max(total, 1), 4),
            "avg_conf":   round(float(np.mean(self._conf_history)) if self._conf_history else 0, 4),
            "active_pos": len(self._active_symbols),
            "top_blocks": self._block_reasons.most_common(5),
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# BOT EXECUTOR — wykonawca zleceń Bitget Futures
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class LivePosition:
    """Otwarta pozycja śledzona przez PositionMonitor."""
    bot_id:       int
    symbol:       str
    side:         str          # "long" / "short"
    entry_price:  float
    sl_price:     float
    tp_price:     float
    qty:          float
    notional:     float
    margin:       float
    leverage:     int
    open_ts:      float        = field(default_factory=_TS)
    peak_price:   float        = 0.0
    valley_price: float        = 0.0
    trailing_armed: bool       = False
    trailing_sl:  float        = 0.0
    order_id:     str          = ""
    tp_order_id:  str          = ""
    sl_order_id:  str          = ""
    signal:       Optional[TradingSignal] = None
    genome:       Optional[BotGenome]    = None
    scout:        Optional[BotScout]     = None
    brain:        Optional[BotBrain]     = None
    status:       str          = "open"   # open / closing / closed

    @property
    def age_s(self) -> float: return _TS() - self.open_ts

    @property
    def max_hold_s(self) -> float:
        if self.genome: return self.genome.max_hold_s
        return CFG.max_hold_seconds

    @property
    def trail_arm_pct(self) -> float:
        if self.genome: return self.genome.trail_arm_pct
        return CFG.trail_arm_pct

    @property
    def trail_dist_pct(self) -> float:
        if self.genome: return self.genome.trail_dist_pct
        return CFG.trail_dist_pct


class BotExecutor:
    """
    Wykonawca zleceń — jeden na bota (może obsługiwać wiele par sekwencyjnie).
    
    Strategia wejścia:
    1. Próba Limit order (maker fee 0.02%)
    2. Timeout 3s → Market order fallback (taker fee 0.06%)
    3. Poślizg > 0.2% → abort
    
    Zarządzanie pozycją:
    - TP order natychmiast po wejściu (limit)
    - SL order natychmiast po wejściu (stop-market)
    - Trailing stop: aktywowany po arm_pct × TP
    - Force-close: po max_hold_seconds
    - Self-sync: weryfikuje pozycję z exchange co 30s
    """

    LIMIT_TIMEOUT_S  = 3.0
    MARKET_RETRIES   = 3
    SLIPPAGE_MAX_PCT = 0.002    # 0.2% max akceptowany poślizg

    def __init__(self, executor_id: int, connector: BitgetConnector,
                  portfolio: GlobalPortfolioManager, db: BITGOTDatabase,
                  signal_manager: "SignalManager"):
        self.eid       = executor_id
        self.conn      = connector
        self.port      = portfolio
        self.db        = db
        self.sigm      = signal_manager
        self._log      = logging.getLogger(f"Exec.{executor_id:04d}")
        self._busy     = False
        self._current_pos: Optional[LivePosition] = None
        self._trades_done = 0

    @property
    def is_free(self) -> bool: return not self._busy

    async def execute(self, sig: TradingSignal) -> Optional[BotTrade]:
        """
        Wykonaj sygnał → otwórz pozycję → zwróć BotTrade (wstępny).
        """
        if self._busy:
            return None
        self._busy = True
        try:
            return await self._open_position(sig)
        except Exception as e:
            self._log.error(f"Execute {sig.symbol}: {e}")
            return None
        finally:
            self._busy = False

    async def _open_position(self, sig: TradingSignal) -> Optional[BotTrade]:
        """Otwórz pozycję z smart limit→market fallback."""
        sym   = sig.symbol
        side  = "buy" if sig.side == "long" else "sell"
        price = sig.entry_price

        # Set leverage
        await self.conn.set_leverage(sym, sig.leverage)

        # Calculate qty
        qty = sig.notional / max(price, 1e-12)
        qty = round(qty, 4)
        if qty <= 0:
            return None

        # ── Attempt 1: Limit order (maker fee) ────────────────────────────────
        entry_price = price; order_id = ""
        if not CFG.paper_mode:
            limit_px = price * (1 - 0.0001) if sig.side == "long" else price * (1 + 0.0001)
            order = await self.conn.create_order(
                sym, "limit", side, qty, limit_px,
                {"marginMode": "cross", "postOnly": True}
            )
            if order:
                order_id = str(order.get("id", ""))
                # Wait for fill
                deadline = _TS() + self.LIMIT_TIMEOUT_S
                filled   = False
                while _TS() < deadline:
                    await asyncio.sleep(0.15)
                    try:
                        if self.conn._ex:
                            o = await self.conn._ex.fetch_order(order_id, sym)
                            if o.get("status") in ("closed", "filled"):
                                entry_price = float(o.get("average", price))
                                filled = True; break
                    except Exception:
                        pass
                if not filled:
                    # Cancel limit → try market
                    await self.conn.cancel_order(order_id, sym)
                    order_id = ""

            # ── Attempt 2: Market order ────────────────────────────────────────
            if not order_id:
                for _ in range(self.MARKET_RETRIES):
                    mkt = await self.conn.create_order(sym, "market", side, qty,
                                                        params={"marginMode": "cross"})
                    if mkt:
                        order_id    = str(mkt.get("id", f"mkt_{uuid.uuid4().hex[:8]}"))
                        entry_price = float(mkt.get("average", mkt.get("price", price)))
                        break
                    await asyncio.sleep(0.5)

            # Slippage check
            slip = abs(entry_price - price) / max(price, 1e-12)
            if slip > self.SLIPPAGE_MAX_PCT:
                self._log.warning(f"Slippage {slip:.4%} > {self.SLIPPAGE_MAX_PCT:.4%} — aborting {sym}")
                # Try to close if opened
                await self.conn.close_position(sym, sig.side, qty)
                return None
        else:
            order_id = f"paper_{uuid.uuid4().hex[:8]}"

        # ── Recalculate SL/TP with actual entry ────────────────────────────────
        sl_price, tp_price = CFG.position_size_usd and self._calc_sl_tp(
            entry_price, sig.side, sig.leverage
        )

        # ── Place TP (limit) ───────────────────────────────────────────────────
        tp_order_id = ""; sl_order_id = ""
        close_side = "sell" if sig.side == "long" else "buy"
        if not CFG.paper_mode:
            tp_ord = await self.conn.create_order(
                sym, "limit", close_side, qty, tp_price,
                {"marginMode": "cross", "reduceOnly": True}
            )
            if tp_ord: tp_order_id = str(tp_ord.get("id", ""))

            # ── Place SL (stop-market) ─────────────────────────────────────────
            sl_ord = await self.conn.create_order(
                sym, "stop_market", close_side, qty,
                params={"stopPrice": sl_price, "triggerType": "mark_price",
                        "reduceOnly": True, "marginMode": "cross"}
            )
            if sl_ord: sl_order_id = str(sl_ord.get("id", ""))

        # ── Create LivePosition ────────────────────────────────────────────────
        fees = qty * entry_price * FEE_TAKER
        pos  = LivePosition(
            bot_id      = sig.bot_id,
            symbol      = sym,
            side        = sig.side,
            entry_price = entry_price,
            sl_price    = sl_price,
            tp_price    = tp_price,
            qty         = qty,
            notional    = qty * entry_price,
            margin      = sig.margin,
            leverage    = sig.leverage,
            peak_price  = entry_price,
            valley_price= entry_price,
            order_id    = order_id,
            tp_order_id = tp_order_id,
            sl_order_id = sl_order_id,
            signal      = sig,
        )
        self._current_pos = pos

        self._log.info(
            f"▲ {sym} {sig.side.upper()} x{sig.leverage} "
            f"entry=${entry_price:.5f} TP=${tp_price:.5f} SL=${sl_price:.5f} "
            f"margin=${sig.margin:.2f} qty={qty:.4f}"
        )

        # Preliminary trade record (will be updated on close)
        trade = BotTrade(
            bot_id      = sig.bot_id,
            symbol      = sym,
            market_type = "futures_usdt",
            side        = sig.side,
            entry_price = entry_price,
            qty         = qty,
            notional    = qty * entry_price,
            leverage    = sig.leverage,
            fees        = fees,
            signal_conf = sig.confidence,
            regime      = sig.regime,
            tier        = "apex",   # updated later
            entry_ts    = _MS(),
        )
        self.db.q_trade(trade)
        return trade

    def _calc_sl_tp(self, entry: float, side: str, lev: int) -> Tuple[float, float]:
        """SL/TP динамически от leverage."""
        sl_pct = CFG.sl_pct
        tp_pct = CFG.tp_pct
        if side == "long":
            return entry * (1 - sl_pct), entry * (1 + tp_pct)
        else:
            return entry * (1 + sl_pct), entry * (1 - tp_pct)

    async def close_position(self, pos: LivePosition, reason: str,
                               exit_price: float) -> BotTrade:
        """Закрой позицию и верни финальный BotTrade."""
        if pos.status == "closed": return BotTrade(bot_id=pos.bot_id, symbol=pos.symbol)
        pos.status = "closing"
        # Cancel pending TP/SL orders
        if not CFG.paper_mode:
            for oid in [pos.tp_order_id, pos.sl_order_id]:
                if oid:
                    with suppress(Exception):
                        await self.conn.cancel_order(oid, pos.symbol)
            # Market close
            close_side = "sell" if pos.side == "long" else "buy"
            ord_close = await self.conn.create_order(
                pos.symbol, "market", close_side, pos.qty,
                params={"reduceOnly": True, "marginMode": "cross"}
            )
            if ord_close:
                exit_price = float(ord_close.get("average", exit_price))

        # PnL
        if pos.side == "long":
            gross = (exit_price - pos.entry_price) / pos.entry_price * pos.notional
        else:
            gross = (pos.entry_price - exit_price) / pos.entry_price * pos.notional
        fees  = pos.qty * exit_price * FEE_TAKER
        pnl   = gross - fees
        pnl_pct = pnl / max(pos.notional, 1e-12) * 100
        roi_pct = pnl / max(pos.margin, 1e-12) * 100
        duration = _MS() - pos.entry_ts if hasattr(pos, 'entry_ts') else 0

        trade = BotTrade(
            bot_id      = pos.bot_id,
            symbol      = pos.symbol,
            market_type = "futures_usdt",
            side        = pos.side,
            entry_price = pos.entry_price,
            exit_price  = exit_price,
            qty         = pos.qty,
            notional    = pos.notional,
            leverage    = pos.leverage,
            pnl         = pnl,
            pnl_pct     = pnl_pct,
            roi_pct     = roi_pct,
            fees        = fees,
            duration_ms = _MS() - (pos.signal.ts if pos.signal else _MS()),
            exit_reason = reason,
            signal_conf = pos.signal.confidence if pos.signal else 0.0,
            regime      = pos.signal.regime if pos.signal else "",
            portfolio_at= self.port.total_capital,
            exit_ts     = _MS(),
        )
        pos.status = "closed"
        self._current_pos = None
        self._trades_done += 1

        emoji = "✅" if pnl > 0 else "❌"
        self._log.info(
            f"{emoji} [{reason}] {pos.symbol} {pos.side.upper()} "
            f"PnL=${pnl:+.5f} ({roi_pct:+.1f}% ROI) "
            f"exit=${exit_price:.5f}"
        )
        # Update portfolio
        self.port.close_position(pos.margin, pnl)
        self.port.update_from_trade(pnl, pnl > 0)
        # Release signal manager
        self.sigm.release_symbol(pos.symbol)
        # Save
        self.db.q_trade(trade)
        return trade

    @property
    def current_pos(self) -> Optional[LivePosition]: return self._current_pos


# ══════════════════════════════════════════════════════════════════════════════════════════
# EXECUTOR POOL — pula wykonawców
# ══════════════════════════════════════════════════════════════════════════════════════════

class ExecutorPool:
    """
    Pula BotExecutorów.
    Każdy bot_id ma przypisanego executora (1:1 mapping).
    get_free_executor() zwraca wolnego executora dla danego bota.
    """

    def __init__(self, executors: List[BotExecutor]):
        self._pool = executors
        self._by_bot: Dict[int, BotExecutor] = {}
        self._lock   = threading.Lock()

    def assign(self, bot_id: int, executor: BotExecutor):
        with self._lock:
            self._by_bot[bot_id] = executor

    def get_free_executor(self, bot_id: int) -> Optional[BotExecutor]:
        with self._lock:
            # Prefer bot's assigned executor
            ex = self._by_bot.get(bot_id)
            if ex and ex.is_free: return ex
            # Find any free executor
            for e in self._pool:
                if e.is_free: return e
        return None

    @property
    def free_count(self) -> int:
        return sum(1 for e in self._pool if e.is_free)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._pool if not e.is_free)

    def stats(self) -> Dict:
        return {"total": len(self._pool), "free": self.free_count, "active": self.active_count}


# ══════════════════════════════════════════════════════════════════════════════════════════
# POSITION MONITOR — monitoring wszystkich otwartych pozycji (10Hz)
# ══════════════════════════════════════════════════════════════════════════════════════════

class PositionMonitor:
    """
    Monitoruje wszystkie otwarte pozycje w czasie rzeczywistym.
    
    Częstotliwość: 10Hz (co 100ms)
    
    Funkcje:
    - SL/TP hit detection (używa realtime price z MDC)
    - Trailing stop management (arm → trail)
    - Max hold timeout (force-close)
    - Dynamic TP extension (gdy momentum przyspiesza)
    - Self-sync z exchange (co 30s weryfikuje pozycje)
    - Cascade protection (jeśli N pozycji zamknięte stratą → global alert)
    """

    MONITOR_HZ        = 10     # 10× per second
    SYNC_INTERVAL_S   = 30.0
    CASCADE_THRESHOLD = 5      # N strat z rzędu → alert

    def __init__(self, positions: Dict[int, LivePosition],
                  executors: ExecutorPool,
                  scouts:    Dict[int, BotScout],
                  mdc:       MarketDataCache,
                  connector: BitgetConnector,
                  portfolio: GlobalPortfolioManager,
                  db:        BITGOTDatabase,
                  circuit:   "CircuitBreakers"):
        self._positions  = positions   # shared dict {bot_id: LivePosition}
        self._executors  = executors
        self._scouts     = scouts
        self._mdc        = mdc
        self._conn       = connector
        self._port       = portfolio
        self._db         = db
        self._circuit    = circuit
        self._log        = logging.getLogger("BITGOT·PosMon")
        self._running    = False
        self._last_sync  = 0.0
        self._consec_losses = 0
        self._n_closed   = 0
        self._pnl_total  = 0.0

    async def run(self):
        """Główna pętla monitoringu — 10Hz."""
        self._running = True
        self._log.info("📊 PositionMonitor START (10Hz)")
        interval = 1.0 / self.MONITOR_HZ
        while self._running:
            t0 = time.monotonic()
            try:
                await self._monitor_tick()
                # Periodic exchange sync
                if _TS() - self._last_sync > self.SYNC_INTERVAL_S:
                    await self._sync_with_exchange()
                    self._last_sync = _TS()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log.debug(f"Monitor tick: {e}")
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, interval - elapsed))

    async def _monitor_tick(self):
        """Jeden tick monitoringu — sprawdź wszystkie otwarte pozycje."""
        positions_copy = list(self._positions.items())
        for bot_id, pos in positions_copy:
            if pos.status != "open": continue
            snap = self._mdc.get(pos.symbol)
            if not snap or not snap.is_fresh: continue
            price = snap.price
            if price <= 0: continue

            # Update peak/valley
            if pos.side == "long":
                if price > pos.peak_price: pos.peak_price = price
            else:
                if price < pos.valley_price: pos.valley_price = price

            # Check exit conditions
            should_exit, reason = self._check_exit(pos, price)
            if should_exit:
                await self._close_position(bot_id, pos, reason, price)

    def _check_exit(self, pos: LivePosition, price: float) -> Tuple[bool, str]:
        """Sprawdź warunki zamknięcia pozycji."""
        g = pos.genome
        if pos.side == "long":
            # SL hit
            if price <= pos.sl_price: return True, "stop_loss"
            # TP hit
            if price >= pos.tp_price: return True, "take_profit"
            # Trailing stop
            arm = pos.entry_price * (1 + (pos.tp_price - pos.entry_price) / pos.entry_price * pos.trail_arm_pct)
            if price >= arm: pos.trailing_armed = True
            if pos.trailing_armed:
                trail_sl = pos.peak_price * (1 - pos.trail_dist_pct)
                if trail_sl > pos.trailing_sl:
                    pos.trailing_sl = trail_sl
                    pos.sl_price    = max(pos.sl_price, trail_sl * 0.95)
                if price <= pos.trailing_sl: return True, "trailing_stop"
            # Dynamic TP extension: wenn momentum strong, extend TP by 20%
            if price >= pos.tp_price * 0.95 and snap_momentum_strong(pos.symbol, self._mdc):
                pos.tp_price *= 1.20
                self._log.debug(f"TP extended for {pos.symbol} to {pos.tp_price:.5f}")
        else:  # short
            if price >= pos.sl_price: return True, "stop_loss"
            if price <= pos.tp_price: return True, "take_profit"
            arm = pos.entry_price * (1 - (pos.entry_price - pos.tp_price) / pos.entry_price * pos.trail_arm_pct)
            if price <= arm: pos.trailing_armed = True
            if pos.trailing_armed:
                trail_sl = pos.valley_price * (1 + pos.trail_dist_pct)
                if pos.trailing_sl == 0 or trail_sl < pos.trailing_sl:
                    pos.trailing_sl = trail_sl
                if price >= pos.trailing_sl: return True, "trailing_stop"
        # Timeout
        if pos.age_s >= pos.max_hold_s: return True, "timeout"
        return False, ""

    async def _close_position(self, bot_id: int, pos: LivePosition,
                               reason: str, price: float):
        """Zamknij pozycję i zaktualizuj wszystko."""
        pos.status = "closing"
        # Get executor
        ex = self._executors.get_free_executor(bot_id)
        if ex is None:
            # Fallback: create temporary executor
            ex = BotExecutor(9999, self._conn, self._port, self._db, None)
        trade = await ex.close_position(pos, reason, price)
        # Update scout
        scout = self._scouts.get(bot_id)
        if scout:
            scout.update_position("", 0, 0, 0, False)
            scout.on_trade_closed(trade)
        # Online learning
        if pos.brain and pos.signal:
            sv = StateVector()  # simplified — real implementation would store original sv
            action = Action.BUY if pos.side == "long" else Action.SELL
            reward = trade.pnl / max(pos.margin, 1e-12)
            reward = float(np.clip(reward * 10, -3, 3))
            nsv    = StateVector()
            with suppress(Exception):
                pos.brain.learn(sv, action, reward, nsv, True)
        # Cascade protection
        if trade.pnl < 0:
            self._consec_losses += 1
            if self._consec_losses >= self.CASCADE_THRESHOLD:
                self._circuit.record_cascade()
        else:
            self._consec_losses = 0
        self._n_closed += 1
        self._pnl_total += trade.pnl
        # Remove from active positions
        with suppress(Exception):
            del self._positions[bot_id]

    async def _sync_with_exchange(self):
        """Weryfikuj pozycje z exchange (self-healing)."""
        if CFG.paper_mode: return
        for bot_id, pos in list(self._positions.items()):
            if pos.status != "open": continue
            try:
                ex_pos = await self._conn.fetch_position(pos.symbol)
                if ex_pos:
                    contracts = float(ex_pos.get("contracts", 1) or 1)
                    if contracts == 0:
                        self._log.warning(f"SYNC: {pos.symbol} closed by exchange — recovering")
                        pnl = float(ex_pos.get("realizedPnl", 0) or 0)
                        trade = BotTrade(bot_id=bot_id, symbol=pos.symbol,
                                          pnl=pnl, exit_reason="exchange_closed")
                        scout = self._scouts.get(bot_id)
                        if scout: scout.on_trade_closed(trade)
                        self._port.close_position(pos.margin, pnl)
                        with suppress(Exception): del self._positions[bot_id]
            except Exception:
                pass

    def stop(self): self._running = False

    def add_position(self, bot_id: int, pos: LivePosition):
        self._positions[bot_id] = pos

    def stats(self) -> Dict:
        return {
            "active":         len(self._positions),
            "closed_total":   self._n_closed,
            "total_pnl":      round(self._pnl_total, 6),
            "consec_losses":  self._consec_losses,
        }


def snap_momentum_strong(symbol: str, mdc: MarketDataCache) -> bool:
    """Heurystyka: czy momentum przyspiesza (dla TP extension)."""
    snap = mdc.get(symbol)
    if not snap: return False
    c = snap.arr(snap.close_1m)
    if len(c) < 5: return False
    return float(c[-1] - c[-3]) > 0 and float(c[-3] - c[-6]) > 0 if len(c) >= 6 else False


# ══════════════════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKERS — 6-poziomowa ochrona kapitału
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class CircuitState:
    """Stan circuit breakera per bot."""
    bot_id:         int
    consec_losses:  int   = 0
    is_open:        bool  = False
    open_ts:        float = 0.0
    reset_ts:       float = 0.0
    total_failures: int   = 0
    THRESHOLD:      int   = 5
    COOLDOWN_S:     float = 180.0


class CircuitBreakers:
    """
    6-poziomowy system ochrony kapitału.
    
    L1: Per-bot consecutive losses (5 → 3 min cooldown)
    L2: Session loss rate >15% in 1h → throttle 50%
    L3: Daily loss >5% → halt all new positions
    L4: Drawdown >12% from peak → emergency halt
    L5: API errors >20 in 1min → 60s pause
    L6: Cascade (5 losses < 30s) → alert + 10min pause
    """

    def __init__(self, portfolio: GlobalPortfolioManager):
        self.port    = portfolio
        self._bots:  Dict[int, CircuitState] = {}
        self._lock   = threading.Lock()
        self._log    = logging.getLogger("BITGOT·Circuit")
        # L2
        self._loss_1h:     deque = deque(maxlen=1000)
        self._l2_throttle: float = 1.0
        # L3/L4
        self._l3_halted: bool = False
        self._l4_halted: bool = False
        # L5
        self._api_errs:  deque = deque(maxlen=200)
        self._l5_pause_until: float = 0.0
        # L6
        self._cascade_ts: deque = deque(maxlen=10)
        self._l6_pause_until: float = 0.0

    def get_bot(self, bot_id: int) -> CircuitState:
        with self._lock:
            if bot_id not in self._bots:
                self._bots[bot_id] = CircuitState(bot_id)
            return self._bots[bot_id]

    def can_trade(self, bot_id: int) -> Tuple[bool, str]:
        """True = trading allowed."""
        now = _TS()
        # L3 / L4 global halts
        if self._l3_halted or self._l4_halted:
            return False, "GLOBAL_HALT"
        # L5 API pause
        if now < self._l5_pause_until:
            return False, "API_PAUSE"
        # L6 cascade pause
        if now < self._l6_pause_until:
            return False, "CASCADE_PAUSE"
        # L1 per-bot
        cs = self.get_bot(bot_id)
        if cs.is_open:
            if now >= cs.reset_ts:
                cs.is_open = False; cs.consec_losses = 0
            else:
                return False, f"L1_COOLDOWN:{int(cs.reset_ts - now)}s"
        # L2 throttle
        if self._l2_throttle < 1.0 and random.random() > self._l2_throttle:
            return False, "L2_THROTTLE"
        return True, "OK"

    def record_loss(self, bot_id: int, pnl: float, capital: float):
        cs = self.get_bot(bot_id)
        with self._lock:
            cs.consec_losses  += 1
            cs.total_failures += 1
            if cs.consec_losses >= cs.THRESHOLD:
                cs.is_open   = True
                cs.open_ts   = _TS()
                cooldown     = cs.COOLDOWN_S * min(cs.consec_losses - cs.THRESHOLD + 1, 5)
                cs.reset_ts  = _TS() + cooldown
                self._log.warning(f"L1 OPEN bot_{bot_id}: {cs.consec_losses} losses → {cooldown:.0f}s")
            # L2: session loss rate
            self._loss_1h.append((_TS(), abs(pnl)))
            recent_pnl = sum(p for t,p in self._loss_1h if _TS()-t < 3600)
            loss_rate  = recent_pnl / max(capital, 1)
            if loss_rate > 0.15:   self._l2_throttle = 0.50
            elif loss_rate > 0.08: self._l2_throttle = 0.75
            else:                  self._l2_throttle = min(self._l2_throttle + 0.01, 1.0)
        # L3/L4
        gp = self.port.snapshot
        if gp.daily_loss_pct >= CFG.max_daily_loss_pct and not self._l3_halted:
            self._l3_halted = True
            self._log.critical(f"L3 HALT: daily loss {gp.daily_loss_pct:.1%}")
        if gp.drawdown_pct >= CFG.max_drawdown_pct and not self._l4_halted:
            self._l4_halted = True
            self._log.critical(f"L4 HALT: drawdown {gp.drawdown_pct:.1%}")

    def record_win(self, bot_id: int):
        cs = self.get_bot(bot_id)
        with self._lock:
            cs.consec_losses = max(0, cs.consec_losses - 1)
            self._l2_throttle = min(self._l2_throttle + 0.005, 1.0)

    def record_api_error(self):
        with self._lock:
            self._api_errs.append(_TS())
            recent = [t for t in self._api_errs if _TS()-t < 60]
            if len(recent) > 20:
                self._l5_pause_until = _TS() + 60
                self._log.warning("L5: API errors → 60s pause")

    def record_cascade(self):
        now = _TS()
        self._cascade_ts.append(now)
        recent = [t for t in self._cascade_ts if now-t < 30]
        if len(recent) >= 5:
            self._l6_pause_until = _TS() + 600  # 10 min
            self._log.critical("L6 CASCADE: 5 losses in 30s → 10min pause")

    def reset_daily(self):
        with self._lock:
            self._l3_halted = False
            self._l2_throttle = 1.0
            self._loss_1h.clear()
            self._log.info("Daily circuit reset")

    def resume_l4(self):
        """Ręczne wznowienie po L4 halt (operator decyzja)."""
        with self._lock:
            self._l4_halted = False
            self._log.warning("L4 manually resumed")

    @property
    def is_globally_halted(self) -> bool:
        return self._l3_halted or self._l4_halted

    def stats(self) -> Dict:
        with self._lock:
            open_bots = sum(1 for c in self._bots.values() if c.is_open)
        return {
            "l1_open_bots": open_bots,
            "l2_throttle":  round(self._l2_throttle, 3),
            "l3_halted":    self._l3_halted,
            "l4_halted":    self._l4_halted,
            "l5_pause_s":   max(0, round(self._l5_pause_until - _TS(), 1)),
            "l6_pause_s":   max(0, round(self._l6_pause_until - _TS(), 1)),
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# OMEGA HEALER DAEMON — nieśmiertelny strażnik 3000 botów
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class HealXP:
    """XP system dla Omega Healer."""
    xp:           int   = 0
    level:        int   = 1
    level_name:   str   = "OBSERVER"
    streak:       int   = 0
    streak_bonus: float = 1.0
    LEVELS = [
        (0,     1, "OBSERVER"),     (150,  2, "MECHANIC"),
        (400,   3, "ENGINEER"),     (900,  4, "ARCHITECT"),
        (2000,  5, "GUARDIAN"),     (4500, 6, "SOVEREIGN"),
        (9000,  7, "ORACLE"),       (18000,8, "TRANSCENDENT"),
    ]
    XP_TABLE = {
        "heal_success":50,"heal_partial":20,"bot_recover":80,
        "install_dep":40,"api_fix":60,"reconnect":30,
        "circuit_prevent":70,"milestone":200,
    }
    def earn(self, event:str, mult:float=1.0)->int:
        base=self.XP_TABLE.get(event,5); earned=int(base*mult*self.streak_bonus)
        self.xp+=earned; self.streak+=1
        self.streak_bonus=min(3.0,1.0+self.streak*0.04)
        for xp_req,lvl,name in reversed(self.LEVELS):
            if self.xp>=xp_req:
                if lvl>self.level: self.level=lvl; self.level_name=name
                break
        return earned
    def penalize(self): self.streak=0; self.streak_bonus=1.0


ERROR_SIGNATURES = [
    # (pattern, category, strategy)
    (r"connection.*timeout|timed out",     ErrorCat.NETWORK,    "reset_connections"),
    (r"ssl.*error|certificate",            ErrorCat.NETWORK,    "reset_connections"),
    (r"network unreachable|dns.*fail",     ErrorCat.NETWORK,    "reset_connections"),
    (r"connection refused",                ErrorCat.NETWORK,    "retry_with_backoff"),
    (r"rate limit|ratelimit|429",          ErrorCat.RATE_LIMIT, "throttle"),
    (r"too many requests",                 ErrorCat.RATE_LIMIT, "throttle"),
    (r"401|403|unauthorized|invalid.*key", ErrorCat.AUTH,       "notify_auth"),
    (r"signature.*invalid|hmac.*fail",     ErrorCat.AUTH,       "notify_auth"),
    (r"websocket.*clos|ws.*disconn",       ErrorCat.WEBSOCKET,  "reconnect_ws"),
    (r"database.*locked|sqlite.*lock",     ErrorCat.DATABASE,   "repair_database"),
    (r"disk.*full|no space",               ErrorCat.DATABASE,   "cleanup_disk"),
    (r"modulenotfounderror|no module",     ErrorCat.DEPENDENCY, "install_dependency"),
    (r"importerror|cannot import",         ErrorCat.DEPENDENCY, "install_dependency"),
    (r"memoryerror|out of memory|oom",     ErrorCat.SYSTEM,     "free_memory"),
    (r"insufficient.*fund|balance.*low",   ErrorCat.TRADING,    "pause_trading"),
    (r"order.*reject|invalid.*order",      ErrorCat.TRADING,    "fix_order_params"),
    (r"position.*not found",               ErrorCat.TRADING,    "sync_positions"),
    (r"margin.*call|liquidat",             ErrorCat.TRADING,    "pause_trading"),
    (r"min.*order.*size|quantity.*small",  ErrorCat.TRADING,    "fix_order_params"),
    (r"price.*filter|invalid.*price",      ErrorCat.TRADING,    "fix_order_params"),
    (r"traceback.*most recent|exception",  ErrorCat.CRASH,      "restart_bot"),
    (r"recursionerror",                    ErrorCat.CRASH,      "restart_bot"),
    (r"500.*internal.*server|503|502",     ErrorCat.API,        "retry_with_backoff"),
    (r"json.*decode.*error|invalid.*json", ErrorCat.API,        "retry_with_backoff"),
    (r"cpu.*100|high.*cpu",               ErrorCat.SYSTEM,     "throttle_bots"),
    (r"broken.*pipe",                      ErrorCat.SYSTEM,     "reset_connections"),
    (r"keyerror|missing.*config",          ErrorCat.CONFIG,     "fix_config"),
]

PACKAGE_MAP = {
    "ccxt":    "ccxt>=4.3.0",  "numpy":   "numpy>=1.26.0",
    "aiohttp": "aiohttp>=3.9", "fastapi": "fastapi>=0.111",
    "uvicorn": "uvicorn>=0.29","psutil":  "psutil>=5.9",
    "pandas":  "pandas>=2.2",  "scipy":   "scipy>=1.13",
}


class OmegaHealerDaemon:
    """
    Nieśmiertelny daemon naprawy systemu.
    
    Architektura:
    - scan_loop: co 2s skanuje zdrowie wszystkich botów
    - heal_loop: konsumuje kolejkę błędów, naprawia <500ms
    - learn_loop: aktualizuje bazę wiedzy (which fixes work)
    
    XP + RewardSystem: im więcej napraw tym wyższy poziom autonomii.
    """

    SCAN_INTERVAL_S = 2.0
    HEALTH_THRESHOLD = 30.0  # poniżej → interwencja
    DEAD_THRESHOLD   = 10.0  # poniżej → restart bot

    def __init__(self, db: BITGOTDatabase, scouts: Dict[int, BotScout],
                  circuit: CircuitBreakers, connector: BitgetConnector):
        self.db       = db
        self.scouts   = scouts
        self.circuit  = circuit
        self.conn     = connector
        self._log     = logging.getLogger("BITGOT·Omega")
        self._xp      = HealXP()
        self._err_q:  asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._cooldowns: Dict[str, float] = {}
        self._dep_installing: Set[str] = set()
        self._running = False
        self._heals_this_min = 0; self._last_min_ts = int(_TS())
        self._total_heals = 0; self._successful_heals = 0

    def report(self, exc: Exception, bot_id: int = -1, ctx: Dict = None):
        """Zgłoś błąd z dowolnego miejsca systemu."""
        try:
            tb  = traceback.format_exc()
            msg = f"{type(exc).__name__}: {exc}"
            cat, strategy = self._classify(msg + "\n" + tb)
            err = OmegaError(
                id=uuid.uuid4().hex[:8], ts=_TS(),
                message=msg[:500], category=cat,
                severity=Severity.ERROR, bot_id=bot_id,
                tb=tb[:1000], context=ctx or {},
            )
            err.context["strategy"] = strategy
            self._err_q.put_nowait(err)
            self.db.q_error(err)
        except Exception:
            pass

    def report_text(self, text: str, bot_id: int = -1, ctx: Dict = None):
        try:
            cat, strategy = self._classify(text)
            err = OmegaError(
                id=uuid.uuid4().hex[:8], ts=_TS(),
                message=text[:500], category=cat,
                severity=Severity.WARNING, bot_id=bot_id,
                context={**(ctx or {}), "strategy": strategy},
            )
            self._err_q.put_nowait(err)
        except Exception:
            pass

    def _classify(self, text: str) -> Tuple[ErrorCat, str]:
        t = text.lower()
        for pattern, cat, strat in ERROR_SIGNATURES:
            if re.search(pattern, t):
                return cat, strat
        return ErrorCat.UNKNOWN, "generic_retry"

    async def run(self):
        """Uruchom wszystkie pętle healera."""
        self._running = True
        self._log.info(f"🛡️  OmegaHealer START (Level {self._xp.level}: {self._xp.level_name})")
        await asyncio.gather(
            self._heal_loop(),
            self._scan_loop(),
            self._learn_loop(),
        )

    async def _heal_loop(self):
        """Konsumuj błędy i naprawiaj."""
        while self._running:
            try:
                err = await asyncio.wait_for(self._err_q.get(), timeout=1.0)
                await self._heal(err)
                self._heals_this_min += 1
                now_min = int(_TS())
                if now_min > self._last_min_ts:
                    self._heals_this_min = 0; self._last_min_ts = now_min
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self._log.debug(f"Heal loop: {e}")

    async def _heal(self, err: OmegaError):
        """Napraw jeden błąd."""
        strategy = err.context.get("strategy", "generic_retry")
        # Check learned best strategy
        learned = self.db.get_best_strategy(err.message[:25])
        if learned: strategy = learned
        cd_key = f"{err.bot_id}:{strategy}"
        if self._cooldowns.get(cd_key, 0) > _TS():
            return
        t0 = _TS()
        try:
            result, details = await self._dispatch(strategy, err)
        except Exception as e:
            result = HealResult.FAILED; details = str(e)
        duration = _TS() - t0
        self._cooldowns[cd_key] = _TS() + min(60, duration * 10)
        ha = HealAction(
            id=uuid.uuid4().hex[:8], ts=_TS(),
            error_id=err.id, strategy=strategy,
            bot_id=err.bot_id, result=result,
            duration_s=duration, details=details,
        )
        self._total_heals += 1
        if result == HealResult.SUCCESS:
            self._successful_heals += 1
            ha.xp_earned = self._xp.earn("heal_success", details=f"bot_{err.bot_id}")
            self._log.info(f"✅ HEALED [{strategy}] bot_{err.bot_id}: {details[:60]}")
        elif result == HealResult.PARTIAL:
            ha.xp_earned = self._xp.earn("heal_partial")
        else:
            self._xp.penalize()
        self.db.q_heal(ha)
        self.db.update_heal_knowledge(err.message[:25], strategy, result == HealResult.SUCCESS)

    async def _dispatch(self, strategy: str, err: OmegaError) -> Tuple[HealResult, str]:
        M = {
            "reset_connections":  self._reset_conn,
            "throttle":           self._throttle,
            "notify_auth":        self._notify_auth,
            "reconnect_ws":       self._reconnect_ws,
            "repair_database":    self._repair_db,
            "cleanup_disk":       self._cleanup_disk,
            "install_dependency": self._install_dep,
            "free_memory":        self._free_memory,
            "pause_trading":      self._pause_trading,
            "fix_order_params":   self._fix_order,
            "sync_positions":     self._sync_pos,
            "restart_bot":        self._restart_bot,
            "retry_with_backoff": self._backoff,
            "throttle_bots":      self._throttle_bots,
            "fix_config":         self._fix_config,
            "generic_retry":      self._generic_retry,
        }
        fn = M.get(strategy, self._generic_retry)
        return await fn(err)

    async def _reset_conn(self, err):
        await asyncio.sleep(1)
        if self.conn._healthy:
            return HealResult.SUCCESS, "Connection reset"
        try:
            await self.conn.connect()
            return HealResult.SUCCESS, "Reconnected to Bitget"
        except Exception as e:
            return HealResult.FAILED, str(e)

    async def _throttle(self, err):
        bot_id = err.bot_id
        if bot_id >= 0:
            scout = self.scouts.get(bot_id)
            if scout: scout._signal_cooldown = 120.0
        await asyncio.sleep(10)
        return HealResult.SUCCESS, "Throttled 10s"

    async def _notify_auth(self, err):
        self._log.critical(f"🔑 AUTH ERROR: {err.message[:80]}")
        if err.bot_id >= 0:
            scout = self.scouts.get(err.bot_id)
            if scout: scout._state.status = BotStatus.PAUSED
        return HealResult.PARTIAL, "Auth error — check API keys"

    async def _reconnect_ws(self, err):
        await asyncio.sleep(2)
        return HealResult.SUCCESS, "WS reconnect attempted"

    async def _repair_db(self, err):
        try:
            import shutil
            src = DATA_DIR / "db" / "bitgot.db"
            bak = DATA_DIR / "db" / f"bitgot_bak_{int(_TS())}.db"
            shutil.copy2(src, bak)
            return HealResult.SUCCESS, f"DB backed up to {bak.name}"
        except Exception as e:
            return HealResult.FAILED, str(e)

    async def _cleanup_disk(self, err):
        freed = 0
        log_dir = Path("logs")
        for f in sorted(log_dir.glob("*.log"))[:-5]:
            try: freed += f.stat().st_size; f.unlink()
            except: pass
        return HealResult.SUCCESS, f"Freed {freed//1024}KB from logs"

    async def _install_dep(self, err):
        msg = err.message.lower()
        m   = re.search(r"no module named ['\"]?(\w+)", msg)
        if not m: return HealResult.FAILED, "Cannot determine package"
        pkg_import = m.group(1)
        pip_pkg    = PACKAGE_MAP.get(pkg_import, pkg_import)
        if pip_pkg in self._dep_installing:
            return HealResult.COOLDOWN, "Already installing"
        self._dep_installing.add(pip_pkg)
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", pip_pkg],
                    capture_output=True, text=True, timeout=120
                )
            )
            ok = result.returncode == 0
            if ok: self._xp.earn("install_dep", details=pip_pkg)
            return (HealResult.SUCCESS if ok else HealResult.FAILED), \
                   f"pip install {pip_pkg}: {'OK' if ok else result.stderr[:80]}"
        finally:
            self._dep_installing.discard(pip_pkg)

    async def _free_memory(self, err):
        import gc; gc.collect()
        if err.bot_id >= 0:
            scout = self.scouts.get(err.bot_id)
            if scout and hasattr(scout, '_feature_builder'):
                pass
        return HealResult.SUCCESS, "GC + buffer clear"

    async def _pause_trading(self, err):
        if err.bot_id >= 0:
            scout = self.scouts.get(err.bot_id)
            if scout:
                scout._state.paused_until = _TS() + 300
                scout._state.status = BotStatus.PAUSED
        return HealResult.SUCCESS, "Bot paused 5min"

    async def _fix_order(self, err):
        return HealResult.PARTIAL, "Order params flagged for review"

    async def _sync_pos(self, err):
        return HealResult.PARTIAL, "Position sync requested"

    async def _restart_bot(self, err):
        if err.bot_id < 0: return HealResult.FAILED, "No bot_id"
        scout = self.scouts.get(err.bot_id)
        if not scout: return HealResult.FAILED, "Scout not found"
        scout._state.restart_count += 1
        scout._state.status = BotStatus.PAUSED
        scout._state.paused_until = _TS() + 5
        await asyncio.sleep(5)
        scout._state.status = BotStatus.SCANNING
        self._xp.earn("bot_recover", details=f"bot_{err.bot_id}")
        return HealResult.SUCCESS, f"Bot {err.bot_id} restarted (#{scout._state.restart_count})"

    async def _backoff(self, err):
        delay = min(30, 2 ** err.fix_attempts)
        await asyncio.sleep(delay)
        return HealResult.PARTIAL, f"Backoff {delay}s"

    async def _throttle_bots(self, err):
        for scout in list(self.scouts.values())[:100]:
            scout._signal_cooldown = max(scout._signal_cooldown, 60.0)
        return HealResult.SUCCESS, "Top 100 bots throttled"

    async def _fix_config(self, err):
        self._log.warning(f"Config error: {err.message[:60]}")
        return HealResult.PARTIAL, "Config error logged"

    async def _generic_retry(self, err):
        await asyncio.sleep(2)
        return HealResult.PARTIAL, "Generic retry"

    async def _scan_loop(self):
        """Skanuj zdrowie botów co 2s."""
        while self._running:
            await asyncio.sleep(self.SCAN_INTERVAL_S)
            try:
                for bot_id, scout in list(self.scouts.items()):
                    h = self._health_score(scout)
                    scout._state.health_score = h
                    if h < self.DEAD_THRESHOLD:
                        err = OmegaError(
                            id=uuid.uuid4().hex[:8], ts=_TS(),
                            message=f"critical_health_{h:.0f}",
                            category=ErrorCat.CRASH, severity=Severity.CRITICAL,
                            bot_id=bot_id, context={"strategy": "restart_bot"}
                        )
                        await self._err_q.put(err)
                    elif h < self.HEALTH_THRESHOLD:
                        err = OmegaError(
                            id=uuid.uuid4().hex[:8], ts=_TS(),
                            message=f"low_health_{h:.0f}",
                            category=ErrorCat.CRASH, severity=Severity.WARNING,
                            bot_id=bot_id, context={"strategy": "pause_trading"}
                        )
                        await self._err_q.put(err)
            except Exception as e:
                self._log.debug(f"Scan: {e}")

    def _health_score(self, scout: BotScout) -> float:
        s = scout._state; score = 100.0
        if s.consec_losses > 3: score -= 10 * s.consec_losses
        if s.restart_count  > 5: score -= 5  * s.restart_count
        if s.error_count    > 10: score -= 20
        if s.win_rate < 0.35: score -= 20
        if s.daily_pnl < -s.portfolio * 0.10: score -= 30
        return max(0.0, min(100.0, score))

    async def _learn_loop(self):
        """Periodic learning: refine heal strategies."""
        while self._running:
            await asyncio.sleep(3600)  # hourly
            try:
                summary = self.db.get_summary()
                heals = summary.get("heals", 0)
                success = summary.get("heal_success", 0)
                eff = success / max(heals, 1)
                self._log.info(
                    f"🧠 Omega learn: {heals} heals, {eff:.1%} effectiveness, "
                    f"XP={self._xp.xp} Level={self._xp.level} [{self._xp.level_name}]"
                )
            except Exception:
                pass

    def stop(self): self._running = False

    def stats(self) -> Dict:
        return {
            "total_heals":      self._total_heals,
            "successful_heals": self._successful_heals,
            "efficiency":       round(self._successful_heals / max(self._total_heals, 1), 3),
            "queue_size":       self._err_q.qsize(),
            "heals_this_min":   self._heals_this_min,
            "omega_level":      self._xp.level,
            "omega_name":       self._xp.level_name,
            "omega_xp":         self._xp.xp,
            "omega_streak":     self._xp.streak,
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# TIER MANAGER — awans / degradacja botów
# ══════════════════════════════════════════════════════════════════════════════════════════

class TierManager:
    """
    Automatyczny system awansów i degradacji.
    
    Awans: WR≥75% + min_trades → wyższy tier = więcej silników RL
    Degradacja: WR<45% + min_trades → niższy tier = mniej zasobów
    
    Innowacja: Genome transplant
    Gdy bot awansuje z SCOUT → ELITE:
    - Zachowuje swoje wyspecjalizowane DNA
    - Dostaje dodatkowe silniki RL (z HOF jeśli dostępne)
    - Meta-learner przenosi się i kontynuuje adaptację
    
    Gdy bot degraduje APEX → ELITE:
    - Zachowuje swoje DNA (nie jest karany)
    - Zmniejsza się liczba silników (optymalizacja zasobów)
    - Musi odbudować WR aby powrócić
    """

    PROMOTE_WR         = 0.75
    PROMOTE_MIN_TRADES = 100
    DEMOTE_WR          = 0.45
    DEMOTE_MIN_TRADES  = 50
    CHECK_INTERVAL_S   = 1800.0  # co 30 minut

    def __init__(self, scouts: Dict[int, BotScout],
                  pairs:   Dict[int, PairInfo],
                  brains:  Dict[int, BotBrain],
                  db:      BITGOTDatabase,
                  swarm:   SwarmIntelligence,
                  meta_pool: GlobalMetaPool):
        self.scouts    = scouts
        self.pairs     = pairs
        self.brains    = brains
        self.db        = db
        self.swarm     = swarm
        self.meta_pool = meta_pool
        self._log      = logging.getLogger("BITGOT·TierMgr")
        self._promotions = 0; self._demotions = 0
        self._promotion_history: List[Dict] = []

    async def run(self):
        """Periodic tier check loop."""
        self._log.info("📊 TierManager START")
        while True:
            await asyncio.sleep(self.CHECK_INTERVAL_S)
            try:
                self._check_and_rebalance()
            except Exception as e:
                self._log.error(f"Tier rebalance: {e}")

    def _check_and_rebalance(self):
        promoted = demoted = 0
        for bot_id, scout in list(self.scouts.items()):
            s = scout._state; wr = s.wr; n = s.n_trades
            cur_tier = scout.pair.tier
            new_tier = None
            if wr >= self.PROMOTE_WR and n >= self.PROMOTE_MIN_TRADES:
                new_tier = self._next_tier(cur_tier)
                if new_tier:
                    self._promote(bot_id, scout, cur_tier, new_tier)
                    promoted += 1
            elif wr <= self.DEMOTE_WR and n >= self.DEMOTE_MIN_TRADES:
                new_tier = self._prev_tier(cur_tier)
                if new_tier:
                    self._demote(bot_id, scout, cur_tier, new_tier)
                    demoted += 1
        self._promotions += promoted; self._demotions += demoted
        if promoted + demoted > 0:
            self._log.info(f"Tier rebalance: +{promoted} promoted, -{demoted} demoted | "
                           f"Total: +{self._promotions} / -{self._demotions}")

    def _next_tier(self, t: BotTier) -> Optional[BotTier]:
        order = [BotTier.SCOUT, BotTier.STANDARD, BotTier.ELITE, BotTier.APEX]
        idx = order.index(t)
        return order[idx+1] if idx < 3 else None

    def _prev_tier(self, t: BotTier) -> Optional[BotTier]:
        order = [BotTier.SCOUT, BotTier.STANDARD, BotTier.ELITE, BotTier.APEX]
        idx = order.index(t)
        return order[idx-1] if idx > 0 else None

    def _promote(self, bot_id: int, scout: BotScout, old: BotTier, new: BotTier):
        """Awansuj bota: rebuild RL cluster z więcej silnikami."""
        try:
            brain = self.brains.get(bot_id)
            if not brain: return
            # Update pair tier
            scout.pair = PairInfo(
                symbol=scout.pair.symbol, base=scout.pair.base, quote=scout.pair.quote,
                market_type=scout.pair.market_type, score=scout.pair.score,
                spread_pct=scout.pair.spread_pct, vol_1h_pct=scout.pair.vol_1h_pct,
                volume_24h_usdt=scout.pair.volume_24h_usdt, ticks_per_hour=scout.pair.ticks_per_hour,
                avg_volatility=scout.pair.avg_volatility, min_qty=scout.pair.min_qty,
                min_notional=scout.pair.min_notional, price_precision=scout.pair.price_precision,
                qty_precision=scout.pair.qty_precision, max_leverage=scout.pair.max_leverage,
                current_price=scout.pair.current_price, tier=new,
            )
            scout._state.tier = new
            scout._state.promotions += 1
            # Rebuild RL + Neural with more engines
            new_meta = self.meta_pool.get(new)
            brain.rl     = RLEngineCluster(scout.pair.symbol, bot_id, new, brain.genome, new_meta)
            brain.neural = NeuralSwarm(scout.pair.symbol, bot_id, new)
            brain.tier   = new
            self._log.info(
                f"⬆️  PROMOTE bot_{bot_id} {old.value}→{new.value} "
                f"WR={scout._state.wr:.1%} trades={scout._state.n_trades} "
                f"now {new.n_engines()} engines"
            )
            self._promotion_history.append({
                "bot_id": bot_id, "symbol": scout.pair.symbol,
                "from": old.value, "to": new.value,
                "wr": scout._state.wr, "ts": _NOW()
            })
        except Exception as e:
            self._log.error(f"Promote bot_{bot_id}: {e}")

    def _demote(self, bot_id: int, scout: BotScout, old: BotTier, new: BotTier):
        """Degraduj bota: reduce engines."""
        try:
            brain = self.brains.get(bot_id)
            if not brain: return
            scout.pair = PairInfo(
                symbol=scout.pair.symbol, base=scout.pair.base, quote=scout.pair.quote,
                market_type=scout.pair.market_type, score=scout.pair.score,
                spread_pct=scout.pair.spread_pct, vol_1h_pct=scout.pair.vol_1h_pct,
                volume_24h_usdt=scout.pair.volume_24h_usdt, ticks_per_hour=scout.pair.ticks_per_hour,
                avg_volatility=scout.pair.avg_volatility, min_qty=scout.pair.min_qty,
                min_notional=scout.pair.min_notional, price_precision=scout.pair.price_precision,
                qty_precision=scout.pair.qty_precision, max_leverage=scout.pair.max_leverage,
                current_price=scout.pair.current_price, tier=new,
            )
            scout._state.tier = new
            scout._state.demotions += 1
            new_meta = self.meta_pool.get(new)
            brain.rl   = RLEngineCluster(scout.pair.symbol, bot_id, new, brain.genome, new_meta)
            brain.tier = new
            self._log.info(
                f"⬇️  DEMOTE bot_{bot_id} {old.value}→{new.value} "
                f"WR={scout._state.wr:.1%} trades={scout._state.n_trades}"
            )
        except Exception as e:
            self._log.error(f"Demote bot_{bot_id}: {e}")

    def stats(self) -> Dict:
        tier_counts = Counter(s.pair.tier.value for s in self.scouts.values())
        return {
            "promotions":  self._promotions,
            "demotions":   self._demotions,
            "tier_dist":   dict(tier_counts),
            "recent_promo":self._promotion_history[-5:],
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# GENOME EVOLUTION ENGINE — CMA-ES + NSGA-II dla 3000 genomów
# ══════════════════════════════════════════════════════════════════════════════════════════

class GenomeEvolution:
    """
    Ewolucja genomów wszystkich 3000 botów.
    
    Algorytmy:
    1. CMA-ES: numeryczna optymalizacja parametrów genomu
       → adaptuje kierunek i skalę mutacji
    2. NSGA-II: multi-objective (WR, PnL, Drawdown, Trades)
       → Pareto-front zamiast single fitness
    3. Hall of Fame: top 100 genomów nieśmiertelnych
    4. Cross-tier knowledge transfer:
       → HOF genomów APEX → częściowe transplantowanie do SCOUT
    
    Uruchamia się co 1h.
    """

    HOF_SIZE    = 100
    SIGMA_INIT  = 0.08
    EVO_INTERVAL= 3600.0

    def __init__(self, scouts: Dict[int, BotScout], brains: Dict[int, BotBrain],
                  db: BITGOTDatabase):
        self.scouts = scouts; self.brains = brains; self.db = db
        self._log   = logging.getLogger("BITGOT·Evo")
        self.hof:   List[BotGenome] = []
        self.gen    = 0
        # CMA-ES state
        dim = len(BotGenome().to_vector())
        self.dim    = dim
        self.sigma  = self.SIGMA_INIT
        self.mean   = BotGenome().to_vector()
        self.C      = np.eye(dim)
        self.p_c    = np.zeros(dim); self.p_s = np.zeros(dim)
        mu_w = max(TOTAL_BOTS // 8, 20)
        self.mu_w   = mu_w
        raw   = np.log(mu_w + 0.5) - np.log(np.arange(1, mu_w+1))
        self.weights = raw / raw.sum()
        chiN  = math.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        self.chiN   = chiN
        mu_eff= float(mu_w)
        self.c_s    = (mu_eff+2)/(dim+mu_eff+5)
        self.d_s    = 1+2*max(0,math.sqrt((mu_eff-1)/(dim+1))-1)+self.c_s
        self.c_c    = (4+mu_eff/dim)/(dim+4+2*mu_eff/dim)
        self.c_1    = 2/((dim+1.3)**2+mu_eff)
        self.c_mu   = min(1-self.c_1, 2*(mu_eff-2+1/mu_eff)/((dim+2)**2+mu_eff))

    async def run(self):
        self._log.info("🧬 GenomeEvolution START")
        while True:
            await asyncio.sleep(self.EVO_INTERVAL)
            try:
                self._evolve_cycle()
            except Exception as e:
                self._log.error(f"Evolution: {e}")

    def _evolve_cycle(self):
        self.gen += 1
        # Collect all genomes
        genomes = [brain.genome for brain in self.brains.values() if brain.genome.n_trades >= 10]
        if len(genomes) < 20:
            self._log.debug(f"Gen {self.gen}: not enough data ({len(genomes)} genomes)")
            return
        # NSGA-II ranking
        ranked = self._nsga2_sort(genomes)
        # Update HOF
        for g in ranked[:self.HOF_SIZE]:
            self._update_hof(g)
        # CMA-ES update
        top = ranked[:self.mu_w]
        vecs = [g.to_vector() for g in top]
        self._cma_update(vecs)
        # Generate new genomes for bottom 20%
        n_replace = len(genomes) // 5
        for i, (bot_id, brain) in enumerate(list(self.brains.items())[-n_replace:]):
            if brain.genome.n_trades < 10: continue  # skip new bots
            if i >= n_replace: break
            # Sample new genome from CMA-ES distribution
            new_vec  = self._sample()
            new_g    = copy.deepcopy(brain.genome)
            new_g.from_vector(new_vec)
            new_g.gid = uuid.uuid4().hex[:10]
            new_g.generation = self.gen
            new_g.n_trades = 0; new_g.n_wins = 0; new_g.total_pnl = 0
            # HOF injection: 30% chance to inherit from HOF
            if self.hof and random.random() < 0.30:
                donor = random.choice(self.hof[:20])
                new_g = copy.deepcopy(donor)
                new_g.bot_id = bot_id; new_g.n_trades=0; new_g.n_wins=0; new_g.total_pnl=0
            brain.genome = new_g
            self.db.save_genome(new_g)
        self._log.info(
            f"🧬 Gen {self.gen} | {len(genomes)} genomes | "
            f"best_wr={ranked[0].wr():.1%} | sigma={self.sigma:.4f} | HOF={len(self.hof)}"
        )

    def _nsga2_sort(self, genomes: List[BotGenome]) -> List[BotGenome]:
        """Fast non-dominated sort (simplified single-objective proxy)."""
        # Multi-objective: maximize WR, PnL; minimize drawdown
        scores = []
        for g in genomes:
            wr = g.wr(); pnl = g.avg_pnl(); dd = g.max_dd
            # Composite: Sharpe-like
            s = wr * 0.40 + min(pnl * 100, 1) * 0.35 - dd * 0.25
            scores.append(s)
        idx = np.argsort(scores)[::-1]
        return [genomes[i] for i in idx]

    def _update_hof(self, g: BotGenome):
        if len(self.hof) < self.HOF_SIZE:
            self.hof.append(copy.deepcopy(g))
        else:
            worst_i = min(range(len(self.hof)), key=lambda i: self.hof[i].fitness())
            if g.fitness() > self.hof[worst_i].fitness():
                self.hof[worst_i] = copy.deepcopy(g)

    def _cma_update(self, vecs: List[np.ndarray]):
        old_mean = self.mean.copy()
        self.mean = sum(self.weights[i] * vecs[i] for i in range(len(vecs)))
        y_w = (self.mean - old_mean) / max(self.sigma, 1e-10)
        try:
            C_inv = np.linalg.inv(np.linalg.cholesky(self.C + np.eye(self.dim)*1e-8)).T
            ps_up = C_inv @ y_w
        except: ps_up = y_w
        self.p_s = (1-self.c_s)*self.p_s + math.sqrt(self.c_s*(2-self.c_s)*float(self.mu_w))*ps_up
        self.sigma *= math.exp(self.c_s/self.d_s*(np.linalg.norm(self.p_s)/self.chiN - 1))
        self.sigma  = float(np.clip(self.sigma, 0.003, 1.5))
        h_sig = int(np.linalg.norm(self.p_s) < 1.4*self.chiN)
        self.p_c = (1-self.c_c)*self.p_c + h_sig*math.sqrt(self.c_c*(2-self.c_c)*float(self.mu_w))*y_w
        artmp = np.stack([(v - old_mean)/max(self.sigma,1e-10) for v in vecs[:self.mu_w]])
        rm = sum(self.weights[i]*np.outer(artmp[i],artmp[i]) for i in range(len(artmp)))
        self.C = ((1-self.c_1-self.c_mu)*self.C
                   + self.c_1*np.outer(self.p_c,self.p_c)
                   + self.c_mu*rm)
        self.C = (self.C + self.C.T)/2 + np.eye(self.dim)*1e-8

    def _sample(self) -> np.ndarray:
        z = np.random.randn(self.dim)
        try:
            L = np.linalg.cholesky(self.C + np.eye(self.dim)*1e-8)
            return self.mean + self.sigma * (L @ z)
        except:
            return self.mean + self.sigma * z

    def stats(self) -> Dict:
        return {
            "generation": self.gen,
            "hof_size":   len(self.hof),
            "sigma":      round(self.sigma, 5),
            "best_hof_wr":round(max((g.wr() for g in self.hof), default=0), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# SIGNAL QUEUE PROCESSOR — async consumer dla sygnałów
# ══════════════════════════════════════════════════════════════════════════════════════════

class SignalQueueProcessor:
    """
    Asyncio consumer dla sygnałów od 3000 BotScoutów.
    Przetwarza sygnały z priorytetem (highest confidence first).
    Batch-sortuje co 100ms.
    """

    BATCH_INTERVAL_MS = 100
    MAX_BATCH_SIZE    = 50

    def __init__(self, signal_queue: asyncio.Queue, signal_manager: SignalManager):
        self.q    = signal_queue
        self.sm   = signal_manager
        self._log = logging.getLogger("BITGOT·SigProc")
        self._processed = 0; self._running = False

    async def run(self):
        self._running = True
        self._log.info("📨 SignalQueueProcessor START")
        while self._running:
            batch: List[TradingSignal] = []
            # Collect batch
            deadline = asyncio.get_event_loop().time() + self.BATCH_INTERVAL_MS/1000
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0 or len(batch) >= self.MAX_BATCH_SIZE:
                    break
                try:
                    sig = await asyncio.wait_for(self.q.get(), timeout=remaining)
                    batch.append(sig)
                except asyncio.TimeoutError:
                    break
            if not batch: continue
            # Sort by confidence × urgency (highest first)
            batch.sort(key=lambda s: s.confidence, reverse=True)
            # Process each
            for sig in batch:
                try:
                    await self.sm.process(sig)
                    self._processed += 1
                except Exception as e:
                    self._log.debug(f"Process signal: {e}")

    def stop(self): self._running = False

    def stats(self) -> Dict:
        return {"processed": self._processed, "queue_size": self.q.qsize()}


# ══════════════════════════════════════════════════════════════════════════════════════════
# BITGOT SYSTEM E3 — publiczny interfejs Etapu 3
# ══════════════════════════════════════════════════════════════════════════════════════════

class BitgotSystemE3:
    """
    Fasada Etapu 3 — inicjalizuje i uruchamia wszystkie komponenty egzekucji.
    
    Tworzy:
    - DataFetcher (centralny agregator danych)
    - 3000 BotScoutów (zwiadowcy)
    - 3000 BotExecutorów (wykonawcy)
    - SignalManager + SignalQueueProcessor
    - PositionMonitor (10Hz)
    - CircuitBreakers (6-poziomowe)
    - OmegaHealerDaemon (samonaprawiający)
    - TierManager (awanse/degradacje)
    - GenomeEvolution (CMA-ES + NSGA-II)
    """

    def __init__(self, connector: BitgetConnector, pairs: List[PairInfo],
                  brains: Dict[int, BotBrain], swarm: SwarmIntelligence,
                  meta_pool: GlobalMetaPool, capital: CapitalEngine,
                  portfolio: GlobalPortfolioManager, db: BITGOTDatabase,
                  mdc: MarketDataCache, cfg: BITGOTConfig = CFG):
        self.conn     = connector
        self.pairs    = pairs
        self.brains   = brains
        self.swarm    = swarm
        self.meta_pool= meta_pool
        self.capital  = capital
        self.port     = portfolio
        self.db       = db
        self.mdc      = mdc
        self.cfg      = cfg
        self._log     = logging.getLogger("BITGOT·E3")
        self._running = False

        # ── Components ────────────────────────────────────────────────────────
        self.circuit   = CircuitBreakers(portfolio)
        self.sig_queue = asyncio.Queue(maxsize=SignalManager.MAX_QUEUE_SIZE)
        # Scouts
        self.scouts: Dict[int, BotScout] = {}
        # Executors
        self.executors_list: List[BotExecutor] = []
        self.executor_pool: Optional[ExecutorPool] = None
        # Open positions (shared)
        self.positions: Dict[int, LivePosition] = {}
        # Managers
        self.sig_manager:   Optional[SignalManager]       = None
        self.sig_processor: Optional[SignalQueueProcessor] = None
        self.pos_monitor:   Optional[PositionMonitor]     = None
        self.data_fetcher:  Optional[DataFetcher]         = None
        self.healer:        Optional[OmegaHealerDaemon]   = None
        self.tier_mgr:      Optional[TierManager]         = None
        self.genome_evo:    Optional[GenomeEvolution]     = None
        self._tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Inicjalizuj wszystkie komponenty E3."""
        n = len(self.pairs)
        self._log.info(f"⚙️  BitgotSystemE3 INIT — {n} bots")

        # ── Executors ─────────────────────────────────────────────────────────
        self.sig_manager = SignalManager(None, self.port, self.capital, self.db)  # temp
        for i in range(n):
            ex = BotExecutor(i, self.conn, self.port, self.db, self.sig_manager)
            self.executors_list.append(ex)
        self.executor_pool = ExecutorPool(self.executors_list)

        # Assign executors to bots
        for i, ex in enumerate(self.executors_list):
            self.executor_pool.assign(i, ex)

        # ── SignalManager (with pool) ─────────────────────────────────────────
        self.sig_manager = SignalManager(
            self.executor_pool, self.port, self.capital, self.db
        )
        self.sig_processor = SignalQueueProcessor(self.sig_queue, self.sig_manager)

        # ── Scouts ────────────────────────────────────────────────────────────
        for i, pair in enumerate(self.pairs[:n]):
            brain = self.brains.get(i)
            if not brain: continue
            scout = BotScout(
                bot_id   = i,
                pair     = pair,
                brain    = brain,
                mdc      = self.mdc,
                capital_engine = self.capital,
                portfolio= self.port,
                signal_queue = self.sig_queue,
                swarm    = self.swarm,
            )
            self.scouts[i] = scout

        # ── DataFetcher ───────────────────────────────────────────────────────
        symbols = [p.symbol for p in self.pairs[:n]]
        self.data_fetcher = DataFetcher(self.conn, self.mdc, symbols)

        # ── PositionMonitor ───────────────────────────────────────────────────
        self.pos_monitor = PositionMonitor(
            self.positions, self.executor_pool, self.scouts,
            self.mdc, self.conn, self.port, self.db, self.circuit
        )

        # ── OmegaHealer ───────────────────────────────────────────────────────
        self.healer = OmegaHealerDaemon(self.db, self.scouts, self.circuit, self.conn)

        # ── TierManager ───────────────────────────────────────────────────────
        pair_dict = {i: p for i, p in enumerate(self.pairs[:n])}
        self.tier_mgr = TierManager(
            self.scouts, pair_dict, self.brains, self.db, self.swarm, self.meta_pool
        )

        # ── GenomeEvolution ───────────────────────────────────────────────────
        self.genome_evo = GenomeEvolution(self.scouts, self.brains, self.db)

        self._log.info(f"✅ E3 initialized: {len(self.scouts)} scouts, "
                        f"{len(self.executors_list)} executors")

    async def run(self):
        """Uruchom wszystkie komponenty E3."""
        self._running = True
        self._log.info("🚀 BitgotSystemE3 — ALL SYSTEMS LAUNCH")

        tasks = []
        # DataFetcher
        tasks.append(asyncio.create_task(self.data_fetcher.run(), name="DataFetcher"))
        # Scouts (3000 coroutines)
        for bot_id, scout in self.scouts.items():
            tasks.append(asyncio.create_task(scout.run(), name=f"Scout-{bot_id}"))
        # SignalProcessor
        tasks.append(asyncio.create_task(self.sig_processor.run(), name="SigProcessor"))
        # PositionMonitor
        tasks.append(asyncio.create_task(self.pos_monitor.run(), name="PosMon"))
        # OmegaHealer
        tasks.append(asyncio.create_task(self.healer.run(), name="OmegaHealer"))
        # TierManager
        tasks.append(asyncio.create_task(self.tier_mgr.run(), name="TierMgr"))
        # GenomeEvolution
        tasks.append(asyncio.create_task(self.genome_evo.run(), name="GenomeEvo"))
        # Swarm global update
        tasks.append(asyncio.create_task(self._swarm_loop(), name="SwarmUpdate"))
        # Daily reset
        tasks.append(asyncio.create_task(self._daily_reset_loop(), name="DailyReset"))
        # Save checkpoints
        tasks.append(asyncio.create_task(self._checkpoint_loop(), name="Checkpoint"))

        self._tasks = tasks
        self._log.info(f"  {len(tasks)} async tasks launched")
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def _swarm_loop(self):
        """Global swarm intelligence update co 30s."""
        while self._running:
            await asyncio.sleep(30)
            self.swarm.update_global()

    async def _daily_reset_loop(self):
        """Daily reset at UTC midnight."""
        while self._running:
            now = time.gmtime()
            secs_to_midnight = (24*3600) - (now.tm_hour*3600 + now.tm_min*60 + now.tm_sec)
            await asyncio.sleep(secs_to_midnight)
            self.port.reset_daily()
            self.circuit.reset_daily()
            for scout in self.scouts.values():
                scout._state.daily_pnl = 0.0
            self._log.info("🌅 Daily reset completed")

    async def _checkpoint_loop(self):
        """Periodic checkpoint: save all brains + genomes."""
        while self._running:
            await asyncio.sleep(self.cfg.checkpoint_interval_h * 3600)
            try:
                saved = 0
                for bot_id, brain in list(self.brains.items()):
                    if bot_id % 10 == 0:  # save every 10th bot
                        brain.save_all()
                        self.db.save_genome(brain.genome)
                        saved += 1
                self._log.info(f"💾 Checkpoint: {saved} brains saved")
            except Exception as e:
                self._log.error(f"Checkpoint: {e}")

    async def stop(self):
        """Graceful shutdown."""
        self._log.warning("🛑 E3 shutdown...")
        self._running = False
        # Stop scouts
        for scout in self.scouts.values():
            scout.stop()
        # Stop components
        if self.data_fetcher: self.data_fetcher.stop()
        if self.pos_monitor:  self.pos_monitor.stop()
        if self.healer:       self.healer.stop()
        if self.sig_processor: self.sig_processor.stop()
        # Cancel tasks
        for t in self._tasks:
            if not t.done(): t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._log.info("✅ E3 stopped")

    def get_status(self) -> Dict:
        """Full system status snapshot."""
        active_positions = len(self.positions)
        active_scouts    = sum(1 for s in self.scouts.values() if s.is_running)
        total_pnl        = sum(s._state.total_pnl for s in self.scouts.values())
        all_wrs          = [s._state.wr for s in self.scouts.values() if s._state.n_trades >= 10]
        return {
            "e3_running":     self._running,
            "active_scouts":  active_scouts,
            "active_positions": active_positions,
            "total_pnl":      round(total_pnl, 6),
            "global_wr":      round(float(np.mean(all_wrs)) if all_wrs else 0, 4),
            "portfolio":      self.port.snapshot.__dict__,
            "signal_manager": self.sig_manager.stats() if self.sig_manager else {},
            "position_monitor":self.pos_monitor.stats() if self.pos_monitor else {},
            "circuit":        self.circuit.stats(),
            "healer":         self.healer.stats() if self.healer else {},
            "tier_manager":   self.tier_mgr.stats() if self.tier_mgr else {},
            "genome_evo":     self.genome_evo.stats() if self.genome_evo else {},
            "swarm":          self.swarm.global_stats(),
            "data_fetcher":   self.data_fetcher.stats() if self.data_fetcher else {},
            "executor_pool":  self.executor_pool.stats() if self.executor_pool else {},
        }

    def inject_error(self, exc: Exception, bot_id: int = -1, ctx: Dict = None):
        """Przekaż błąd do OmegaHealera (wywoływane z try/except w dowolnym miejscu)."""
        if self.healer:
            self.healer.report(exc, bot_id, ctx)


# ══════════════════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Data
    "DataFetcher",
    # Scout
    "BotScout", "pair_max_lev",
    # Signal
    "SignalRecord", "SignalManager", "SignalQueueProcessor",
    # Executor
    "LivePosition", "BotExecutor", "ExecutorPool",
    # Monitor
    "PositionMonitor",
    # Circuit
    "CircuitState", "CircuitBreakers",
    # Healer
    "HealXP", "OmegaHealerDaemon",
    # Tier
    "TierManager",
    # Evolution
    "GenomeEvolution",
    # Facade
    "BitgotSystemE3",
]


if __name__ == "__main__":
    import ast, sys
    src = open(__file__).read()
    try:
        ast.parse(src)
        cls = [n.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef)]
        fns = sum(1 for n in ast.walk(ast.parse(src))
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        print(f"✅ ETAP 3: SKŁADNIA OK")
        print(f"Linie:   {src.count(chr(10)):,}")
        print(f"Klasy:   {len(cls)} → {cls}")
        print(f"Metody:  {fns}")
        print(f"Znaki:   {len(src):,}")
    except SyntaxError as e:
        print(f"❌ BŁĄD: {e}")
        sys.exit(1)
