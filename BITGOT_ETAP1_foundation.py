"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║  ██████╗ ██╗████████╗ ██████╗  ██████╗ ████████╗                                          ║
║  ██╔══██╗██║╚══██╔══╝██╔════╝ ██╔═══██╗╚══██╔══╝                                          ║
║  ██████╔╝██║   ██║   ██║  ███╗██║   ██║   ██║                                             ║
║  ██╔══██╗██║   ██║   ██║   ██║██║   ██║   ██║                                             ║
║  ██████╔╝██║   ██║   ╚██████╔╝╚██████╔╝   ██║                                             ║
║  ╚═════╝ ╚═╝   ╚═╝    ╚═════╝  ╚═════╝    ╚═╝                                             ║
║                                                                                              ║
║  3 0 0 0   B O T   B I T G E T   S U P R E M A C Y   S Y S T E M   v ∞                  ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║  ETAP 1 / 4 — FUNDAMENT                                                                     ║
║                                                                                              ║
║  ▸ BITGOTConfig       — absolutna konfiguracja systemu (jeden plik, zero wątpliwości)      ║
║  ▸ BitgetConnector    — dedykowany konektor Bitget (Futures USDT-M + Spot)                 ║
║  ▸ PairDiscovery      — odkrywa WSZYSTKIE pary Bitget, rankuje, przydziela unikalnie       ║
║  ▸ CapitalEngine      — dynamiczne skalowanie kapitału per bot (rośnie z portfelem)        ║
║  ▸ CapitalTierSystem  — 8 tierów kapitału od $50 do $∞ ze skalowanymi parametrami         ║
║  ▸ BITGOTDatabase     — SQLite WAL, batch writes, 3000 concurrent writers                  ║
║  ▸ MathCore           — NumPy-only: EMA, RSI, MACD, ATR, ADX, Hurst, KAMA, OFI, VPIN    ║
║  ▸ StateVector        — 80-wymiarowy zunifikowany wektor stanu Bitget-specific             ║
║  ▸ Genome             — DNA bota z dynamicznym skalowaniem i CMA-ES                       ║
║  ▸ OmegaTypes         — wszystkie typy, enumeracje, dataclasses                           ║
║                                                                                              ║
║  ARCHITEKTURA SYSTEMU:                                                                       ║
║  ╔═══════════════╗  ╔══════════════╗  ╔═══════════════╗  ╔══════════════════╗            ║
║  ║ 3000 SCOUTS   ║→ ║ SIGNAL MGR   ║→ ║ 3000 EXECUTORS║→ ║  BITGET EXCHANGE ║            ║
║  ║ (analiza)     ║  ║ (80% filter) ║  ║ (egzekucja)   ║  ║  (USDT futures)  ║            ║
║  ╚═══════════════╝  ╚══════════════╝  ╚═══════════════╝  ╚══════════════════╝            ║
║                           │                                                                  ║
║           ╔═══════════════╩════════════════════╗                                            ║
║           ║  META-LEARNING RL (learn to learn)  ║                                           ║
║           ║  25 engines × 3000 bots             ║                                           ║
║           ╚════════════════════════════════════╝                                            ║
║                                                                                              ║
║  TARGET: 200,000 trades/day · 85%+ WR · x25-x125 leverage · Self-improving               ║
║  EXCHANGE: Bitget ONLY (Futures USDT-M + Spot)                                             ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────────────────
import asyncio
import collections
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import random
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque, Counter
from contextlib import suppress
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Generator, Iterator,
                    List, Optional, Set, Sequence, Tuple, Union)

# ── 3rd party (numpy always available) ───────────────────────────────────────────────────
import numpy as np

# ── optional ─────────────────────────────────────────────────────────────────────────────
def _try_import(name: str):
    try: return __import__(name)
    except ImportError: return None

ccxt_mod      = _try_import("ccxt")
ccxt_async    = _try_import("ccxt.async_support")
aiohttp_mod   = _try_import("aiohttp")
fastapi_mod   = _try_import("fastapi")
uvicorn_mod   = _try_import("uvicorn")
psutil_mod    = _try_import("psutil")
websockets_m  = _try_import("websockets")

UTC   = timezone.utc
_NOW  = lambda: datetime.now(UTC).isoformat()
_TS   = lambda: time.time()
_MS   = lambda: int(time.monotonic() * 1_000)
_US   = lambda: int(time.monotonic() * 1_000_000)

# ══════════════════════════════════════════════════════════════════════════════════════════
# DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════════════════

_DIRS = [
    "logs", "bitgot_data", "bitgot_data/db", "bitgot_data/models",
    "bitgot_data/genomes", "bitgot_data/trades", "bitgot_data/signals",
    "bitgot_data/snapshots", "bitgot_data/metrics", "bitgot_data/omega",
    "bitgot_data/checkpoints",
]
for _d in _DIRS:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d │ %(name)-28s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/bitgot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        ),
    ],
)
_log = logging.getLogger("BITGOT·CORE")


# ══════════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS — ABSOLUTNE PARAMETRY SYSTEMU (nie zmieniaj bez przemyślenia)
# ══════════════════════════════════════════════════════════════════════════════════════════

TOTAL_BOTS              = 3_000     # każdy bot = unikalna para
TARGET_TRADES_PER_DAY   = 200_000   # 200k trades/dzień = ~66.7/bot
TRADES_PER_BOT_PER_DAY  = TARGET_TRADES_PER_DAY / TOTAL_BOTS   # ≈66.7
TRADE_INTERVAL_S        = 86_400 / TRADES_PER_BOT_PER_DAY      # ≈21.6 minut

MIN_CONFIDENCE          = 0.80      # 80% — żelazny próg sygnału
TARGET_WIN_RATE         = 0.85      # 85% docelowe WR
STATE_DIM               = 80        # 80-wymiarowy wektor stanu
N_ACTIONS               = 5         # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
N_RL_ENGINES            = 25        # 25 silników RL per APEX bot

GAMMA                   = 0.995
LR_FAST                 = 0.002
LR_SLOW                 = 0.0004
LR_META                 = 0.0001    # meta-learning rate
BATCH_SIZE              = 64
BUFFER_CAP              = 100_000
EPS_START               = 0.15
EPS_MIN                 = 0.003
EPS_DECAY               = 0.9999

FEE_MAKER               = 0.00020   # 0.020% Bitget maker
FEE_TAKER               = 0.00060   # 0.060% Bitget taker
FEE_FUNDING_AVG         = 0.00010   # avg funding fee proxy

DATA_DIR   = Path("bitgot_data")
MODELS_DIR = DATA_DIR / "models"
DB_PATH    = DATA_DIR / "db" / "bitgot.db"
GENOME_DIR = DATA_DIR / "genomes"
SIGNAL_DIR = DATA_DIR / "signals"
CKPT_DIR   = DATA_DIR / "checkpoints"


# ══════════════════════════════════════════════════════════════════════════════════════════
# BITGOT CONFIG — jedyne miejsce konfiguracji
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class BITGOTConfig:
    """
    Centralna konfiguracja BITGOT.
    Każdy parametr jest precyzyjnie dobrany dla strategii 85% WR.
    """

    # ── Tryb ─────────────────────────────────────────────────────────────────────
    paper_mode:     bool  = True       # False = live trading z prawdziwym kapitałem
    debug_mode:     bool  = False

    # ── Bitget API ────────────────────────────────────────────────────────────────
    api_key:        str   = field(default_factory=lambda: os.getenv("BITGET_KEY",    ""))
    api_secret:     str   = field(default_factory=lambda: os.getenv("BITGET_SECRET", ""))
    api_passphrase: str   = field(default_factory=lambda: os.getenv("BITGET_PASS",   ""))
    testnet:        bool  = True       # Bitget sandbox

    # ── Rój ───────────────────────────────────────────────────────────────────────
    n_bots:         int   = TOTAL_BOTS
    n_executors:    int   = TOTAL_BOTS   # 1:1 scout:executor

    # ── Rynki ─────────────────────────────────────────────────────────────────────
    markets: List[str]    = field(default_factory=lambda: [
        "futures_usdt",      # USDT-M perpetual futures (primary)
        "futures_coin",      # Coin-M futures
        "spot",              # Spot market
    ])
    # Minimalne wymagania dla pary
    min_volume_usdt_24h:  float = 1_000_000.0   # min $1M dzienny wolumen
    min_price:            float = 0.0001
    max_spread_pct:       float = 0.08           # max 0.08% spread
    min_vol_1h_pct:       float = 0.08           # min 0.08% zmienności/h

    # ── Kapitał startowy ─────────────────────────────────────────────────────────
    start_capital:        float = 3_000.0        # $3000 start

    # ── Dynamiczne skalowanie kapitału per bot ────────────────────────────────────
    # Każdy bot przeznacza base_position_pct% AKTUALNEGO portfela per transakcję
    # Przykład: portfel $3000, base=0.033% → $1/bot
    #           portfel $6000, base=0.033% → $2/bot (automatyczny wzrost)
    base_position_pct:    float = 0.0333         # 0.0333% kapitału per transakcja
    min_position_usd:     float = 1.0            # absolutne minimum $1
    max_position_pct:     float = 0.10           # max 10% kapitału na jedną pozycję
    kelly_fraction:       float = 0.25           # ułamek Kelly (konserwatywny)

    # ── Dźwignia ─────────────────────────────────────────────────────────────────
    leverage_default:     int   = 25             # default dla niskich tierów
    leverage_max:         int   = 125            # max przy wysokim WR
    leverage_min:         int   = 10             # minimum bezpieczeństwa

    # ── Sygnał / Pewność ─────────────────────────────────────────────────────────
    confidence_threshold: float = 0.80           # 80% — żelazny próg
    confidence_strong:    float = 0.90           # silny sygnał (wyższa pozycja)
    confidence_absolute:  float = 0.95           # absolutny (max pozycja)
    min_engines_agree:    int   = 15             # min silników RL w zgodzie (z 25)

    # ── Stop Loss / Take Profit ───────────────────────────────────────────────────
    sl_pct:               float = 0.0060         # 0.60% SL
    tp_pct:               float = 0.0120         # 1.20% TP (2:1 RR)
    trail_arm_pct:        float = 0.40           # trailing po 40% TP
    trail_dist_pct:       float = 0.0030         # trailing odległość 0.30%
    max_hold_seconds:     int   = 900            # max 15 minut hold

    # ── Timing ────────────────────────────────────────────────────────────────────
    tick_interval_ms:     int   = 500            # 2× na sekundę per bot (zmniejszona intensywność)
    signal_cooldown_s:    float = 60.0           # min czas między sygnałami
    cool_win_s:           float = 30.0           # cooldown po wygranej
    cool_loss_s:          float = 120.0          # cooldown po przegranej (selekcja)
    ohlcv_refresh_s:      float = 120.0          # odświeżanie OHLCV co 2min

    # ── Risk Management ───────────────────────────────────────────────────────────
    max_daily_loss_pct:   float = 0.05           # -5% portfela → dzienny halt
    max_drawdown_pct:     float = 0.12           # -12% peak → circuit breaker
    max_positions:        int   = 500            # max jednoczesnych pozycji globalne
    max_pos_per_tier:     Dict[str,int] = field(default_factory=lambda: {
        "apex":    50,
        "elite":   150,
        "standard":200,
        "scout":   100,
    })
    max_correlation:      float = 0.75           # max korelacja między otwartymi pozycjami

    # ── Ewolucja ─────────────────────────────────────────────────────────────────
    evolution_interval_h: float = 1.0            # CMA-ES co 1h
    hof_size:             int   = 100            # Hall of Fame top 100 genomów
    novelty_archive:      int   = 500
    evo_elite_pct:        float = 0.15

    # ── Meta-Learning ─────────────────────────────────────────────────────────────
    meta_update_interval: int   = 50             # update meta-learner co 50 trade'ów
    maml_inner_steps:     int   = 5              # MAML inner adaptation steps
    maml_inner_lr:        float = 0.01           # MAML inner learning rate

    # ── Tier promotion ────────────────────────────────────────────────────────────
    promote_wr_threshold: float = 0.75           # WR do awansu
    promote_min_trades:   int   = 100
    demote_wr_threshold:  float = 0.45           # WR do degradacji
    demote_min_trades:    int   = 50

    # ── Self-Healing ─────────────────────────────────────────────────────────────
    max_bot_errors:       int   = 10             # błędy zanim bot zawieszony
    heal_scan_interval_s: float = 2.0
    circuit_failure_thr:  int   = 7              # błędy do otwarcia circuit

    # ── Persistence ───────────────────────────────────────────────────────────────
    checkpoint_interval_h:float = 4.0
    metrics_interval_s:   float = 30.0
    db_flush_interval_s:  float = 0.5

    # ── REST API ──────────────────────────────────────────────────────────────────
    api_port:             int   = 8888
    ws_port:              int   = 8889

    def leverage_for_wr(self, win_rate: float) -> int:
        """Dynamiczna dźwignia bazująca na aktualnym WR bota."""
        if win_rate >= 0.90: return self.leverage_max
        if win_rate >= 0.85: return 75
        if win_rate >= 0.80: return 50
        if win_rate >= 0.75: return 35
        if win_rate >= 0.70: return 25
        return self.leverage_default

    def position_size_usd(self, portfolio: float) -> float:
        """Dynamiczne skalowanie pozycji z portfelem."""
        base = portfolio * self.base_position_pct / 100.0
        return float(np.clip(base, self.min_position_usd,
                              portfolio * self.max_position_pct / 100.0))

    def validate(self) -> List[str]:
        warnings = []
        if not self.api_key:    warnings.append("⚠️  BITGET_KEY nie ustawiony")
        if not self.api_secret: warnings.append("⚠️  BITGET_SECRET nie ustawiony")
        if not self.api_passphrase: warnings.append("⚠️  BITGET_PASS nie ustawiony")
        if self.paper_mode:     warnings.append("ℹ️  PAPER MODE aktywny")
        if self.testnet:        warnings.append("ℹ️  TESTNET aktywny")
        return warnings

# Globalny singleton konfiguracji
CFG = BITGOTConfig()


# ══════════════════════════════════════════════════════════════════════════════════════════
# ENUMERACJE
# ══════════════════════════════════════════════════════════════════════════════════════════

class Action(IntEnum):
    STRONG_BUY  = 0
    BUY         = 1
    HOLD        = 2
    SELL        = 3
    STRONG_SELL = 4

    def to_side(self) -> str:
        if self in (Action.STRONG_BUY, Action.BUY):  return "long"
        if self in (Action.STRONG_SELL, Action.SELL): return "short"
        return "flat"

    def is_bullish(self) -> bool: return self in (Action.STRONG_BUY, Action.BUY)
    def is_bearish(self) -> bool: return self in (Action.STRONG_SELL, Action.SELL)
    def is_strong(self) -> bool:  return self in (Action.STRONG_BUY, Action.STRONG_SELL)

class BotTier(Enum):
    APEX     = "apex"      # WR≥85%, 25 engines, max leverage
    ELITE    = "elite"     # WR≥75%, 15 engines
    STANDARD = "standard"  # WR≥60%, 8 engines
    SCOUT    = "scout"     # WR≥50%, 3 engines (new bots start here)

    def n_engines(self) -> int:
        return {self.APEX:25, self.ELITE:15, self.STANDARD:8, self.SCOUT:3}[self]

    def n_neural_archs(self) -> int:
        return {self.APEX:8, self.ELITE:5, self.STANDARD:3, self.SCOUT:1}[self]

    def consensus_threshold(self) -> float:
        return {self.APEX:0.52, self.ELITE:0.55, self.STANDARD:0.58, self.SCOUT:0.62}[self]

    def max_leverage(self) -> int:
        return {self.APEX:125, self.ELITE:75, self.STANDARD:50, self.SCOUT:25}[self]

    @classmethod
    def from_wr(cls, wr: float) -> 'BotTier':
        if wr >= 0.85: return cls.APEX
        if wr >= 0.75: return cls.ELITE
        if wr >= 0.60: return cls.STANDARD
        return cls.SCOUT

class BotStatus(Enum):
    INITIALIZING = "init"
    SCANNING     = "scanning"
    SIGNAL_WAIT  = "signal_wait"
    LONG         = "long"
    SHORT        = "short"
    CLOSING      = "closing"
    COOLING      = "cooling"
    PAUSED       = "paused"
    HEALING      = "healing"
    PROMOTED     = "promoted"
    DEMOTED      = "demoted"
    DEAD         = "dead"

class MarketType(Enum):
    FUTURES_USDT = "futures_usdt"
    FUTURES_COIN = "futures_coin"
    SPOT         = "spot"

class SignalState(Enum):
    PENDING   = "pending"
    CONFIRMED = "confirmed"
    REJECTED  = "rejected"
    EXECUTED  = "executed"
    EXPIRED   = "expired"

class Regime(Enum):
    STEALTH_ACCUMULATION = "stealth_acc"
    MARKUP               = "markup"
    DISTRIBUTION         = "distribution"
    MARKDOWN             = "markdown"
    RANGING_TIGHT        = "ranging_tight"
    RANGING_WIDE         = "ranging_wide"
    PARABOLIC            = "parabolic"
    CAPITULATION         = "capitulation"
    LIQUIDITY_HUNT       = "liquidity_hunt"
    FLASH_CRASH          = "flash_crash"
    FLASH_PUMP           = "flash_pump"
    MANIPULATED          = "manipulated"

class HealResult(Enum):
    SUCCESS  = "success"
    PARTIAL  = "partial"
    FAILED   = "failed"
    SKIPPED  = "skipped"
    COOLDOWN = "cooldown"

class Severity(Enum):
    DEBUG    = 0; INFO  = 1; WARNING = 2
    ERROR    = 3; CRITICAL = 4; FATAL = 5

class ErrorCat(Enum):
    NETWORK    = "network";    API      = "api"
    DATABASE   = "database";   TRADING  = "trading"
    SYSTEM     = "system";     WEBSOCKET= "websocket"
    AUTH       = "auth";       RATE_LIMIT="rate_limit"
    DEPENDENCY = "dependency"; CRASH    = "crash"
    MEMORY     = "memory";     CONFIG   = "config"
    SIGNAL     = "signal";     UNKNOWN  = "unknown"


# ══════════════════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class PairInfo:
    """Kompletne informacje o parze handlowej na Bitget."""
    symbol:          str          # np. "BTC/USDT:USDT"
    base:            str          # "BTC"
    quote:           str          # "USDT"
    market_type:     MarketType
    exchange_id:     str          = "bitget"
    # Skalowanie
    score:           float        = 0.5
    spread_pct:      float        = 0.0
    vol_1h_pct:      float        = 0.0
    volume_24h_usdt: float        = 0.0
    ticks_per_hour:  float        = 60.0
    avg_volatility:  float        = 0.0
    # Limity
    min_qty:         float        = 1.0
    min_notional:    float        = 5.0
    contract_size:   float        = 1.0
    price_precision: int          = 6
    qty_precision:   int          = 4
    max_leverage:    int          = 125
    current_price:   float        = 0.0
    # Tier
    tier:            BotTier      = BotTier.SCOUT

    def __hash__(self): return hash(f"{self.exchange_id}:{self.symbol}")
    def __eq__(self, o): return hash(self) == hash(o)

    def key(self) -> str: return f"{self.exchange_id}:{self.symbol}"


@dataclass
class BotTrade:
    """Zapis jednej transakcji bota."""
    id:           str   = field(default_factory=lambda: uuid.uuid4().hex[:14])
    bot_id:       int   = 0
    symbol:       str   = ""
    market_type:  str   = "futures_usdt"
    side:         str   = ""           # "long" / "short"
    entry_price:  float = 0.0
    exit_price:   float = 0.0
    qty:          float = 0.0
    notional:     float = 0.0
    leverage:     int   = 1
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    roi_pct:      float = 0.0          # pnl / margin
    fees:         float = 0.0
    funding_fee:  float = 0.0
    duration_ms:  int   = 0
    exit_reason:  str   = ""
    signal_conf:  float = 0.0
    regime:       str   = ""
    tier:         str   = ""
    portfolio_at: float = 0.0          # portfolio value gdy zamknięto
    entry_ts:     int   = field(default_factory=_MS)
    exit_ts:      int   = 0

    @property
    def won(self) -> bool: return self.pnl > 0

    @property
    def margin(self) -> float:
        return self.notional / max(self.leverage, 1)


@dataclass
class BotState:
    """Pełen stan bota — jeden snapshot."""
    bot_id:         int
    symbol:         str
    market_type:    MarketType
    tier:           BotTier       = BotTier.SCOUT
    status:         BotStatus     = BotStatus.INITIALIZING
    portfolio:      float         = 0.0     # alokowana kwota dla tego bota
    n_trades:       int           = 0
    n_wins:         int           = 0
    total_pnl:      float         = 0.0
    daily_pnl:      float         = 0.0
    win_rate:       float         = 0.5
    last_price:     float         = 0.0
    health_score:   float         = 100.0
    genome_id:      str           = ""
    current_regime: str           = "ranging_tight"
    consec_losses:  int           = 0
    consec_wins:    int           = 0
    paused_until:   float         = 0.0
    error_count:    int           = 0
    restart_count:  int           = 0
    promotions:     int           = 0
    demotions:      int           = 0
    last_trade_ts:  int           = 0
    last_signal_ts: float         = 0.0
    sharpe_ratio:   float         = 0.0
    max_drawdown:   float         = 0.0
    peak_pnl:       float         = 0.0

    @property
    def wr(self) -> float: return self.n_wins / max(self.n_trades, 1)

    @property
    def can_trade(self) -> bool:
        return (self.status not in (BotStatus.DEAD, BotStatus.PAUSED, BotStatus.HEALING)
                and _TS() >= self.paused_until
                and self.error_count < CFG.max_bot_errors)


@dataclass
class GlobalPortfolio:
    """Globalny portfel systemu — śledzony w czasie rzeczywistym."""
    total_capital:   float = 0.0
    peak_capital:    float = 0.0
    day_start_cap:   float = 0.0
    available:       float = 0.0
    in_positions:    float = 0.0
    total_pnl:       float = 0.0
    daily_pnl:       float = 0.0
    total_trades:    int   = 0
    total_wins:      int   = 0
    active_positions:int   = 0
    halted:          bool  = False
    halt_reason:     str   = ""
    day_start_ts:    float = field(default_factory=_TS)

    @property
    def global_wr(self) -> float: return self.total_wins / max(self.total_trades, 1)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital <= 0: return 0.0
        return (self.peak_capital - self.total_capital) / self.peak_capital

    @property
    def daily_loss_pct(self) -> float:
        if self.day_start_cap <= 0: return 0.0
        return (self.day_start_cap - self.total_capital) / self.day_start_cap

    def update_peak(self):
        if self.total_capital > self.peak_capital:
            self.peak_capital = self.total_capital


@dataclass
class TradingSignal:
    """Sygnał handlowy przesyłany przez SignalManager do wykonawców."""
    id:             str     = field(default_factory=lambda: uuid.uuid4().hex[:12])
    bot_id:         int     = 0
    symbol:         str     = ""
    side:           str     = ""        # "long" / "short"
    confidence:     float   = 0.0
    raw_score:      float   = 0.0      # surowy sygnał [-1,+1]
    regime:         str     = ""
    state:          SignalState = SignalState.PENDING
    # Parametry pozycji (obliczone przez SignalManager)
    entry_price:    float   = 0.0
    sl_price:       float   = 0.0
    tp_price:       float   = 0.0
    leverage:       int     = 25
    notional:       float   = 0.0
    margin:         float   = 0.0
    # Metadane
    engines_voted:  int     = 0        # ile silników RL głosowało
    n_agree:        int     = 0        # ile w zgodzie
    created_ts:     float   = field(default_factory=_TS)
    expires_ts:     float   = 0.0
    executed_ts:    float   = 0.0
    # Debug
    vote_breakdown: Dict    = field(default_factory=dict)
    council_scores: Dict    = field(default_factory=dict)


@dataclass
class OmegaError:
    """Błąd wykryty przez Omega Healer."""
    id:          str
    ts:          float
    message:     str
    category:    ErrorCat
    severity:    Severity
    bot_id:      int           = -1
    module:      str           = "unknown"
    tb:          str           = ""
    context:     Dict          = field(default_factory=dict)
    fix_attempts:int           = 0
    resolved:    bool          = False
    resolved_ts: float         = 0.0
    resolved_by: str           = ""


@dataclass
class HealAction:
    """Akcja naprawcza podjęta przez Omega Healer."""
    id:         str
    ts:         float
    error_id:   str
    strategy:   str
    bot_id:     int
    result:     HealResult
    duration_s: float
    details:    str
    xp_earned:  int = 0


# ══════════════════════════════════════════════════════════════════════════════════════════
# CAPITAL ENGINE — dynamiczne skalowanie kapitału
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class CapitalTier:
    """Jeden tier kapitałowy z pełnymi parametrami."""
    name:             str
    min_portfolio:    float   # min wartość portfela
    max_portfolio:    float   # max (inf = bez ograniczeń)
    base_margin_usd:  float   # bazowy margin per transakcja ($)
    max_concurrent:   int     # max jednoczesnych pozycji dla całego roju
    lev_min:          int
    lev_max:          int
    sl_mult:          float   # mnożnik SL (wyższy kapitał = ciaśniejszy SL)
    tp_mult:          float   # mnożnik TP
    max_hold_s:       int     # max czas trzymania pozycji


class CapitalEngine:
    """
    Dynamiczny silnik skalowania kapitału.
    
    KLUCZOWE: każdy bot przeznacza base_position_pct% AKTUALNEGO portfela per trade.
    Wraz ze wzrostem portfela automatycznie rosną pozycje.
    
    Przykład: portfel $3000, pct=0.0333 → $1/bot
              portfel $6000, pct=0.0333 → $2/bot  (automatycznie)
              portfel $30000, pct=0.0333 → $10/bot
    """

    TIERS: List[CapitalTier] = [
        CapitalTier("NANO",   0,      500,    0.50,   100,  10,  25,  1.2, 2.0,  60),
        CapitalTier("MICRO",  500,    2_000,  1.00,   200,  15,  50,  1.1, 2.0,  90),
        CapitalTier("SMALL",  2_000,  5_000,  2.00,   350,  20,  75,  1.0, 2.2, 120),
        CapitalTier("MED",    5_000,  15_000, 5.00,   500,  25,  100, 0.95,2.5, 180),
        CapitalTier("LARGE",  15_000, 50_000, 15.00,  750,  25,  125, 0.90,2.8, 300),
        CapitalTier("XLARGE", 50_000, 200_000,50.00,  1000, 20,  125, 0.85,3.0, 450),
        CapitalTier("ULTRA",  200_000,1_000_000,100.0,1500, 15,  125, 0.80,3.0, 600),
        CapitalTier("APEX",   1_000_000,float('inf'),200.0,2000,10, 125, 0.75,3.0, 900),
    ]

    def __init__(self, start_capital: float = 3_000.0):
        self._portfolio = start_capital
        self._lock = threading.Lock()
        self._log  = logging.getLogger("BITGOT·Capital")

    @property
    def portfolio(self) -> float:
        with self._lock: return self._portfolio

    @portfolio.setter
    def portfolio(self, v: float):
        with self._lock: self._portfolio = max(0.0, v)

    def current_tier(self, portfolio: Optional[float] = None) -> CapitalTier:
        p = portfolio if portfolio is not None else self._portfolio
        for t in reversed(self.TIERS):
            if p >= t.min_portfolio: return t
        return self.TIERS[0]

    def position_size_usd(self, portfolio: Optional[float] = None,
                           confidence: float = 0.80,
                           win_rate: float = 0.50) -> float:
        """
        Dynamiczne skalowanie pozycji:
        1. Base = portfolio × base_position_pct%
        2. Kelly adjustment dla confidence × WR
        3. Tier adjustment
        4. Clip do min/max
        """
        p = portfolio if portfolio is not None else self._portfolio
        t = self.current_tier(p)
        # Base position: proporcjonalne do portfela
        base = p * CFG.base_position_pct / 100.0
        base = max(base, t.base_margin_usd)
        # Kelly adjustment
        edge = confidence * win_rate - (1 - confidence) * (1 - win_rate)
        kelly = edge / max(confidence, 0.01) * CFG.kelly_fraction
        kelly = float(np.clip(kelly, 0.5, 1.5))
        # Confidence boost
        conf_mult = 1.0 + (confidence - CFG.confidence_threshold) * 2.0
        conf_mult = float(np.clip(conf_mult, 0.8, 1.5))
        size = base * kelly * conf_mult
        max_size = p * CFG.max_position_pct / 100.0
        return float(np.clip(size, CFG.min_position_usd, max_size))

    def leverage(self, win_rate: float = 0.50,
                  confidence: float = 0.80,
                  portfolio: Optional[float] = None) -> int:
        """Dynamiczna dźwignia bazująca na WR + confidence + tier."""
        t = self.current_tier(portfolio)
        # Bazowa dźwignia z WR
        if win_rate >= 0.90: lev = t.lev_max
        elif win_rate >= 0.87: lev = int(t.lev_max * 0.85)
        elif win_rate >= 0.85: lev = int(t.lev_max * 0.75)
        elif win_rate >= 0.80: lev = int(t.lev_max * 0.60)
        elif win_rate >= 0.75: lev = int(t.lev_max * 0.50)
        elif win_rate >= 0.70: lev = int(t.lev_max * 0.40)
        else: lev = t.lev_min
        # Confidence boost/reduction
        if confidence >= 0.95: lev = min(int(lev * 1.20), t.lev_max)
        elif confidence >= 0.90: lev = min(int(lev * 1.10), t.lev_max)
        elif confidence < 0.82: lev = max(int(lev * 0.80), t.lev_min)
        return int(np.clip(lev, t.lev_min, t.lev_max))

    def sl_tp_prices(self, entry: float, side: str,
                      win_rate: float = 0.70,
                      portfolio: Optional[float] = None) -> Tuple[float, float]:
        """SL i TP z dynamicznym dostosowaniem do aktualnego tiera."""
        t = self.current_tier(portfolio)
        sl_pct = CFG.sl_pct * t.sl_mult
        tp_pct = CFG.tp_pct * t.tp_mult
        # Wyższy WR → ciaśniejszy SL (bardziej pewni, mniej ryzyka)
        if win_rate >= 0.85:
            sl_pct *= 0.85; tp_pct *= 1.10
        elif win_rate >= 0.80:
            sl_pct *= 0.90
        elif win_rate < 0.60:
            sl_pct *= 1.20; tp_pct *= 0.90  # loose SL przy niskim WR
        if side == "long":
            return entry * (1 - sl_pct), entry * (1 + tp_pct)
        else:
            return entry * (1 + sl_pct), entry * (1 - tp_pct)

    def summary(self) -> Dict:
        t = self.current_tier()
        return {
            "portfolio":     self._portfolio,
            "tier":          t.name,
            "base_margin":   t.base_margin_usd,
            "example_size":  self.position_size_usd(confidence=0.85, win_rate=0.85),
            "leverage_range":f"x{t.lev_min}–x{t.lev_max}",
            "max_positions": t.max_concurrent,
        }


# ══════════════════════════════════════════════════════════════════════════════════════════
# BITGET CONNECTOR — dedykowany konektor z kompletnym API
# ══════════════════════════════════════════════════════════════════════════════════════════

class BitgetConnector:
    """
    Dedykowany konektor dla Bitget Exchange.
    Obsługuje: Futures USDT-M, Spot
    Features:
    - Auto-retry z exponential backoff
    - Rate limiting (300 req/min)
    - WebSocket tick streaming
    - Order lifecycle management
    - Position monitoring
    - Funding rate history
    - Open Interest tracking
    - Liquidation zone analysis
    """

    RATE_LIMIT_PER_MIN = 300
    MAX_RETRIES        = 4
    BACKOFF_BASE       = 1.5

    def __init__(self, cfg: BITGOTConfig = CFG):
        self.cfg  = cfg
        self._log = logging.getLogger("BITGOT·Bitget")
        self._ex  = None    # ccxt exchange instance
        self._ex_async = None
        self._markets: Dict[str, Any] = {}
        self._healthy = False
        self._last_req_ts: float = 0.0
        self._req_count_min: int = 0
        self._req_window_start: float = _TS()
        self._lock = asyncio.Lock()
        self._api_errors: deque = deque(maxlen=100)
        self._latency_ms: deque = deque(maxlen=50)

    async def connect(self) -> bool:
        """Inicjalizuj połączenie z Bitget."""
        if not ccxt_async:
            self._log.warning("ccxt nie zainstalowane — tryb symulacji")
            self._healthy = False
            return False
        try:
            params = {
                "apiKey":   self.cfg.api_key,
                "secret":   self.cfg.api_secret,
                "password": self.cfg.api_passphrase,
                "enableRateLimit": True,
                "rateLimit": 200,
                "timeout":   15_000,
                "options": {
                    "defaultType":  "swap",
                    "marginMode":   "cross",
                    "fetchMarkets": {"type": "swap"},
                },
            }
            cls = getattr(ccxt_async, "bitget", None)
            if cls is None:
                self._log.error("ccxt.bitget nie znaleziono")
                return False
            if self.cfg.testnet:
                params["urls"] = {"api": {"public": "https://api.bitget.com"}}
            self._ex = cls(params)
            self._log.info("📡 Łączę z Bitget...")
            t0 = _TS()
            self._markets = await self._ex.load_markets()
            lat = (_TS() - t0) * 1000
            self._latency_ms.append(lat)
            self._healthy = True
            self._log.info(f"✅ Bitget połączony | {len(self._markets)} rynków | "
                           f"latency={lat:.0f}ms")
            return True
        except Exception as e:
            self._log.error(f"❌ Bitget connect: {e}")
            self._healthy = False
            return False

    async def _rate_limit(self):
        now = _TS()
        if now - self._req_window_start >= 60:
            self._req_count_min = 0
            self._req_window_start = now
        if self._req_count_min >= self.RATE_LIMIT_PER_MIN:
            wait = 60 - (now - self._req_window_start)
            if wait > 0:
                self._log.debug(f"Rate limit — waiting {wait:.1f}s")
                await asyncio.sleep(wait)
            self._req_count_min = 0
            self._req_window_start = _TS()
        self._req_count_min += 1

    async def _safe_call(self, coro, fallback=None, name: str = ""):
        """Wrapper z retry + backoff + latency tracking."""
        for attempt in range(self.MAX_RETRIES):
            try:
                await self._rate_limit()
                t0 = _TS()
                result = await coro
                lat = (_TS() - t0) * 1000
                self._latency_ms.append(lat)
                return result
            except Exception as e:
                msg = str(e).lower()
                self._api_errors.append((_TS(), str(e)))
                # Non-retryable
                if any(k in msg for k in ["invalid key","unauthorized","403","invalid api"]):
                    self._log.error(f"🔑 Auth error [{name}]: {e}")
                    return fallback
                # Rate limit
                if any(k in msg for k in ["rate limit","429","too many"]):
                    wait = 60.0
                    self._log.warning(f"⏱ Rate limited [{name}] — wait {wait}s")
                    await asyncio.sleep(wait)
                    continue
                # Retryable
                wait = self.BACKOFF_BASE ** attempt
                self._log.debug(f"[{name}] attempt {attempt+1}/{self.MAX_RETRIES}: {e} — retry {wait:.1f}s")
                await asyncio.sleep(wait)
        return fallback

    async def fetch_all_markets(self) -> Dict[str, Any]:
        """Pobierz wszystkie rynki Bitget."""
        if not self._healthy: return {}
        return self._markets

    async def fetch_ticker(self, symbol: str) -> Dict:
        if not self._healthy: return {}
        return await self._safe_call(
            self._ex.fetch_ticker(symbol), {}, f"ticker:{symbol}"
        ) or {}

    async def fetch_tickers(self, symbols: Optional[List[str]] = None) -> Dict:
        if not self._healthy: return {}
        try:
            if symbols:
                return await self._safe_call(
                    self._ex.fetch_tickers(symbols), {}, "tickers_batch"
                ) or {}
            else:
                return await self._safe_call(
                    self._ex.fetch_tickers(), {}, "tickers_all"
                ) or {}
        except Exception: return {}

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "5m",
                           limit: int = 200) -> List:
        if not self._healthy: return []
        return await self._safe_call(
            self._ex.fetch_ohlcv(symbol, timeframe, limit=limit),
            [], f"ohlcv:{symbol}"
        ) or []

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        if not self._healthy: return {"bids":[],"asks":[]}
        return await self._safe_call(
            self._ex.fetch_order_book(symbol, limit=limit),
            {"bids":[],"asks":[]}, f"ob:{symbol}"
        ) or {"bids":[],"asks":[]}

    async def fetch_funding_rate(self, symbol: str) -> float:
        if not self._healthy: return 0.0
        try:
            data = await self._safe_call(
                self._ex.fetch_funding_rate(symbol), {}, f"funding:{symbol}"
            )
            return float(data.get("fundingRate", 0) or 0) if data else 0.0
        except Exception: return 0.0

    async def fetch_open_interest(self, symbol: str) -> float:
        if not self._healthy: return 0.0
        try:
            data = await self._safe_call(
                self._ex.fetch_open_interest(symbol), {}, f"oi:{symbol}"
            )
            return float(data.get("openInterest", 0) or 0) if data else 0.0
        except Exception: return 0.0

    async def fetch_balance(self) -> float:
        """Pobierz dostępne saldo USDT."""
        if not self._healthy: return 0.0
        try:
            bal = await self._safe_call(self._ex.fetch_balance(), {}, "balance")
            if not bal: return 0.0
            usdt = bal.get("USDT", {})
            if isinstance(usdt, dict): return float(usdt.get("free", 0) or 0)
            return float(usdt or 0)
        except Exception: return 0.0

    async def set_leverage(self, symbol: str, leverage: int,
                            margin_mode: str = "cross") -> bool:
        if not self._healthy or self.cfg.paper_mode: return True
        try:
            await self._safe_call(
                self._ex.set_leverage(leverage, symbol,
                                       params={"marginMode": margin_mode}),
                None, f"set_lev:{symbol}"
            )
            return True
        except Exception: return False

    async def create_order(self, symbol: str, order_type: str,
                            side: str, qty: float,
                            price: Optional[float] = None,
                            params: Optional[Dict] = None) -> Optional[Dict]:
        if self.cfg.paper_mode:
            return {"id": f"paper_{uuid.uuid4().hex[:8]}",
                    "status": "closed", "average": price or 1.0}
        if not self._healthy: return None
        kw = params or {}
        try:
            return await self._safe_call(
                self._ex.create_order(symbol, order_type, side,
                                       qty, price, kw),
                None, f"order:{symbol}"
            )
        except Exception as e:
            self._log.error(f"Order error {symbol}: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        if self.cfg.paper_mode: return True
        if not self._healthy: return False
        try:
            await self._safe_call(
                self._ex.cancel_order(order_id, symbol),
                None, f"cancel:{order_id}"
            )
            return True
        except Exception: return False

    async def fetch_position(self, symbol: str) -> Optional[Dict]:
        if self.cfg.paper_mode: return None
        if not self._healthy: return None
        try:
            positions = await self._safe_call(
                self._ex.fetch_position(symbol), None, f"pos:{symbol}"
            )
            return positions
        except Exception: return None

    async def close_position(self, symbol: str, side: str, qty: float) -> Optional[Dict]:
        """Market close pozycji."""
        close_side = "sell" if side == "long" else "buy"
        return await self.create_order(
            symbol, "market", close_side, qty,
            params={"reduceOnly": True, "marginMode": "cross"}
        )

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_ms: return 0.0
        return float(np.mean(list(self._latency_ms)))

    @property
    def is_healthy(self) -> bool: return self._healthy

    def error_rate_1min(self) -> float:
        now = _TS()
        recent = [t for t,_ in self._api_errors if now-t < 60]
        return len(recent) / 60.0

    async def close(self):
        if self._ex:
            with suppress(Exception):
                await self._ex.close()


# ══════════════════════════════════════════════════════════════════════════════════════════
# PAIR DISCOVERY — odkrywanie i rankowanie par Bitget
# ══════════════════════════════════════════════════════════════════════════════════════════

class PairDiscovery:
    """
    Odkrywanie WSZYSTKICH par handlowych na Bitget.
    Przydziela unikalne pary dla 3000 botów.
    Priorytet: Futures USDT-M (największy wolumen) → Futures Coin → Spot
    """

    def __init__(self, connector: BitgetConnector, cfg: BITGOTConfig = CFG):
        self.connector = connector
        self.cfg       = cfg
        self._pairs: List[PairInfo] = []
        self._assigned: Set[str] = set()  # klucze już przydzielonych par
        self._log = logging.getLogger("BITGOT·PairDisc")

    async def discover(self) -> List[PairInfo]:
        """Odkryj i ocen wszystkie dostępne pary."""
        self._log.info("🔍 Skanowanie par Bitget...")
        all_pairs: List[PairInfo] = []

        markets = await self.connector.fetch_all_markets()
        if not markets:
            self._log.warning("Brak rynków — generuję syntetyczne pary")
            return self._synthetic_pairs(self.cfg.n_bots)

        # Pobierz tickery (masowo, jeden call)
        try:
            tickers = await self.connector.fetch_tickers()
        except Exception:
            tickers = {}

        for sym, mkt in markets.items():
            try:
                if not mkt.get("active", True): continue
                mtype = self._classify_market(mkt)
                if mtype is None: continue

                # Dane tickera
                tk = tickers.get(sym, {})
                if not tk:
                    continue

                last  = float(tk.get("last")   or 0)
                bid   = float(tk.get("bid")    or 0)
                ask   = float(tk.get("ask")    or 0)
                vol24 = float(tk.get("quoteVolume") or tk.get("baseVolume") or 0)
                hi    = float(tk.get("high")   or last)
                lo    = float(tk.get("low")    or last)

                if last <= 0 or bid <= 0 or ask <= 0: continue
                if vol24 < self.cfg.min_volume_usdt_24h: continue

                spread_pct = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 100
                if spread_pct > self.cfg.max_spread_pct: continue

                vol_24h_pct = (hi - lo) / (last + 1e-12) * 100
                vol_1h_pct  = vol_24h_pct / 24
                if vol_1h_pct < self.cfg.min_vol_1h_pct: continue

                # Skalowanie score [0,1]
                vol_score     = min(vol_1h_pct / 0.5, 1.0)
                spread_score  = 1.0 - spread_pct / self.cfg.max_spread_pct
                volume_score  = min(math.log10(vol24 + 1) / 10, 1.0)
                futures_bonus = 0.15 if mtype == MarketType.FUTURES_USDT else 0.0

                score = (0.40 * vol_score + 0.25 * spread_score +
                         0.20 * volume_score + 0.15 * futures_bonus)

                limits = mkt.get("limits", {})
                min_qty   = float(limits.get("amount", {}).get("min", 1.0) or 1.0)
                min_notio = float(limits.get("cost",   {}).get("min", 5.0) or 5.0)
                prec      = mkt.get("precision", {})

                base  = mkt.get("base",  sym.split("/")[0])
                quote = mkt.get("quote", "USDT")

                pair = PairInfo(
                    symbol         = sym,
                    base           = base,
                    quote          = quote,
                    market_type    = mtype,
                    score          = float(score),
                    spread_pct     = float(spread_pct),
                    vol_1h_pct     = float(vol_1h_pct),
                    volume_24h_usdt= float(vol24),
                    ticks_per_hour = float(max(vol24 / (last + 1e-12) * 5, 60)),
                    avg_volatility = float(vol_1h_pct),
                    min_qty        = float(min_qty),
                    min_notional   = float(min_notio),
                    price_precision= int(prec.get("price", 6)),
                    qty_precision  = int(prec.get("amount", 4)),
                    max_leverage   = int(mkt.get("limits", {}).get("leverage", {}).get("max", 125) or 125),
                    current_price  = float(last),
                    tier           = self._score_to_tier(score),
                )
                all_pairs.append(pair)

            except Exception as e:
                self._log.debug(f"Pair {sym}: {e}")
                continue

        all_pairs.sort(key=lambda p: p.score, reverse=True)
        self._pairs = all_pairs
        self._log.info(f"✅ {len(all_pairs)} par odkrytych i ocenionych")
        return all_pairs

    def _classify_market(self, mkt: Dict) -> Optional[MarketType]:
        t    = str(mkt.get("type", "")).lower()
        sub  = str(mkt.get("subType", "")).lower()
        quote= str(mkt.get("quote", "")).upper()
        if t in ("swap", "future"):
            if quote in ("USDT", "USD"): return MarketType.FUTURES_USDT
            return MarketType.FUTURES_COIN
        if t == "spot": return MarketType.SPOT
        return None

    def _score_to_tier(self, score: float) -> BotTier:
        if score >= 0.80: return BotTier.APEX
        if score >= 0.65: return BotTier.ELITE
        if score >= 0.45: return BotTier.STANDARD
        return BotTier.SCOUT

    def assign_unique(self, n: int) -> List[PairInfo]:
        """
        Przydziel UNIKALNE pary.
        Każda (symbol) pojawia się tylko RAZ.
        Priorytet: FUTURES_USDT → FUTURES_COIN → SPOT
        """
        assigned: List[PairInfo] = []
        seen: Set[str] = set()

        # Priority: futures first
        sorted_pairs = sorted(self._pairs, key=lambda p: (
            0 if p.market_type == MarketType.FUTURES_USDT else
            1 if p.market_type == MarketType.FUTURES_COIN else 2,
            -p.score
        ))

        for p in sorted_pairs:
            if p.symbol in seen: continue
            seen.add(p.symbol)
            assigned.append(p)
            if len(assigned) >= n: break

        # Pad z syntetycznymi jeśli za mało
        if len(assigned) < n:
            self._log.warning(f"Tylko {len(assigned)} real par — padding do {n}")
            synth = self._synthetic_pairs(n - len(assigned), offset=len(assigned))
            for sp in synth:
                if sp.symbol not in seen:
                    seen.add(sp.symbol)
                    assigned.append(sp)

        self._log.info(f"🎯 Przydzielono {len(assigned)} unikalnych par dla {n} botów")

        # Statystyki
        by_type = Counter(p.market_type.value for p in assigned[:n])
        by_tier = Counter(p.tier.value for p in assigned[:n])
        self._log.info(f"   Typy: {dict(by_type)}")
        self._log.info(f"   Tiery: {dict(by_tier)}")

        return assigned[:n]

    def _synthetic_pairs(self, n: int, offset: int = 0) -> List[PairInfo]:
        """Generuj syntetyczne pary do paper tradingu gdy API niedostępne."""
        COINS = [
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

        TYPES = [MarketType.FUTURES_USDT, MarketType.FUTURES_USDT, MarketType.SPOT]
        result = []
        used: Set[str] = set()

        for i in range(n):
            for attempt in range(100):
                coin = COINS[(offset + i + attempt) % len(COINS)]
                mtype = TYPES[i % len(TYPES)]
                suffix = (offset + i) // len(COINS)
                coin_full = f"{coin}{suffix if suffix else ''}"
                if mtype == MarketType.FUTURES_USDT:
                    sym = f"{coin_full}/USDT:USDT"
                elif mtype == MarketType.FUTURES_COIN:
                    sym = f"{coin_full}/USD:{coin_full}"
                else:
                    sym = f"{coin_full}/USDT"
                if sym not in used:
                    used.add(sym)
                    break
            else:
                sym = f"SYNTH{offset+i}/USDT:USDT"

            score = max(0.10, 0.90 - i / max(n, 1) * 0.60)
            vol = max(1_000_000, 50_000_000 - i * 15_000)
            result.append(PairInfo(
                symbol          = sym,
                base            = sym.split("/")[0],
                quote           = "USDT",
                market_type     = mtype,
                score           = score,
                spread_pct      = 0.02 + i / n * 0.04,
                vol_1h_pct      = max(0.08, 0.40 - i / n * 0.20),
                volume_24h_usdt = vol,
                ticks_per_hour  = max(60, 500 - i // 10),
                avg_volatility  = max(0.08, 0.35 - i / n * 0.15),
                min_qty         = 1.0,
                min_notional    = 5.0,
                current_price   = random.uniform(0.01, 50_000),
                tier            = self._score_to_tier(score),
            ))
        return result

    def refresh_scores(self, new_tickers: Dict):
        """Odśwież score par na podstawie nowych tickerów."""
        for pair in self._pairs:
            tk = new_tickers.get(pair.symbol, {})
            if not tk: continue
            try:
                last = float(tk.get("last") or pair.current_price)
                vol24 = float(tk.get("quoteVolume") or pair.volume_24h_usdt)
                pair.current_price = last
                pair.volume_24h_usdt = vol24
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════════════════
# MATH CORE — wszystkie obliczenia numeryczne (czyste NumPy, zero pandas)
# ══════════════════════════════════════════════════════════════════════════════════════════

class MathCore:
    """
    Ultra-wydajna biblioteka wskaźników technicznych.
    Zero pandas. Zero zewnętrznych zależności.
    Każda metoda zoptymalizowana dla szybkości.
    """

    # ── Primitive activations ────────────────────────────────────────────────────
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -10, 10))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-12)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return x * MathCore.sigmoid(1.702 * x)

    # ── EMA / SMA ────────────────────────────────────────────────────────────────
    @staticmethod
    def ema(arr: np.ndarray, span: int) -> np.ndarray:
        """Exponential Moving Average (Wilder's smoothing)."""
        out = np.full(len(arr), np.nan, dtype=float)
        if len(arr) < span: return out
        k = 2.0 / (span + 1)
        out[span - 1] = arr[:span].mean()
        for i in range(span, len(arr)):
            out[i] = arr[i] * k + out[i-1] * (1 - k)
        return out

    @staticmethod
    def ema_scalar(values: List[float], span: int) -> float:
        """Fast EMA scalar for live streaming."""
        if not values: return 0.0
        k = 2.0 / (span + 1)
        e = values[0]
        for v in values[1:]: e = v * k + e * (1 - k)
        return float(e)

    @staticmethod
    def sma(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(len(arr), np.nan, dtype=float)
        for i in range(p - 1, len(arr)):
            out[i] = arr[i-p+1:i+1].mean()
        return out

    @staticmethod
    def kama(arr: np.ndarray, er_p: int = 10,
              fast: int = 2, slow: int = 30) -> np.ndarray:
        """Kaufman's Adaptive Moving Average."""
        out = arr.astype(float).copy()
        if len(arr) < er_p + 1: return out
        fs = 2.0 / (fast + 1); ss = 2.0 / (slow + 1)
        for i in range(er_p, len(arr)):
            change    = abs(arr[i] - arr[i - er_p])
            vol       = sum(abs(arr[j] - arr[j-1]) for j in range(i - er_p + 1, i+1))
            er        = change / max(vol, 1e-12)
            sc        = (er * (fs - ss) + ss) ** 2
            out[i]    = out[i-1] + sc * (arr[i] - out[i-1])
        return out

    @staticmethod
    def wma(arr: np.ndarray, p: int) -> np.ndarray:
        """Weighted Moving Average (linear weights)."""
        out = np.full(len(arr), np.nan, dtype=float)
        weights = np.arange(1, p + 1, dtype=float)
        w_sum = weights.sum()
        for i in range(p - 1, len(arr)):
            out[i] = (arr[i-p+1:i+1] * weights).sum() / w_sum
        return out

    # ── RSI ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(arr: np.ndarray, p: int = 14) -> np.ndarray:
        """Wilder's RSI — full series."""
        out = np.full(len(arr), np.nan, dtype=float)
        if len(arr) < p + 1: return out
        d = np.diff(arr)
        g = np.where(d > 0, d, 0.0)
        l = np.where(d < 0, -d, 0.0)
        ag = g[:p].mean(); al = l[:p].mean()
        out[p] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
        for i in range(p, len(d)):
            ag = (ag * (p - 1) + g[i]) / p
            al = (al * (p - 1) + l[i]) / p
            out[i + 1] = 100.0 if al < 1e-12 else 100.0 - 100.0 / (1.0 + ag / al)
        return out

    @staticmethod
    def rsi_scalar(prices: Sequence[float], p: int = 14) -> float:
        """Fast RSI scalar [0..1]."""
        if len(prices) < p + 1: return 0.5
        arr = np.array(prices[-(p+2):], dtype=float)
        d = np.diff(arr)
        g = float(d[d > 0].mean()) if (d > 0).any() else 0.0
        l = float(-d[d < 0].mean()) if (d < 0).any() else 1e-10
        return g / (g + l)

    @staticmethod
    def rsi_slope(prices: Sequence[float], p: int = 14,
                   slope_p: int = 5) -> float:
        """RSI slope over last N periods."""
        if len(prices) < p + slope_p + 2: return 0.0
        vals = [MathCore.rsi_scalar(list(prices)[:-(slope_p-i)] if i > 0 else list(prices), p)
                for i in range(slope_p)]
        try: return float(np.polyfit(range(slope_p), vals[::-1], 1)[0])
        except: return 0.0

    # ── MACD ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def macd(arr: np.ndarray, fast: int = 12, slow: int = 26,
              sig: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (macd_line, signal_line, histogram, cross)."""
        ef = MathCore.ema(arr, fast)
        es = MathCore.ema(arr, slow)
        ml = ef - es
        sl = MathCore.ema(np.where(np.isnan(ml), 0, ml), sig)
        hist = ml - sl
        cross = np.zeros(len(arr), dtype=int)
        for i in range(1, len(hist)):
            if not (np.isnan(hist[i]) or np.isnan(hist[i-1])):
                cross[i] = int(np.sign(hist[i])) - int(np.sign(hist[i-1]))
        return ml, sl, hist, cross

    @staticmethod
    def macd_scalar(prices: Sequence[float]) -> Tuple[float, float, int]:
        """(histogram, signal, cross) — fast scalar."""
        if len(prices) < 35: return 0.0, 0.0, 0
        arr = np.array(prices, dtype=float)
        _,_,hist,cross = MathCore.macd(arr)
        h = float(hist[-1]) if not np.isnan(hist[-1]) else 0.0
        s = float(hist[-1]-hist[-2]) if len(hist)>1 and not np.isnan(hist[-2]) else 0.0
        c = int(cross[-1])
        return h, s, c

    # ── Bollinger Bands ───────────────────────────────────────────────────────────
    @staticmethod
    def bollinger(arr: np.ndarray, p: int = 20,
                   k: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = MathCore.sma(arr, p)
        std = np.array([np.std(arr[max(0,i-p+1):i+1]) for i in range(len(arr))])
        return m + k * std, m, m - k * std

    @staticmethod
    def bb_scalar(prices: Sequence[float], p: int = 20,
                   k: float = 2.0) -> Tuple[float, float, float]:
        """(bb_pos [0,1], bb_squeeze, bb_width)."""
        if len(prices) < p: return 0.5, 0.0, 0.0
        w = np.array(prices[-p:], dtype=float)
        m, s = w.mean(), w.std() + 1e-12
        up, lo = m + k * s, m - k * s
        pos = float(np.clip((prices[-1] - lo) / (up - lo), 0, 1))
        kcu = m + 1.5 * s  # simplified Keltner
        sq  = float(up < kcu)
        return pos, sq, float(2 * k * s / (m + 1e-12))

    # ── ATR ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, p: int = 14) -> np.ndarray:
        n = len(close)
        tr = np.zeros(n, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                         abs(high[i] - close[i-1]),
                         abs(low[i]  - close[i-1]))
        out = np.full(n, np.nan, dtype=float)
        if n >= p:
            out[p-1] = tr[:p].mean()
            for i in range(p, n):
                out[i] = (out[i-1] * (p-1) + tr[i]) / p
        return out

    @staticmethod
    def atr_scalar(high: Sequence[float], low: Sequence[float],
                    close: Sequence[float], p: int = 14) -> float:
        n = min(len(close), p * 3)
        if n < 2: return 0.0
        h, l, c = (np.array(x[-n:], dtype=float) for x in (high, low, close))
        a = MathCore.atr(h, l, c, p)
        v = a[~np.isnan(a)]
        return float(v[-1]) if len(v) else 0.0

    # ── ADX ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, p: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ADX, +DI, -DI)."""
        n = len(close)
        tr  = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
        for i in range(1, n):
            tr[i]  = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            pu = high[i] - high[i-1]; nd = low[i-1] - low[i]
            pdm[i] = max(pu, 0) if pu > nd else 0
            ndm[i] = max(nd, 0) if nd > pu else 0

        def smma(x, n):
            out = np.full(len(x), np.nan)
            if len(x) < n: return out
            out[n-1] = x[:n].mean()
            for i in range(n, len(x)):
                out[i] = out[i-1] * (n-1) / n + x[i] / n
            return out

        tr14  = smma(tr, p); pdi = 100 * smma(pdm, p) / np.maximum(tr14, 1e-10)
        ndi   = 100 * smma(ndm, p) / np.maximum(tr14, 1e-10)
        dx    = 100 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-10)
        adx_v = smma(np.where(np.isnan(dx), 0, dx), p)
        return adx_v, pdi, ndi

    @staticmethod
    def adx_scalar(high: Sequence[float], low: Sequence[float],
                    close: Sequence[float], p: int = 14) -> Tuple[float, float]:
        """(adx_value, adx_slope)."""
        n = min(len(close), p * 4)
        if n < p * 2: return 25.0, 0.0
        h, l, c = (np.array(x[-n:], dtype=float) for x in (high, low, close))
        adx_arr, _, _ = MathCore.adx(h, l, c, p)
        v = adx_arr[~np.isnan(adx_arr)]
        if len(v) < 2: return 25.0, 0.0
        return float(v[-1]), float(v[-1] - v[-2])

    # ── Stochastic RSI ────────────────────────────────────────────────────────────
    @staticmethod
    def stoch_rsi(arr: np.ndarray, rsi_p: int = 14,
                   stoch_p: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        rsi_arr = MathCore.rsi(arr, rsi_p)
        k = np.full(len(arr), np.nan, dtype=float)
        for i in range(stoch_p - 1, len(rsi_arr)):
            w = rsi_arr[i-stoch_p+1:i+1]
            if np.all(~np.isnan(w)):
                lo, hi = w.min(), w.max()
                k[i] = (rsi_arr[i] - lo) / (hi - lo + 1e-10) * 100
        d = MathCore.sma(np.where(np.isnan(k), 0, k), 3)
        return k, d

    # ── Ichimoku ─────────────────────────────────────────────────────────────────
    @staticmethod
    def ichimoku(high: np.ndarray, low: np.ndarray,
                  t: int = 9, k: int = 26, s: int = 52) -> Dict[str, np.ndarray]:
        def midline(n):
            out = np.full(len(high), np.nan, dtype=float)
            for i in range(n-1, len(high)):
                out[i] = (high[i-n+1:i+1].max() + low[i-n+1:i+1].min()) / 2
            return out
        ten = midline(t); kij = midline(k)
        sa  = (ten + kij) / 2
        sb  = midline(s)
        sa_disp = np.concatenate([np.full(k, np.nan), sa[:-k]])
        sb_disp = np.concatenate([np.full(k, np.nan), sb[:-k]])
        return {"tenkan": ten, "kijun": kij, "senkou_a": sa_disp, "senkou_b": sb_disp}

    # ── Hurst Exponent ────────────────────────────────────────────────────────────
    @staticmethod
    def hurst(arr: np.ndarray) -> float:
        """R/S Analysis — Hurst exponent [0,1]. 0.5=random, >0.5=trending, <0.5=mean-rev."""
        n = len(arr)
        if n < 32: return 0.5
        lags, rs = [], []
        for seg in [16, 32, 64]:
            if seg > n: break
            s = arr[-seg:]
            ret = np.diff(np.log(np.maximum(s, 1e-12)))
            if len(ret) < 2: continue
            cumdev = np.cumsum(ret - ret.mean())
            R = cumdev.max() - cumdev.min()
            S = ret.std() + 1e-12
            rs.append(R / S); lags.append(seg)
        if len(lags) < 2: return 0.5
        try:
            return float(np.clip(np.polyfit(np.log(lags), np.log(rs), 1)[0], 0.01, 1.99))
        except: return 0.5

    # ── VWAP ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def vwap(high: np.ndarray, low: np.ndarray,
              close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        tp = (high + low + close) / 3
        cum_v = np.cumsum(volume + 1e-12)
        cum_pv = np.cumsum(tp * volume)
        return cum_pv / cum_v

    @staticmethod
    def vwap_scalar(high: Sequence[float], low: Sequence[float],
                     close: Sequence[float], volume: Sequence[float],
                     p: int = 20) -> float:
        n = min(len(close), p)
        if n < 2: return float(close[-1]) if close else 0.0
        tp = (np.array(high[-n:]) + np.array(low[-n:]) + np.array(close[-n:])) / 3
        v  = np.array(volume[-n:]) + 1e-12
        return float((tp * v).sum() / v.sum())

    # ── OBV ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        out = np.zeros(len(close), dtype=float)
        for i in range(1, len(close)):
            out[i] = out[i-1] + (volume[i] if close[i] > close[i-1]
                                   else -volume[i] if close[i] < close[i-1]
                                   else 0)
        return out

    # ── Williams %R ───────────────────────────────────────────────────────────────
    @staticmethod
    def williams_r(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, p: int = 14) -> np.ndarray:
        out = np.full(len(close), np.nan, dtype=float)
        for i in range(p-1, len(close)):
            hh = high[i-p+1:i+1].max(); ll = low[i-p+1:i+1].min()
            out[i] = (hh - close[i]) / (hh - ll + 1e-10) * -100
        return out

    # ── Swing Structure ───────────────────────────────────────────────────────────
    @staticmethod
    def swing_structure(high: np.ndarray, low: np.ndarray,
                         pivot_p: int = 5) -> Dict[str, bool]:
        n = len(high)
        ph = [i for i in range(pivot_p, n - pivot_p)
              if all(high[i] >= high[j]
                     for j in range(i-pivot_p, i+pivot_p+1) if j != i)]
        pl = [i for i in range(pivot_p, n - pivot_p)
              if all(low[i] <= low[j]
                     for j in range(i-pivot_p, i+pivot_p+1) if j != i)]
        hh_hl = (len(ph) >= 2 and len(pl) >= 2 and
                  high[ph[-1]] > high[ph[-2]] and low[pl[-1]] > low[pl[-2]])
        lh_ll = (len(ph) >= 2 and len(pl) >= 2 and
                  high[ph[-1]] < high[ph[-2]] and low[pl[-1]] < low[pl[-2]])
        bos_up = len(ph) >= 1 and high[-1] > high[ph[-1]]
        bos_dn = len(pl) >= 1 and low[-1] < low[pl[-1]]
        return {"hh_hl": hh_hl, "lh_ll": lh_ll, "bos_up": bos_up, "bos_dn": bos_dn}

    # ── Order Flow Imbalance ──────────────────────────────────────────────────────
    @staticmethod
    def ofi(bids: List[Tuple[float, float]],
             asks: List[Tuple[float, float]]) -> float:
        """Order Flow Imbalance ∈ [-1, +1]."""
        bid_v = sum(p * v for p, v in bids[:10])
        ask_v = sum(p * v for p, v in asks[:10])
        total = bid_v + ask_v + 1e-12
        return float((bid_v - ask_v) / total)

    @staticmethod
    def ob_entropy(bids: List[Tuple[float,float]],
                    asks: List[Tuple[float,float]], levels: int = 20) -> float:
        """Shannon entropy rozkładu order book."""
        vols = [b[1] for b in bids[:levels]] + [a[1] for a in asks[:levels]]
        total = sum(vols) + 1e-12
        probs = [v / total for v in vols if v > 0]
        return float(-sum(p * math.log2(p + 1e-12) for p in probs)) / max(math.log2(len(probs) + 1), 1)

    @staticmethod
    def vpin(buy_vol_hist: Sequence[float], sell_vol_hist: Sequence[float],
              window: int = 20) -> float:
        """VPIN proxy (Volume-synchronized PIN)."""
        if len(buy_vol_hist) < 2: return 0.3
        bv = np.array(list(buy_vol_hist)[-window:])
        sv = np.array(list(sell_vol_hist)[-window:])
        total = bv + sv + 1e-12
        return float(np.mean(np.abs(bv - sv) / total))

    # ── Point of Control ─────────────────────────────────────────────────────────
    @staticmethod
    def poc(close: np.ndarray, volume: np.ndarray, bins: int = 50) -> float:
        """Price level with maximum traded volume."""
        if len(close) < 2: return float(close[-1]) if len(close) else 0.0
        n = min(len(close), 200)
        hist, edges = np.histogram(close[-n:], bins=bins,
                                    weights=volume[-n:] if len(volume) >= n else None)
        idx = int(hist.argmax())
        return float((edges[idx] + edges[idx+1]) / 2)

    # ── Realized Volatility ───────────────────────────────────────────────────────
    @staticmethod
    def realized_vol(close: np.ndarray, p: int = 24) -> float:
        """Annualized realized volatility (hourly data assumed)."""
        if len(close) < 2: return 0.0
        rets = np.diff(np.log(np.maximum(close[-(p+1):], 1e-12)))
        return float(rets.std() * math.sqrt(24 * 365))

    # ── Kelly Criterion ───────────────────────────────────────────────────────────
    @staticmethod
    def kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
        """Full Kelly fraction [0, 0.5]."""
        if avg_loss_pct <= 0: return 0.0
        b = avg_win_pct / avg_loss_pct
        k = (b * win_rate - (1 - win_rate)) / b
        return float(np.clip(k, 0.0, 0.50))

    # ── Autocorrelation ───────────────────────────────────────────────────────────
    @staticmethod
    def autocorr(arr: np.ndarray, lag: int = 1) -> float:
        """Autocorrelation at specified lag."""
        if len(arr) < lag + 2: return 0.0
        try:
            c = np.corrcoef(arr[:-lag], arr[lag:])
            return float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
        except: return 0.0


# ══════════════════════════════════════════════════════════════════════════════════════════
# GENOME — DNA każdego bota (ewoluuje przez CMA-ES)
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class BotGenome:
    """
    DNA bota — wszystkie mutowalny parametry.
    
    Podzielony na grupy:
    1. Wagi sygnałów (jak ważyć każdy komponent)
    2. Parametry techniczne (okresy wskaźników)
    3. Parametry wejścia (progi pewności)
    4. Parametry zarządzania pozycją
    5. Parametry meta-uczenia
    """

    gid:         str  = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation:  int  = 0
    bot_id:      int  = -1

    # ── Wagi komponentów sygnału [suma=1] ─────────────────────────────────────
    w_rl:        float = 0.45  # waga głosowania RL engines
    w_neural:    float = 0.30  # waga NeuralSwarm
    w_micro:     float = 0.15  # waga MicroSignal
    w_regime:    float = 0.10  # waga Regime Oracle

    # ── Wagi poszczególnych wskaźników (w RL state) ───────────────────────────
    w_rsi:       float = 0.08; w_macd:    float = 0.08
    w_ema:       float = 0.07; w_kama:    float = 0.06
    w_bb:        float = 0.06; w_adx:     float = 0.07
    w_funding:   float = 0.10; w_oi:      float = 0.09
    w_ob:        float = 0.09; w_cvd:     float = 0.08
    w_toxic:     float = 0.07; w_hurst:   float = 0.06
    w_regime_w:  float = 0.09  # regime weight in signal

    # ── Parametry techniczne ──────────────────────────────────────────────────
    rsi_period:  int   = 14;  rsi_fast:  int   = 7
    rsi_ob:      float = 75.0;rsi_os:    float = 25.0
    ema_fast:    int   = 8;   ema_slow:  int   = 21; ema_trend: int = 89
    macd_fast:   int   = 12;  macd_slow: int   = 26; macd_sig:  int = 9
    bb_period:   int   = 20;  bb_std:    float = 2.0
    kama_period: int   = 10;  atr_period:int   = 14
    adx_period:  int   = 14;  vol_period:int   = 20

    # ── Parametry wejścia ────────────────────────────────────────────────────
    confidence_min:   float = 0.80   # min confidence (odpowiada CFG.confidence_threshold)
    signal_threshold: float = 0.35   # min |signal| do wejścia
    engines_agree_min:int   = 13     # min engines RL w zgodzie (z 25)
    min_regime_conf:  float = 0.55   # min pewność reżimu

    # ── Parametry Kelly ──────────────────────────────────────────────────────
    kelly_frac:       float = 0.25   # Kelly fraction (konserwatywny)
    kelly_min_trades: int   = 20     # min trades dla Kelly

    # ── Parametry pozycji ────────────────────────────────────────────────────
    sl_pct:           float = 0.006  # 0.6% stop loss
    tp_pct:           float = 0.012  # 1.2% take profit (2:1 RR)
    trail_arm_pct:    float = 0.40   # trailing po 40% TP
    trail_dist_pct:   float = 0.003  # trailing distance 0.3%
    max_hold_s:       float = 900.0  # max 15 min

    # ── Cooldown ─────────────────────────────────────────────────────────────
    cool_win_s:       float = 30.0
    cool_loss_s:      float = 120.0
    cool_streak_s:    float = 300.0  # dodatkowy cooldown po serii strat

    # ── Meta-Learning ────────────────────────────────────────────────────────
    maml_lr:          float = 0.01   # inner loop learning rate
    meta_gamma:       float = 0.99   # discount factor dla meta-learner

    # ── Performance (nie mutowane) ────────────────────────────────────────────
    n_trades:    int   = 0; n_wins:     int   = 0
    total_pnl:   float = 0.0; max_dd:  float = 0.0
    best_streak: int   = 0;  avg_hold_s:float = 0.0

    # ── CMA-ES state ─────────────────────────────────────────────────────────
    cma_sigma:   float = 0.08

    def wr(self) -> float: return self.n_wins / max(self.n_trades, 1)
    def avg_pnl(self) -> float: return self.total_pnl / max(self.n_trades, 1)
    def rr_ratio(self) -> float: return self.tp_pct / max(self.sl_pct, 1e-10)

    def fitness(self) -> float:
        """Multi-objective fitness: Sharpe + WR + trades + drawdown penalty."""
        if self.n_trades < 10: return 0.0
        wr = self.wr()
        wr_penalty = max(0, (TARGET_WIN_RATE - wr) * 25) ** 2
        ap = self.avg_pnl()
        sharpe = ap * math.sqrt(self.n_trades) * (wr / max(1 - wr, 0.01))
        dd_pen = self.max_dd * 10
        trade_bonus = min(self.n_trades / 500, 1.0) * 0.1
        f = (0.45 * max(sharpe, 0) +
             0.35 * wr +
             0.10 * trade_bonus +
             0.10 * (1 - dd_pen / 100)
             - wr_penalty)
        return float(np.clip(f, -20, 20))

    def normalize_weights(self):
        """Normalizuj wagi komponentów do sumy 1."""
        # Komponent weights
        tot1 = self.w_rl + self.w_neural + self.w_micro + self.w_regime
        if tot1 > 0:
            self.w_rl /= tot1; self.w_neural /= tot1
            self.w_micro /= tot1; self.w_regime /= tot1
        # Indicator weights
        iw = [self.w_rsi, self.w_macd, self.w_ema, self.w_kama, self.w_bb,
              self.w_adx, self.w_funding, self.w_oi, self.w_ob, self.w_cvd,
              self.w_toxic, self.w_hurst, self.w_regime_w]
        tot2 = sum(max(w, 0.01) for w in iw)
        scale = 1.0 / tot2
        self.w_rsi     *= scale; self.w_macd *= scale; self.w_ema  *= scale
        self.w_kama    *= scale; self.w_bb   *= scale; self.w_adx  *= scale
        self.w_funding *= scale; self.w_oi   *= scale; self.w_ob   *= scale
        self.w_cvd     *= scale; self.w_toxic*= scale; self.w_hurst*= scale
        self.w_regime_w*= scale

    _MUTABLE_FLOATS: ClassVar[List[str]] = [
        "w_rl","w_neural","w_micro","w_regime",
        "w_rsi","w_macd","w_ema","w_kama","w_bb","w_adx",
        "w_funding","w_oi","w_ob","w_cvd","w_toxic","w_hurst","w_regime_w",
        "rsi_ob","rsi_os","bb_std",
        "confidence_min","signal_threshold",
        "kelly_frac","sl_pct","tp_pct",
        "trail_arm_pct","trail_dist_pct","max_hold_s",
        "cool_win_s","cool_loss_s","maml_lr",
    ]

    _BOUNDS: ClassVar[Dict[str, Tuple[float,float]]] = {
        "w_rl":(0.1,0.7),"w_neural":(0.05,0.6),"w_micro":(0.02,0.4),"w_regime":(0.02,0.3),
        "rsi_ob":(60,90),"rsi_os":(10,40),"bb_std":(1.2,3.5),
        "confidence_min":(0.75,0.95),"signal_threshold":(0.20,0.65),
        "kelly_frac":(0.05,0.40),"sl_pct":(0.003,0.020),"tp_pct":(0.006,0.040),
        "trail_arm_pct":(0.10,0.80),"trail_dist_pct":(0.001,0.015),
        "max_hold_s":(120,3600),"cool_win_s":(5,120),"cool_loss_s":(30,600),
        "maml_lr":(0.001,0.05),
    }

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in self._MUTABLE_FLOATS], dtype=np.float64)

    def from_vector(self, v: np.ndarray):
        for i, fname in enumerate(self._MUTABLE_FLOATS):
            if i >= len(v): break
            val = float(v[i])
            if fname in self._BOUNDS:
                lo, hi = self._BOUNDS[fname]
                val = float(np.clip(val, lo, hi))
            setattr(self, fname, val)
        self.normalize_weights()

    def to_dict(self) -> Dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# ══════════════════════════════════════════════════════════════════════════════════════════
# STATE VECTOR — 80-wymiarowy wektor stanu (Bitget-specific)
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class StateVector:
    """
    80-wymiarowy zunifikowany wektor stanu.
    Wszystkie cechy specyficzne dla Bitget Futures.
    Każde pole unormowane do sensownego zakresu.
    """

    # ── Price dynamics [0-9] ─────────────────────────────────────────────────
    pc_1m:    float = 0.0;  pc_5m:     float = 0.0
    pc_15m:   float = 0.0;  pc_1h:     float = 0.0
    pc_4h:    float = 0.0;  pc_24h:    float = 0.0
    price_vs_vwap:  float = 0.0;  price_vs_ema20: float = 0.0
    high_24h_pct:   float = 0.0;  low_24h_pct:    float = 0.0

    # ── Volatility [10-16] ───────────────────────────────────────────────────
    vol_1m:   float = 0.0;  vol_5m:    float = 0.0
    vol_of_vol: float = 0.0; atr_pct:  float = 0.0
    atr_ratio:  float = 1.0; realized_vol: float = 0.0
    bb_width:   float = 0.0

    # ── Momentum/Oscillators [17-27] ─────────────────────────────────────────
    rsi_fast:    float = 0.5; rsi_14:  float = 0.5
    rsi_slow:    float = 0.5; rsi_slope: float = 0.0
    rsi_div:     float = 0.0  # RSI divergence
    macd_hist:   float = 0.0; macd_slope: float = 0.0
    macd_cross:  int   = 0
    stoch_k:     float = 0.5; stoch_d: float = 0.5
    wr_14:       float = 0.0  # Williams %R

    # ── Trend [28-36] ────────────────────────────────────────────────────────
    ema_8_21:  float = 0.0;  ema_21_89: float = 0.0
    kama_dev:  float = 0.0;  bb_pos:    float = 0.5
    bb_sq:     float = 0.0;  adx:       float = 0.25
    adx_slope: float = 0.0;  hurst:     float = 0.5
    autocorr:  float = 0.0

    # ── Volume [37-43] ───────────────────────────────────────────────────────
    vol_ratio: float = 1.0;  vol_trend:  float = 0.0
    obv_slope: float = 0.0;  vwap_dev:   float = 0.0
    vol_spike: float = 0.0;  cvd_1m:     float = 0.0
    cvd_slope: float = 0.0

    # ── Bitget Futures Microstructure [44-59] ────────────────────────────────
    funding:       float = 0.0;  funding_pred:  float = 0.0
    funding_arb:   float = 0.0   # |funding| > threshold
    oi_abs:        float = 0.0;  oi_chg_1h:     float = 0.0
    oi_chg_4h:     float = 0.0;  oi_price_div:  float = 0.0
    liq_long_prox: float = 0.0;  liq_short_prox:float = 0.0
    liq_cascade:   float = 0.0
    ob_imb:        float = 0.0;  ob_depth5:     float = 1.0
    ob_spread:     float = 0.001;ob_entropy:    float = 0.5
    vpin:          float = 0.3;  toxic_flow:    float = 0.0

    # ── Order Flow [60-66] ───────────────────────────────────────────────────
    taker_ratio:   float = 0.5;  taker_mom:     float = 0.0
    whale_delta:   float = 0.0;  whale_accel:   float = 0.0
    trade_cnt_norm:float = 0.0;  mkt_impact:    float = 0.0
    ob_large_imb:  float = 0.0

    # ── Regime [67-72] ──────────────────────────────────────────────────────
    regime_idx:   float = 0.5;  regime_conf:   float = 0.5
    poc_distance: float = 0.0;  trend_consist: float = 0.5
    swing_hh_hl:  float = 0.0;  swing_bos:     float = 0.0

    # ── Position/Portfolio [73-77] ───────────────────────────────────────────
    pos_side:     float = 0.0;  pos_pnl_pct:   float = 0.0
    pos_age_norm: float = 0.0;  portfolio_wr:  float = 0.5
    kelly_signal: float = 0.0

    # ── Swarm/Meta [78-79] ───────────────────────────────────────────────────
    swarm_signal: float = 0.0;  distill_signal:float = 0.0

    def to_array(self) -> np.ndarray:
        """Konwertuj do flat numpy array [80D]."""
        return np.array([
            # Price [0-9]
            np.tanh(self.pc_1m*50), np.tanh(self.pc_5m*20),
            np.tanh(self.pc_15m*10), np.tanh(self.pc_1h*5),
            np.tanh(self.pc_4h*2), np.tanh(self.pc_24h*1),
            np.tanh(self.price_vs_vwap*20), np.tanh(self.price_vs_ema20*20),
            np.clip(self.high_24h_pct*10, 0, 1), np.clip(self.low_24h_pct*10, 0, 1),
            # Vol [10-16]
            np.clip(abs(self.vol_1m)*50, 0, 1), np.clip(abs(self.vol_5m)*20, 0, 1),
            np.clip(self.vol_of_vol*10, 0, 1), np.clip(self.atr_pct*100, 0, 1),
            np.clip(self.atr_ratio-1, -1, 2), np.clip(self.realized_vol, 0, 1),
            np.clip(self.bb_width*20, 0, 1),
            # Momentum [17-27]
            self.rsi_fast*2-1, self.rsi_14*2-1, self.rsi_slow*2-1,
            np.tanh(self.rsi_slope*5), np.tanh(self.rsi_div*5),
            np.tanh(self.macd_hist*100), np.tanh(self.macd_slope*200),
            float(self.macd_cross),
            self.stoch_k/100*2-1, self.stoch_d/100*2-1,
            np.clip(self.wr_14/100+0.5, -1, 1),
            # Trend [28-36]
            np.tanh(self.ema_8_21*100), np.tanh(self.ema_21_89*100),
            np.tanh(self.kama_dev*50), self.bb_pos*2-1,
            float(self.bb_sq), np.clip(self.adx/50, 0, 1),
            np.tanh(self.adx_slope*5), self.hurst*2-1,
            np.tanh(self.autocorr*5),
            # Volume [37-43]
            np.clip(self.vol_ratio/3-1, -1, 2), np.tanh(self.vol_trend*5),
            np.tanh(self.obv_slope*100), np.tanh(self.vwap_dev*30),
            np.clip(self.vol_spike, 0, 3)/3, np.tanh(self.cvd_1m*0.001),
            np.tanh(self.cvd_slope*0.01),
            # Futures [44-59]
            np.tanh(self.funding*1000), np.tanh(self.funding_pred*1000),
            np.clip(self.funding_arb*300, 0, 1),
            np.clip(self.oi_abs, 0, 1), np.tanh(self.oi_chg_1h*5),
            np.tanh(self.oi_chg_4h*2), np.tanh(self.oi_price_div*5),
            np.clip(self.liq_long_prox, 0, 1), np.clip(self.liq_short_prox, 0, 1),
            np.clip(self.liq_cascade, 0, 1),
            np.tanh(self.ob_imb*3), np.clip(self.ob_depth5, 0, 3)/3,
            np.clip(self.ob_spread*200, 0, 1), self.ob_entropy,
            np.clip(self.vpin, 0, 1), np.clip(self.toxic_flow, 0, 1),
            # Order Flow [60-66]
            self.taker_ratio*2-1, np.tanh(self.taker_mom*10),
            np.tanh(self.whale_delta*3), np.tanh(self.whale_accel*5),
            np.clip(self.trade_cnt_norm, 0, 1), np.tanh(self.mkt_impact*200),
            np.tanh(self.ob_large_imb*3),
            # Regime [67-72]
            self.regime_idx*2-1, self.regime_conf*2-1,
            np.tanh(self.poc_distance*20), self.trend_consist*2-1,
            float(self.swing_hh_hl), float(self.swing_bos),
            # Position [73-77]
            np.clip(self.pos_side, -1, 1), np.tanh(self.pos_pnl_pct*10),
            np.clip(self.pos_age_norm, 0, 1), self.portfolio_wr*2-1,
            np.tanh(self.kelly_signal*5),
            # Swarm [78-79]
            np.tanh(self.swarm_signal*3), np.tanh(self.distill_signal*3),
        ], dtype=np.float32)

    def is_dead_market(self) -> bool:
        return self.vol_ratio < 0.3 and abs(self.pc_1h) < 0.001

    def manipulation_risk(self) -> float:
        """Agregowane ryzyko manipulacji [0,1]."""
        r = 0.0
        if self.toxic_flow > 0.6: r += 0.3
        if self.vpin > 0.7: r += 0.2
        if abs(self.funding) > 0.003: r += 0.2
        if self.liq_cascade > 0.6: r += 0.2
        if self.ob_entropy < 0.3: r += 0.1
        return float(np.clip(r, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════════════════
# BITGOT DATABASE — WAL SQLite dla 3000 botów
# ══════════════════════════════════════════════════════════════════════════════════════════

class BITGOTDatabase:
    """
    Główna baza danych systemu BITGOT.
    WAL journal mode + batch writes co 500ms.
    Obsługuje 3000 concurrent writers bez lock contention.
    """

    BATCH_INTERVAL = 0.5   # flush co 500ms

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=20.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=100000")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA wal_autocheckpoint=1000")
        self._lock = threading.Lock()
        # Batch queues
        self._trade_q:  List[BotTrade]  = []
        self._bot_q:    List[BotState]  = []
        self._sig_q:    List[TradingSignal] = []
        self._err_q:    List[OmegaError] = []
        self._heal_q:   List[HealAction] = []
        self._metric_q: List[Dict] = []
        self._genome_q: List[BotGenome] = []
        self._log = logging.getLogger("BITGOT·DB")
        self._init_schema()
        self._flush_thread = threading.Thread(target=self._batch_flush_loop, daemon=True)
        self._flush_thread.start()
        self._log = logging.getLogger("BITGOT·DB")

    def _init_schema(self):
        self._conn.executescript("""
        -- Boty
        CREATE TABLE IF NOT EXISTS bots (
            bot_id       INTEGER PRIMARY KEY,
            symbol       TEXT NOT NULL,
            market_type  TEXT,
            tier         TEXT,
            status       TEXT,
            portfolio    REAL DEFAULT 0,
            n_trades     INTEGER DEFAULT 0,
            n_wins       INTEGER DEFAULT 0,
            total_pnl    REAL DEFAULT 0,
            daily_pnl    REAL DEFAULT 0,
            win_rate     REAL DEFAULT 0.5,
            health_score REAL DEFAULT 100,
            genome_id    TEXT,
            regime       TEXT,
            promotions   INTEGER DEFAULT 0,
            demotions    INTEGER DEFAULT 0,
            last_updated TEXT
        );

        -- Transakcje
        CREATE TABLE IF NOT EXISTS trades (
            id           TEXT PRIMARY KEY,
            bot_id       INTEGER,
            symbol       TEXT,
            market_type  TEXT,
            side         TEXT,
            entry_price  REAL,
            exit_price   REAL,
            qty          REAL,
            notional     REAL,
            leverage     INTEGER,
            pnl          REAL,
            pnl_pct      REAL,
            roi_pct      REAL,
            fees         REAL,
            duration_ms  INTEGER,
            exit_reason  TEXT,
            signal_conf  REAL,
            regime       TEXT,
            tier         TEXT,
            portfolio_at REAL,
            entry_ts     INTEGER,
            exit_ts      INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_trades_bot ON trades(bot_id, entry_ts);
        CREATE INDEX IF NOT EXISTS idx_trades_ts  ON trades(entry_ts);
        CREATE INDEX IF NOT EXISTS idx_trades_sym ON trades(symbol);

        -- Sygnały
        CREATE TABLE IF NOT EXISTS signals (
            id           TEXT PRIMARY KEY,
            bot_id       INTEGER,
            symbol       TEXT,
            side         TEXT,
            confidence   REAL,
            raw_score    REAL,
            regime       TEXT,
            state        TEXT,
            leverage     INTEGER,
            notional     REAL,
            engines_voted INTEGER,
            n_agree      INTEGER,
            created_ts   REAL,
            executed_ts  REAL
        );
        CREATE INDEX IF NOT EXISTS idx_sig_ts  ON signals(created_ts);
        CREATE INDEX IF NOT EXISTS idx_sig_bot ON signals(bot_id);

        -- Błędy (Omega)
        CREATE TABLE IF NOT EXISTS omega_errors (
            id           TEXT PRIMARY KEY,
            ts           REAL,
            message      TEXT,
            category     TEXT,
            severity     TEXT,
            bot_id       INTEGER,
            module       TEXT,
            traceback    TEXT,
            resolved     INTEGER DEFAULT 0,
            fix_attempts INTEGER DEFAULT 0,
            resolved_by  TEXT
        );

        -- Akcje naprawcze
        CREATE TABLE IF NOT EXISTS heal_actions (
            id           TEXT PRIMARY KEY,
            ts           REAL,
            error_id     TEXT,
            strategy     TEXT,
            bot_id       INTEGER,
            result       TEXT,
            duration_s   REAL,
            details      TEXT,
            xp_earned    INTEGER DEFAULT 0
        );

        -- Wiedza o naprawach (self-learning)
        CREATE TABLE IF NOT EXISTS heal_knowledge (
            id           TEXT PRIMARY KEY,
            error_pattern TEXT,
            strategy     TEXT,
            success_rate REAL DEFAULT 0.5,
            n_attempts   INTEGER DEFAULT 0,
            last_used    REAL
        );

        -- Genomy
        CREATE TABLE IF NOT EXISTS genomes (
            gid          TEXT PRIMARY KEY,
            bot_id       INTEGER,
            generation   INTEGER,
            fitness      REAL,
            wr           REAL,
            n_trades     INTEGER,
            genome_json  TEXT,
            ts           REAL
        );

        -- Metryki globalne
        CREATE TABLE IF NOT EXISTS metrics (
            ts              INTEGER PRIMARY KEY,
            total_bots      INTEGER,
            active_positions INTEGER,
            trades_1min     INTEGER,
            pnl_1min        REAL,
            total_pnl       REAL,
            total_trades    INTEGER,
            global_wr       REAL,
            portfolio       REAL,
            tier_apex_wr    REAL,
            tier_elite_wr   REAL,
            tier_std_wr     REAL,
            tier_scout_wr   REAL,
            signals_1min    INTEGER,
            heals_1min      INTEGER,
            omega_level     INTEGER,
            avg_confidence  REAL
        );

        -- Portfel globalny
        CREATE TABLE IF NOT EXISTS portfolio_history (
            ts           INTEGER PRIMARY KEY,
            total_capital REAL,
            daily_pnl    REAL,
            drawdown_pct REAL,
            active_pos   INTEGER
        );

        -- XP / Rewards
        CREATE TABLE IF NOT EXISTS rewards (
            id           TEXT PRIMARY KEY,
            ts           REAL,
            event        TEXT,
            xp           INTEGER,
            total_xp     INTEGER,
            level        INTEGER,
            details      TEXT
        );
        """)
        self._conn.commit()
        self._log.debug("Schema initialized")

    def _batch_flush_loop(self):
        while True:
            time.sleep(self.BATCH_INTERVAL)
            try: self._flush()
            except Exception as e:
                if hasattr(self, '_log'): self._log.debug(f"Flush: {e}")

    def flush(self):
        self._flush()

    def _flush(self):
        with self._lock:
            bots   = self._bot_q[:];   self._bot_q.clear()
            trades = self._trade_q[:]; self._trade_q.clear()
            sigs   = self._sig_q[:];   self._sig_q.clear()
            errs   = self._err_q[:];   self._err_q.clear()
            heals  = self._heal_q[:];  self._heal_q.clear()
            mets   = self._metric_q[:];self._metric_q.clear()
            genomes= self._genome_q[:];self._genome_q.clear()

        if bots:
            self._conn.executemany(
                "INSERT OR REPLACE INTO bots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(b.bot_id, b.symbol, b.market_type.value, b.tier.value,
                  b.status.value, b.portfolio, b.n_trades, b.n_wins,
                  b.total_pnl, b.daily_pnl, b.win_rate, b.health_score,
                  b.genome_id, b.current_regime, b.promotions, b.demotions, _NOW())
                 for b in bots]
            )
        if trades:
            self._conn.executemany(
                "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(t.id, t.bot_id, t.symbol, t.market_type, t.side,
                  t.entry_price, t.exit_price, t.qty, t.notional, t.leverage,
                  t.pnl, t.pnl_pct, t.roi_pct, t.fees, t.duration_ms,
                  t.exit_reason, t.signal_conf, t.regime, t.tier,
                  t.portfolio_at, t.entry_ts, t.exit_ts) for t in trades]
            )
        if sigs:
            self._conn.executemany(
                "INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(s.id, s.bot_id, s.symbol, s.side, s.confidence, s.raw_score,
                  s.regime, s.state.value, s.leverage, s.notional,
                  s.engines_voted, s.n_agree, s.created_ts, s.executed_ts)
                 for s in sigs]
            )
        if errs:
            self._conn.executemany(
                "INSERT OR REPLACE INTO omega_errors VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(e.id, e.ts, e.message[:500], e.category.value, e.severity.name,
                  e.bot_id, e.module, e.tb[:1000], int(e.resolved),
                  e.fix_attempts, e.resolved_by) for e in errs]
            )
        if heals:
            self._conn.executemany(
                "INSERT OR REPLACE INTO heal_actions VALUES (?,?,?,?,?,?,?,?,?)",
                [(h.id, h.ts, h.error_id, h.strategy, h.bot_id, h.result.value,
                  h.duration_s, h.details[:300], h.xp_earned) for h in heals]
            )
        if mets:
            self._conn.executemany(
                "INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(int(m.get("ts",_TS())), m.get("total_bots",0),
                  m.get("active_positions",0), m.get("trades_1min",0),
                  m.get("pnl_1min",0.0), m.get("total_pnl",0.0),
                  m.get("total_trades",0), m.get("global_wr",0.5),
                  m.get("portfolio",0.0), m.get("tier_apex_wr",0.5),
                  m.get("tier_elite_wr",0.5), m.get("tier_std_wr",0.5),
                  m.get("tier_scout_wr",0.5), m.get("signals_1min",0),
                  m.get("heals_1min",0), m.get("omega_level",1),
                  m.get("avg_confidence",0.0)) for m in mets]
            )
        if genomes:
            self._conn.executemany(
                "INSERT OR REPLACE INTO genomes VALUES (?,?,?,?,?,?,?,?)",
                [(g.gid, g.bot_id, g.generation, g.fitness(), g.wr(),
                  g.n_trades, json.dumps(g.to_dict()), _TS()) for g in genomes]
            )
        if any([bots, trades, sigs, errs, heals, mets, genomes]):
            self._conn.commit()

    # ── Queue methods ──────────────────────────────────────────────────────────
    def q_bot(self, b: BotState):
        with self._lock: self._bot_q.append(b)
    def q_trade(self, t: BotTrade):
        with self._lock: self._trade_q.append(t)
    def q_signal(self, s: TradingSignal):
        with self._lock: self._sig_q.append(s)
    def q_error(self, e: OmegaError):
        with self._lock: self._err_q.append(e)
    def q_heal(self, h: HealAction):
        with self._lock: self._heal_q.append(h)
    def q_metric(self, m: Dict):
        with self._lock: self._metric_q.append(m)

    # ── Direct writes ──────────────────────────────────────────────────────────
    def save_genome(self, g: BotGenome):
        with self._lock:
            self._genome_q.append(g)

    def log_reward(self, event: str, xp: int, total: int, level: int, details: str = ""):
        with self._lock:
            self._conn.execute(
                "INSERT INTO rewards VALUES (?,?,?,?,?,?,?)",
                (uuid.uuid4().hex[:8], _TS(), event, xp, total, level, details)
            )
            self._conn.commit()

    def log_portfolio(self, gp: GlobalPortfolio):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO portfolio_history VALUES (?,?,?,?,?)",
                (int(_TS()), gp.total_capital, gp.daily_pnl,
                 gp.drawdown_pct, gp.active_positions)
            )
            self._conn.commit()

    # ── Read methods ───────────────────────────────────────────────────────────
    def get_best_strategy(self, pattern: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT strategy FROM heal_knowledge "
                "WHERE error_pattern LIKE ? AND n_attempts >= 3 "
                "ORDER BY success_rate DESC LIMIT 1",
                (f"%{pattern[:25]}%",)
            ).fetchone()
        return row[0] if row else None

    def update_heal_knowledge(self, pattern: str, strategy: str, success: bool):
        with self._lock:
            row = self._conn.execute(
                "SELECT id, n_attempts, success_rate FROM heal_knowledge "
                "WHERE error_pattern=? AND strategy=?",
                (pattern, strategy)
            ).fetchone()
            if row:
                n  = row[1] + 1
                sr = (row[2] * row[1] + float(success)) / n
                self._conn.execute(
                    "UPDATE heal_knowledge SET n_attempts=?, success_rate=?, last_used=? WHERE id=?",
                    (n, sr, _TS(), row[0])
                )
            else:
                self._conn.execute(
                    "INSERT INTO heal_knowledge VALUES (?,?,?,?,?,?)",
                    (uuid.uuid4().hex[:8], pattern[:60], strategy, float(success), 1, _TS())
                )
            self._conn.commit()

    def get_summary(self) -> Dict:
        with self._lock:
            bots  = self._conn.execute(
                "SELECT COUNT(*), SUM(n_trades), SUM(total_pnl), AVG(win_rate) FROM bots"
            ).fetchone()
            heals = self._conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN result='success' THEN 1 ELSE 0 END) FROM heal_actions"
            ).fetchone()
            rew   = self._conn.execute(
                "SELECT SUM(xp), MAX(level) FROM rewards"
            ).fetchone()
            port  = self._conn.execute(
                "SELECT total_capital FROM portfolio_history ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        return {
            "bots": bots[0] or 0, "trades": bots[1] or 0,
            "total_pnl": bots[2] or 0.0, "avg_wr": bots[3] or 0.5,
            "heals": heals[0] or 0, "heal_success": heals[1] or 0,
            "total_xp": rew[0] or 0, "level": rew[1] or 1,
            "portfolio": port[0] if port else 0.0,
        }

    def get_top_bots(self, n: int = 10) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT bot_id, symbol, tier, win_rate, total_pnl, n_trades "
                "FROM bots WHERE n_trades >= 20 "
                "ORDER BY win_rate DESC, total_pnl DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"bot_id":r[0],"symbol":r[1],"tier":r[2],
                  "wr":r[3],"pnl":r[4],"trades":r[5]} for r in rows]

    def get_recent_trades(self, hours: float = 1.0, limit: int = 100) -> List[Dict]:
        cutoff = int((_TS() - hours * 3600) * 1000)
        with self._lock:
            rows = self._conn.execute(
                "SELECT symbol, side, pnl, signal_conf, tier, exit_reason "
                "FROM trades WHERE entry_ts >= ? ORDER BY entry_ts DESC LIMIT ?",
                (cutoff, limit)
            ).fetchall()
        return [{"symbol":r[0],"side":r[1],"pnl":r[2],
                  "conf":r[3],"tier":r[4],"reason":r[5]} for r in rows]

    def close(self):
        try:
            self._flush()
            self._conn.close()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════════════════
# ADAM OPTIMIZER — for neural networks (pure NumPy)
# ══════════════════════════════════════════════════════════════════════════════════════════

class AdamOptimizer:
    """Adam optimizer dla wektorów parametrów NumPy."""
    __slots__ = ("lr","b1","b2","eps","m","v","t")

    def __init__(self, dim: int, lr: float = LR_FAST,
                  b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(dim, dtype=float)
        self.v = np.zeros(dim, dtype=float)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * mh / (np.sqrt(vh) + self.eps)


class SGDMomentum:
    """SGD with momentum — lighter alternative for meta-learning."""
    __slots__ = ("lr","mom","velocity")

    def __init__(self, dim: int, lr: float = LR_META, momentum: float = 0.9):
        self.lr, self.mom = lr, momentum
        self.velocity = np.zeros(dim, dtype=float)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.velocity = self.mom * self.velocity - self.lr * grad
        return params + self.velocity


class NumpyMLP:
    """
    3-warstwowy MLP z Adam, dropout, layer-norm.
    Podstawa wszystkich sieci neuronowych BITGOT.
    """

    def __init__(self, in_d: int, h1: int, h2: int, out_d: int,
                  lr: float = LR_FAST, dropout: float = 0.08,
                  name: str = "MLP"):
        self.name     = name
        self.dropout  = dropout
        # Xavier init
        s1 = math.sqrt(2.0 / in_d); s2 = math.sqrt(2.0 / h1); s3 = math.sqrt(2.0 / h2)
        self.W1 = np.random.randn(in_d, h1) * s1; self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1,   h2) * s2; self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, out_d) * s3; self.b3 = np.zeros(out_d)
        # Adam
        dim = self._flat().shape[0]
        self.opt = AdamOptimizer(dim, lr=lr)
        # Cache for backprop
        self._a0 = self._a1 = self._a2 = None
        self._m1 = self._m2 = None

    def _flat(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1, self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3
        ])

    def _unflat(self, f: np.ndarray):
        idx = 0
        def ex(s):
            nonlocal idx
            n = int(np.prod(s)); a = f[idx:idx+n].reshape(s); idx += n; return a
        self.W1 = ex(self.W1.shape); self.b1 = ex(self.b1.shape)
        self.W2 = ex(self.W2.shape); self.b2 = ex(self.b2.shape)
        self.W3 = ex(self.W3.shape); self.b3 = ex(self.b3.shape)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._a0 = x.copy()
        # Layer 1
        z1 = x @ self.W1 + self.b1
        # Layer Norm
        mu1, std1 = z1.mean(), z1.std() + 1e-8
        z1 = (z1 - mu1) / std1
        a1 = MathCore.relu(z1)
        if training and self.dropout > 0:
            m1 = (np.random.rand(*a1.shape) > self.dropout).astype(float)
            a1 *= m1 / (1 - self.dropout + 1e-8)
        else: m1 = np.ones_like(a1)
        self._a1, self._m1 = a1, m1
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        mu2, std2 = z2.mean(), z2.std() + 1e-8
        z2 = (z2 - mu2) / std2
        a2 = MathCore.relu(z2)
        if training and self.dropout > 0:
            m2 = (np.random.rand(*a2.shape) > self.dropout).astype(float)
            a2 *= m2 / (1 - self.dropout + 1e-8)
        else: m2 = np.ones_like(a2)
        self._a2, self._m2 = a2, m2
        return a2 @ self.W3 + self.b3

    def backward(self, grad_out: np.ndarray, clip: float = 1.0):
        """Backprop + Adam update."""
        gW3 = np.outer(self._a2, grad_out); gb3 = grad_out
        da2 = (grad_out @ self.W3.T) * (self._a2 > 0) * self._m2
        gW2 = np.outer(self._a1, da2); gb2 = da2
        da1 = (da2 @ self.W2.T) * (self._a1 > 0) * self._m1
        gW1 = np.outer(self._a0, da1); gb1 = da1
        flat_g = np.concatenate([g.ravel() for g in [gW1, gb1, gW2, gb2, gW3, gb3]])
        # Gradient clipping
        norm = float(np.linalg.norm(flat_g))
        if norm > clip: flat_g *= clip / (norm + 1e-8)
        fp = self._flat()
        fp = self.opt.step(fp, flat_g)
        self._unflat(fp)

    def get_params(self) -> np.ndarray: return self._flat().copy()
    def set_params(self, p: np.ndarray): self._unflat(p)

    def save(self, path: str):
        np.savez_compressed(path, W1=self.W1, b1=self.b1,
                             W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def load(self, path: str):
        p = Path(path + ".npz")
        if p.exists():
            d = np.load(str(p))
            self.W1, self.b1 = d["W1"], d["b1"]
            self.W2, self.b2 = d["W2"], d["b2"]
            self.W3, self.b3 = d["W3"], d["b3"]


class PrioritizedReplayBuffer:
    """PER — Prioritized Experience Replay."""

    def __init__(self, capacity: int = BUFFER_CAP, alpha: float = 0.6,
                  beta_start: float = 0.4):
        self.cap    = capacity; self.alpha = alpha
        self.beta   = beta_start; self.beta_step = (1.0 - beta_start) / 200_000
        self.buf:   deque = deque(maxlen=capacity)
        self.prios: deque = deque(maxlen=capacity)
        self._max_p = 1.0

    def push(self, state, action, reward, next_state, done, p: float = None):
        pr = (p or self._max_p) ** self.alpha
        self.buf.append((state, action, reward, next_state, done))
        self.prios.append(pr); self._max_p = max(self._max_p, pr)

    def sample(self, n: int) -> List:
        if len(self.buf) < n: return list(self.buf)
        probs = np.array(list(self.prios), dtype=float)
        probs /= probs.sum() + 1e-12
        idxs = np.random.choice(len(self.buf), n, replace=False, p=probs)
        items = list(self.buf)
        self.beta = min(1.0, self.beta + self.beta_step)
        return [items[i] for i in idxs]

    def __len__(self): return len(self.buf)


# ══════════════════════════════════════════════════════════════════════════════════════════
# GLOBAL PORTFOLIO — singleton stanu portfela
# ══════════════════════════════════════════════════════════════════════════════════════════

class GlobalPortfolioManager:
    """
    Thread-safe manager globalnego portfela.
    Każda transakcja natychmiast aktualizuje kapitał.
    """

    def __init__(self, start_capital: float = 3_000.0):
        self._gp = GlobalPortfolio(
            total_capital  = start_capital,
            peak_capital   = start_capital,
            day_start_cap  = start_capital,
            available      = start_capital,
        )
        self._lock = threading.Lock()
        self._log  = logging.getLogger("BITGOT·Portfolio")

    @property
    def total_capital(self) -> float:
        with self._lock: return self._gp.total_capital

    @property
    def snapshot(self) -> GlobalPortfolio:
        with self._lock: return copy.copy(self._gp)

    def update_from_trade(self, pnl: float, won: bool):
        with self._lock:
            self._gp.total_pnl    += pnl
            self._gp.daily_pnl    += pnl
            self._gp.total_capital += pnl
            self._gp.total_capital = max(self._gp.total_capital, 0.0)
            self._gp.total_trades += 1
            if won: self._gp.total_wins += 1
            self._gp.update_peak()

    def open_position(self, margin: float):
        with self._lock:
            self._gp.available      = max(0.0, self._gp.available - margin)
            self._gp.in_positions  += margin
            self._gp.active_positions += 1

    def close_position(self, margin: float, pnl: float):
        with self._lock:
            self._gp.in_positions   = max(0.0, self._gp.in_positions - margin)
            self._gp.available      += margin + pnl
            self._gp.active_positions = max(0, self._gp.active_positions - 1)

    def check_halt(self) -> Tuple[bool, str]:
        """True = halt trading."""
        gp = self._gp
        if gp.daily_loss_pct >= CFG.max_daily_loss_pct:
            return True, f"Daily loss {gp.daily_loss_pct:.1%} >= {CFG.max_daily_loss_pct:.1%}"
        if gp.drawdown_pct >= CFG.max_drawdown_pct:
            return True, f"Drawdown {gp.drawdown_pct:.1%} >= {CFG.max_drawdown_pct:.1%}"
        return False, ""

    def reset_daily(self):
        with self._lock:
            self._gp.day_start_cap = self._gp.total_capital
            self._gp.daily_pnl = 0.0
            self._gp.day_start_ts = _TS()


# ══════════════════════════════════════════════════════════════════════════════════════════
# MARKET DATA CACHE — real-time cache danych rynkowych
# ══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class MarketSnapshot:
    """Snapshot rynku dla jednej pary — aktualizowany co tick."""
    symbol:       str
    price:        float = 0.0
    bid:          float = 0.0
    ask:          float = 0.0
    spread_pct:   float = 0.0
    bid_vol:      float = 0.0
    ask_vol:      float = 0.0
    trade_side:   str   = ""     # "buy" / "sell"
    funding:      float = 0.0
    oi:           float = 0.0
    ts:           float = field(default_factory=_TS)
    # OHLCV buffers
    close_1m:  deque = field(default_factory=lambda: deque(maxlen=200))
    high_1m:   deque = field(default_factory=lambda: deque(maxlen=200))
    low_1m:    deque = field(default_factory=lambda: deque(maxlen=200))
    vol_1m:    deque = field(default_factory=lambda: deque(maxlen=200))
    close_5m:  deque = field(default_factory=lambda: deque(maxlen=200))
    high_5m:   deque = field(default_factory=lambda: deque(maxlen=200))
    low_5m:    deque = field(default_factory=lambda: deque(maxlen=200))
    vol_5m:    deque = field(default_factory=lambda: deque(maxlen=200))
    close_15m: deque = field(default_factory=lambda: deque(maxlen=200))
    high_15m:  deque = field(default_factory=lambda: deque(maxlen=200))
    low_15m:   deque = field(default_factory=lambda: deque(maxlen=200))
    close_1h:  deque = field(default_factory=lambda: deque(maxlen=300))
    high_1h:   deque = field(default_factory=lambda: deque(maxlen=300))
    low_1h:    deque = field(default_factory=lambda: deque(maxlen=300))
    vol_1h:    deque = field(default_factory=lambda: deque(maxlen=300))
    close_4h:  deque = field(default_factory=lambda: deque(maxlen=200))
    # Order flow
    buy_vol_hist:  deque = field(default_factory=lambda: deque(maxlen=100))
    sell_vol_hist: deque = field(default_factory=lambda: deque(maxlen=100))
    ob_bids:       List  = field(default_factory=list)   # [[price, qty], ...]
    ob_asks:       List  = field(default_factory=list)
    # Funding history
    funding_hist:  deque = field(default_factory=lambda: deque(maxlen=50))
    oi_hist:       deque = field(default_factory=lambda: deque(maxlen=50))

    def push_ohlcv(self, tf: str, o, h, l, c, v):
        tf_map = {
            "1m":  (self.close_1m,  self.high_1m,  self.low_1m,  self.vol_1m),
            "5m":  (self.close_5m,  self.high_5m,  self.low_5m,  self.vol_5m),
            "15m": (self.close_15m, self.high_15m, self.low_15m, None),
            "1h":  (self.close_1h,  self.high_1h,  self.low_1h,  self.vol_1h),
            "4h":  (self.close_4h,  None,           None,          None),
        }
        bufs = tf_map.get(tf)
        if not bufs: return
        bufs[0].append(float(c))
        if bufs[1] is not None: bufs[1].append(float(h))
        if bufs[2] is not None: bufs[2].append(float(l))
        if bufs[3] is not None: bufs[3].append(float(v))

    def arr(self, buf: deque) -> np.ndarray:
        return np.array(list(buf), dtype=float)

    def age_s(self) -> float: return _TS() - self.ts

    @property
    def is_fresh(self) -> bool: return self.age_s() < 5.0


class MarketDataCache:
    """Thread-safe cache danych rynkowych dla 3000 symboli."""

    def __init__(self):
        self._data: Dict[str, MarketSnapshot] = {}
        self._lock = threading.RLock()
        self._btc_price: float = 0.0
        self._eth_price: float = 0.0

    def get(self, symbol: str) -> Optional[MarketSnapshot]:
        with self._lock: return self._data.get(symbol)

    def get_or_create(self, symbol: str) -> MarketSnapshot:
        with self._lock:
            if symbol not in self._data:
                self._data[symbol] = MarketSnapshot(symbol=symbol)
            return self._data[symbol]

    def update_tick(self, symbol: str, price: float, bid: float, ask: float,
                     bid_vol: float = 0, ask_vol: float = 0, trade_side: str = ""):
        snap = self.get_or_create(symbol)
        with self._lock:
            snap.price = price; snap.bid = bid; snap.ask = ask
            spread = (ask - bid) / ((ask + bid) / 2 + 1e-12) * 100
            snap.spread_pct = spread; snap.bid_vol = bid_vol; snap.ask_vol = ask_vol
            snap.trade_side = trade_side; snap.ts = _TS()
            if trade_side == "buy":  snap.buy_vol_hist.append(bid_vol)
            elif trade_side == "sell": snap.sell_vol_hist.append(ask_vol)
        # Track BTC/ETH
        if "BTC" in symbol and "/USDT" in symbol: self._btc_price = price
        if "ETH" in symbol and "/USDT" in symbol: self._eth_price = price

    def update_ohlcv(self, symbol: str, tf: str, candles: List):
        snap = self.get_or_create(symbol)
        with self._lock:
            for c in candles:
                snap.push_ohlcv(tf, c[1], c[2], c[3], c[4], c[5])

    def update_ob(self, symbol: str, bids: List, asks: List):
        snap = self.get_or_create(symbol)
        with self._lock:
            snap.ob_bids = bids[:20]
            snap.ob_asks = asks[:20]

    def update_funding(self, symbol: str, funding: float, oi: float):
        snap = self.get_or_create(symbol)
        with self._lock:
            snap.funding = funding; snap.oi = oi
            snap.funding_hist.append(funding)
            snap.oi_hist.append(oi)

    @property
    def btc_price(self) -> float: return self._btc_price

    @property
    def eth_price(self) -> float: return self._eth_price

    def symbols(self) -> List[str]:
        with self._lock: return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock: return len(self._data)


# ══════════════════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER — kompiluje StateVector(80D) z MarketSnapshot
# ══════════════════════════════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    Buduje StateVector(80D) z surowych danych rynkowych.
    Wywołany przez każdego zwiadowcę co tick.
    Zoptymalizowany dla szybkości: <2ms per call.
    """

    def __init__(self, genome: BotGenome, pair: PairInfo,
                  mdc: MarketDataCache, portfolio: GlobalPortfolioManager):
        self.genome    = genome
        self.pair      = pair
        self.mdc       = mdc
        self.portfolio = portfolio
        self._regime_hist: deque = deque(maxlen=20)
        self._buy_vol_h: deque = deque(maxlen=30)
        self._sell_vol_h:deque = deque(maxlen=30)
        self._funding_h: deque = deque(maxlen=30)
        self._oi_h:      deque = deque(maxlen=30)

    def build(self, snap: MarketSnapshot,
               pos_side: str = "", pos_pnl: float = 0.0,
               pos_age_s: float = 0.0,
               swarm_signal: float = 0.0,
               distill_signal: float = 0.0) -> Optional[StateVector]:
        """
        Kompiluj StateVector. Zwraca None jeśli za mało danych.
        """
        if not snap or not snap.is_fresh: return None
        c1h = snap.arr(snap.close_1h)
        if len(c1h) < 30: return None   # za mało danych

        price = snap.price; bid = snap.bid; ask = snap.ask
        c1m = snap.arr(snap.close_1m); h1m = snap.arr(snap.high_1m)
        l1m = snap.arr(snap.low_1m);   v1m = snap.arr(snap.vol_1m)
        c5m = snap.arr(snap.close_5m); h5m = snap.arr(snap.high_5m)
        l5m = snap.arr(snap.low_5m);   v5m = snap.arr(snap.vol_5m)
        h1h = snap.arr(snap.high_1h);  l1h = snap.arr(snap.low_1h); v1h = snap.arr(snap.vol_1h)
        c4h = snap.arr(snap.close_4h)

        # ── Price changes ────────────────────────────────────────────────────
        def pc(arr, n): return float(arr[-1]/max(arr[-n],1e-12)-1) if len(arr)>=n else 0.0
        pc_1m   = pc(c1m,  2); pc_5m   = pc(c5m, 2)
        pc_15m  = pc(snap.arr(snap.close_15m), 2)
        pc_1h   = pc(c1h,  2); pc_4h   = pc(c1h, 5)
        pc_24h  = pc(c1h, 25)

        h24 = float(h1h[-24:].max()) if len(h1h)>=24 else price
        l24 = float(l1h[-24:].min()) if len(l1h)>=24 else price

        # ── Volatility ───────────────────────────────────────────────────────
        atr_v  = MathCore.atr_scalar(h1h, l1h, c1h, 14)
        atr_ma = MathCore.atr_scalar(h1h, l1h, c1h, 50) if len(h1h)>=50 else atr_v
        atr_r  = atr_v / max(atr_ma, 1e-12)
        rv     = MathCore.realized_vol(c1h)

        vol_ma = float(v1h[-20:].mean()) if len(v1h)>=20 else float(v1h.mean()) if len(v1h) else 1.0
        vol_r  = float(v1h[-1]/max(vol_ma,1e-12)) if len(v1h)>0 else 1.0
        vol_trend = float(v1h[-1]-v1h[-2])/max(v1h[-2],1e-12) if len(v1h)>=2 else 0.0

        # ── RSI ──────────────────────────────────────────────────────────────
        g = self.genome
        rsi_fast  = MathCore.rsi_scalar(list(c1m),  g.rsi_fast)
        rsi_14    = MathCore.rsi_scalar(list(c1h),  g.rsi_period)
        rsi_slow  = MathCore.rsi_scalar(list(c1h),  21)
        rsi_slope = MathCore.rsi_slope(list(c1h), g.rsi_period, 5)
        rsi_div   = rsi_fast - rsi_14

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_h, macd_slope, macd_cross = MathCore.macd_scalar(list(c1h))

        # ── Bollinger + KAMA ─────────────────────────────────────────────────
        bb_pos, bb_sq, bb_width = MathCore.bb_scalar(list(c1h), g.bb_period, g.bb_std)
        kama_arr = MathCore.kama(c1h, g.kama_period)
        kama_dev = float((price - kama_arr[-1]) / max(kama_arr[-1], 1e-12))

        # ── EMA crosses ──────────────────────────────────────────────────────
        ef8  = MathCore.ema(c1h, g.ema_fast)
        es21 = MathCore.ema(c1h, g.ema_slow)
        et89 = MathCore.ema(c1h, g.ema_trend)
        ema_8_21  = float((ef8[-1]-es21[-1])/(price+1e-12)) if not np.isnan(ef8[-1]) else 0.0
        ema_21_89 = float((es21[-1]-et89[-1])/(price+1e-12)) if not np.isnan(es21[-1]) else 0.0

        # ── ADX + Hurst ──────────────────────────────────────────────────────
        adx_v, adx_slope = MathCore.adx_scalar(h1h, l1h, c1h, g.adx_period)
        hurst_v = MathCore.hurst(c1h)
        ac_v    = MathCore.autocorr(c1h, 1)

        # ── Stoch RSI + Williams %R ───────────────────────────────────────────
        sk_arr, sd_arr = MathCore.stoch_rsi(c1h, g.rsi_period)
        stoch_k = float(sk_arr[-1]) / 100.0 if not np.isnan(sk_arr[-1]) else 0.5
        stoch_d = float(sd_arr[-1]) / 100.0 if not np.isnan(sd_arr[-1]) else 0.5
        wr_arr  = MathCore.williams_r(h1h, l1h, c1h, 14)
        wr_v    = float(wr_arr[-1]) if not np.isnan(wr_arr[-1]) else 0.0

        # ── OBV ──────────────────────────────────────────────────────────────
        obv_arr = MathCore.obv(c1h, v1h) if len(v1h)==len(c1h) and len(c1h)>1 else np.zeros(1)
        obv_slope = float((obv_arr[-1]-obv_arr[-5])/(abs(obv_arr[-5])+1e-12)) if len(obv_arr)>=5 else 0.0

        # ── VWAP ─────────────────────────────────────────────────────────────
        vwap_v = MathCore.vwap_scalar(list(h1h), list(l1h), list(c1h), list(v1h), 20)
        vwap_dev = (price - vwap_v) / max(vwap_v, 1e-12)

        # ── Ichimoku ─────────────────────────────────────────────────────────
        # (used for swing structure indicator)
        swing = MathCore.swing_structure(h1h, l1h) if len(h1h)>=15 else {}

        # ── Point of Control ─────────────────────────────────────────────────
        poc_p = MathCore.poc(c1h, v1h) if len(c1h)>5 else price
        poc_dist = (price - poc_p) / max(price, 1e-12)

        # ── Funding + OI ─────────────────────────────────────────────────────
        funding = snap.funding
        self._funding_h.append(funding)
        fund_pred = float(np.mean(list(self._funding_h)[-3:])) if len(self._funding_h)>=3 else funding
        fund_arb  = float(abs(funding) > 0.001)
        oi = snap.oi
        self._oi_h.append(oi)
        oi_norm = float(oi / max(np.mean(list(self._oi_h)) + 1e-12, 1)) - 1 if len(self._oi_h)>1 else 0.0
        oi_chg_1h = float((self._oi_h[-1]-self._oi_h[-2])/max(abs(self._oi_h[-2]),1e-12)) if len(self._oi_h)>=2 else 0.0
        oi_chg_4h = float((self._oi_h[-1]-self._oi_h[-5])/max(abs(self._oi_h[-5]),1e-12)) if len(self._oi_h)>=5 else 0.0
        oi_price_div = oi_chg_1h - pc_1h

        # ── Liquidation proxies ───────────────────────────────────────────────
        liq_long_prox  = float(np.clip(1-rsi_14, 0, 1)) * float(abs(funding)>0.0005)
        liq_short_prox = float(np.clip(rsi_14,   0, 1)) * float(abs(funding)>0.0005)
        liq_cascade    = float(np.clip(abs(oi_chg_1h)*3 + abs(oi_price_div)*2, 0, 1))

        # ── Order Book ────────────────────────────────────────────────────────
        bids = snap.ob_bids; asks = snap.ob_asks
        bid_v = sum(b[0]*b[1] for b in bids[:10]) if bids else 0.0
        ask_v = sum(a[0]*a[1] for a in asks[:10]) if asks else 0.0
        ob_imb   = (bid_v-ask_v)/(bid_v+ask_v+1e-12)
        ob_d5_b  = sum(b[1] for b in bids[:5]) if bids else 0.0
        ob_d5_a  = sum(a[1] for a in asks[:5]) if asks else 0.0
        ob_depth5= ob_d5_b / max(ob_d5_a, 1e-12)
        ob_spread= (asks[0][0]-bids[0][0])/(bids[0][0]+1e-12) if bids and asks else 0.001
        ob_ent   = MathCore.ob_entropy(bids, asks) if bids and asks else 0.5
        ob_large = float(np.tanh((bid_v-ask_v)/(bid_v+ask_v+1e-12)*3))

        # ── VPIN ─────────────────────────────────────────────────────────────
        vpin_v = MathCore.vpin(list(snap.buy_vol_hist), list(snap.sell_vol_hist))

        # ── Taker ratio / CVD ─────────────────────────────────────────────────
        bvh = list(snap.buy_vol_hist); svh = list(snap.sell_vol_hist)
        tot_bv = sum(bvh[-20:]) if len(bvh)>=20 else sum(bvh)
        tot_sv = sum(svh[-20:]) if len(svh)>=20 else sum(svh)
        taker_r = tot_bv / max(tot_bv + tot_sv, 1e-12)
        taker_mom = 0.0
        if len(bvh)>=10:
            taker_r_prev = sum(bvh[-10:-5]) / max(sum(bvh[-10:-5])+sum(svh[-10:-5]),1e-12)
            taker_mom = taker_r - taker_r_prev

        cvd_1m = float(tot_bv - tot_sv)
        cvd_slope = 0.0
        if len(bvh)>=6:
            c_new = sum(bvh[-3:]) - sum(svh[-3:])
            c_old = sum(bvh[-6:-3]) - sum(svh[-6:-3])
            cvd_slope = c_new - c_old

        # ── Toxic flow + Market Impact ────────────────────────────────────────
        toxic = 0.0
        if len(bvh)>=5:
            d = [bvh[i]-svh[i] for i in range(-min(10,len(bvh)),0) if i < len(svh)]
            if d: toxic = float(np.clip(np.std(d)/(abs(np.mean(d))+1e-12)/10, 0, 1))

        # ── Whale detection ───────────────────────────────────────────────────
        whale_delta = 0.0; whale_accel = 0.0
        if len(bvh)>=5:
            all_vols = [max(b,s) for b,s in zip(bvh,svh)]
            thresh = np.percentile(all_vols, 90) if len(all_vols)>=5 else max(all_vols,default=1)
            big_b = sum(b for b in bvh[-10:] if b>thresh)
            big_s = sum(s for s in svh[-10:] if s>thresh)
            whale_delta = (big_b-big_s)/(big_b+big_s+1e-12)
            if len(bvh)>=10:
                old_wb = sum(b for b in bvh[-20:-10] if b>thresh)
                old_ws = sum(s for s in svh[-20:-10] if s>thresh)
                old_wd = (old_wb-old_ws)/(old_wb+old_ws+1e-12)
                whale_accel = whale_delta - old_wd

        # ── Regime (simplified) ───────────────────────────────────────────────
        regime_idx = 0.5  # default ranging
        regime_conf = 0.5
        if adx_v > 25 and hurst_v > 0.55:
            if ema_21_89 > 0: regime_idx = 0.8; regime_conf = min(0.5+(adx_v-25)/50, 0.9)
            else: regime_idx = 0.2; regime_conf = min(0.5+(adx_v-25)/50, 0.9)
        elif adx_v < 20 and hurst_v < 0.45:
            regime_idx = 0.5; regime_conf = min(0.5+(20-adx_v)/20, 0.85)

        # ── Trend consistency ─────────────────────────────────────────────────
        tc = 0.5
        if len(c1h)>=10:
            dir_exp = 1 if regime_idx > 0.55 else -1
            matches = sum(1 for i in range(-10,-1)
                          if np.sign(float(c1h[i])-float(c1h[i-1]))==dir_exp)
            tc = matches / 9.0

        # ── Portfolio state ───────────────────────────────────────────────────
        gp = self.portfolio.snapshot
        portfolio_wr = gp.global_wr
        kelly_s = MathCore.kelly(portfolio_wr, g.tp_pct, g.sl_pct)
        pos_side_enc = 1.0 if pos_side=="long" else -1.0 if pos_side=="short" else 0.0
        pos_age_norm = float(np.clip(pos_age_s / max(g.max_hold_s, 1.0), 0, 1))

        return StateVector(
            # Price
            pc_1m=pc_1m, pc_5m=pc_5m, pc_15m=pc_15m, pc_1h=pc_1h,
            pc_4h=pc_4h, pc_24h=pc_24h,
            price_vs_vwap=vwap_dev, price_vs_ema20=ema_8_21,
            high_24h_pct=(h24-price)/max(price,1e-12),
            low_24h_pct=(price-l24)/max(price,1e-12),
            # Vol
            vol_1m=abs(pc_1m), vol_5m=abs(pc_5m),
            vol_of_vol=float(np.std([abs(pc_1m),abs(pc_5m),abs(pc_15m)])),
            atr_pct=atr_v/max(price,1e-12), atr_ratio=float(atr_r),
            realized_vol=float(np.clip(rv, 0, 3)),
            bb_width=float(bb_width),
            # Momentum
            rsi_fast=float(rsi_fast), rsi_14=float(rsi_14), rsi_slow=float(rsi_slow),
            rsi_slope=float(rsi_slope), rsi_div=float(rsi_div),
            macd_hist=float(macd_h), macd_slope=float(macd_slope),
            macd_cross=int(macd_cross),
            stoch_k=float(stoch_k), stoch_d=float(stoch_d), wr_14=float(wr_v),
            # Trend
            ema_8_21=float(ema_8_21), ema_21_89=float(ema_21_89),
            kama_dev=float(kama_dev), bb_pos=float(bb_pos), bb_sq=float(bb_sq),
            adx=float(adx_v/100), adx_slope=float(adx_slope),
            hurst=float(hurst_v), autocorr=float(ac_v),
            # Volume
            vol_ratio=float(vol_r), vol_trend=float(vol_trend),
            obv_slope=float(obv_slope), vwap_dev=float(vwap_dev),
            vol_spike=float(np.clip(vol_r-1.5, 0, 5)),
            cvd_1m=float(cvd_1m), cvd_slope=float(cvd_slope),
            # Futures
            funding=float(funding), funding_pred=float(fund_pred),
            funding_arb=float(fund_arb),
            oi_abs=float(np.clip(oi_norm, -1, 5)),
            oi_chg_1h=float(oi_chg_1h), oi_chg_4h=float(oi_chg_4h),
            oi_price_div=float(oi_price_div),
            liq_long_prox=float(liq_long_prox),
            liq_short_prox=float(liq_short_prox),
            liq_cascade=float(liq_cascade),
            ob_imb=float(ob_imb), ob_depth5=float(ob_depth5),
            ob_spread=float(ob_spread), ob_entropy=float(ob_ent),
            vpin=float(vpin_v), toxic_flow=float(toxic),
            # Order Flow
            taker_ratio=float(taker_r), taker_mom=float(taker_mom),
            whale_delta=float(whale_delta), whale_accel=float(whale_accel),
            trade_cnt_norm=float(np.clip(len(bvh)/100, 0, 1)),
            mkt_impact=0.0, ob_large_imb=float(ob_large),
            # Regime
            regime_idx=float(regime_idx), regime_conf=float(regime_conf),
            poc_distance=float(poc_dist), trend_consist=float(tc),
            swing_hh_hl=float(swing.get("hh_hl", False)),
            swing_bos=float(swing.get("bos_up", swing.get("bos_dn", False))),
            # Position
            pos_side=float(pos_side_enc), pos_pnl_pct=float(pos_pnl),
            pos_age_norm=float(pos_age_norm),
            portfolio_wr=float(portfolio_wr), kelly_signal=float(kelly_s),
            # Swarm
            swarm_signal=float(swarm_signal), distill_signal=float(distill_signal),
        )


# ══════════════════════════════════════════════════════════════════════════════════════════
# REGIME ORACLE — 12-stanowy detektor reżimu rynkowego
# ══════════════════════════════════════════════════════════════════════════════════════════

class RegimeOracle:
    """
    12-stanowy Bayesian detektor reżimu rynkowego.
    Każdy bot ma własną instancję — dostosowuje do swojego symbolu.
    
    Stany:
    1. STEALTH_ACCUMULATION — cicha akumulacja, niska zmienność, rosnący OI
    2. MARKUP              — trend wzrostowy, malejący spread, rosnący wolumen
    3. DISTRIBUTION        — dystrybucja, wysoka zmienność, funding wysoki
    4. MARKDOWN            — trend spadkowy, kapitulacja
    5. RANGING_TIGHT       — ograniczony range, BB squeeze
    6. RANGING_WIDE        — szeroki range, wysokie ATR
    7. PARABOLIC           — eksplozja ceny, vol >5×
    8. CAPITULATION        — flash crash, ekstremalne wyprzedanie
    9. LIQUIDITY_HUNT      — polowanie na SL, fakeout patterns
    10. FLASH_CRASH        — natychmiastowy -5%+, vol spike
    11. FLASH_PUMP         — natychmiastowy +5%+
    12. MANIPULATED        — wysokie VPIN, toxic flow, spoofing sygnały
    """

    SMOOTH = 0.12   # Bayesian posterior smoothing
    N      = 12

    def __init__(self):
        self._probs = np.ones(self.N) / self.N
        self._history: deque = deque(maxlen=50)
        self._regime_names = [r.value for r in Regime]

    def detect(self, sv: StateVector) -> Tuple[Regime, float, np.ndarray]:
        """
        Zwraca (Regime, confidence, all_probs).
        Używa Bayesian evidence accumulation.
        """
        ev = np.zeros(self.N)

        # Evidence per regime
        # 0: STEALTH_ACCUMULATION
        if sv.adx < 0.3 and sv.oi_chg_1h > 0.01 and abs(sv.pc_1h) < 0.005:
            ev[0] += 2.0 + sv.oi_chg_1h * 10

        # 1: MARKUP
        if (sv.adx > 0.4 and sv.ema_21_89 > 0 and sv.vol_ratio > 1.2
                and sv.hurst > 0.55):
            ev[1] += (sv.adx * 2 + sv.hurst + sv.ema_21_89 * 20)

        # 2: DISTRIBUTION
        if (sv.funding > 0.0 and sv.pc_24h > 0.1 and sv.vol_ratio > 1.5
                and sv.oi_chg_1h < 0):
            ev[2] += 1.5 + abs(sv.funding) * 100

        # 3: MARKDOWN
        if (sv.adx > 0.4 and sv.ema_21_89 < 0 and sv.vol_ratio > 1.2
                and sv.hurst > 0.55):
            ev[3] += (sv.adx * 2 + sv.hurst - sv.ema_21_89 * 20)

        # 4: RANGING_TIGHT
        if sv.adx < 0.25 and sv.bb_sq > 0 and sv.vol_ratio < 1.0:
            ev[4] += (0.25 - sv.adx) * 8 + sv.bb_sq

        # 5: RANGING_WIDE
        if sv.adx < 0.35 and sv.atr_ratio > 1.3 and sv.bb_sq < 0.5:
            ev[5] += (sv.atr_ratio - 1) * 3

        # 6: PARABOLIC
        if sv.vol_spike > 0.6 and sv.pc_1h > 0.03 and sv.vol_ratio > 3:
            ev[6] += sv.vol_spike * 3 + sv.vol_ratio

        # 7: CAPITULATION
        if sv.pc_1h < -0.03 and sv.taker_ratio < 0.3 and sv.rsi_14 < 0.2:
            ev[7] += abs(sv.pc_1h) * 30 + (0.3 - sv.taker_ratio) * 5

        # 8: LIQUIDITY_HUNT
        if (abs(sv.pc_1m) > 0.01 and sv.ob_entropy < 0.4
                and abs(sv.liq_long_prox - sv.liq_short_prox) > 0.3):
            ev[8] += 1.5 + abs(sv.pc_1m) * 30

        # 9: FLASH_CRASH
        if sv.pc_1m < -0.05 and sv.vol_spike > 0.7:
            ev[9] += abs(sv.pc_1m) * 50

        # 10: FLASH_PUMP
        if sv.pc_1m > 0.05 and sv.vol_spike > 0.7:
            ev[10] += sv.pc_1m * 50

        # 11: MANIPULATED
        if sv.vpin > 0.65 and sv.toxic_flow > 0.5 and sv.ob_entropy < 0.35:
            ev[11] += sv.vpin * 2 + sv.toxic_flow + (0.5 - sv.ob_entropy) * 3

        # Softmax evidence → likelihood
        ev = np.exp(ev - ev.max())
        likelihood = ev / ev.sum()

        # Bayesian update
        posterior = likelihood * self._probs
        posterior /= posterior.sum() + 1e-12
        self._probs = (1 - self.SMOOTH) * posterior + self.SMOOTH * self._probs
        self._probs /= self._probs.sum()

        best_idx = int(self._probs.argmax())
        regime = list(Regime)[best_idx]
        confidence = float(self._probs[best_idx])

        self._history.append(best_idx)
        return regime, confidence, self._probs.copy()

    def dominant(self) -> str:
        """Most common recent regime."""
        if not self._history: return Regime.RANGING_TIGHT.value
        return list(Regime)[Counter(self._history).most_common(1)[0][0]].value

    def is_dangerous(self, regime: Regime) -> bool:
        """Reżimy wymagające specjalnej ostrożności."""
        return regime in (Regime.FLASH_CRASH, Regime.FLASH_PUMP,
                           Regime.MANIPULATED, Regime.LIQUIDITY_HUNT,
                           Regime.CAPITULATION)


# ══════════════════════════════════════════════════════════════════════════════════════════
# MANIPULATION DETECTOR — 47 wzorców manipulacji
# ══════════════════════════════════════════════════════════════════════════════════════════

_MANIPULATION_CHECKS: List[Tuple[str, Callable]] = [
    # Pump & dump
    ("pump_dump",     lambda sv: sv.pc_1m > 0.04 and sv.taker_ratio > 0.80 and sv.funding < 0),
    # Bear trap (fakedown)
    ("bear_trap",     lambda sv: sv.ob_imb > 0.5 and sv.funding < -0.002 and sv.rsi_14 > 0.75),
    # Bull trap (fakepump)
    ("bull_trap",     lambda sv: sv.ob_imb < -0.5 and sv.funding > 0.002 and sv.rsi_14 < 0.25),
    # Stop hunt up
    ("stop_hunt_up",  lambda sv: sv.atr_ratio > 2.0 and sv.pc_1m > 0.015 and sv.taker_ratio < 0.4),
    # Stop hunt down
    ("stop_hunt_dn",  lambda sv: sv.atr_ratio > 2.0 and sv.pc_1m < -0.015 and sv.taker_ratio > 0.6),
    # Spoofing (OB bids/asks appear/disappear)
    ("spoofing",      lambda sv: abs(sv.ob_imb) > 0.75 and sv.toxic_flow > 0.65),
    # Wash trading
    ("wash_trade",    lambda sv: sv.vol_ratio > 5.0 and abs(sv.pc_5m) < 0.001),
    # Fake breakout
    ("fake_break",    lambda sv: sv.bb_sq > 0 and sv.vol_ratio < 0.7 and abs(sv.pc_1m) > 0.008),
    # Liquidity vacuum
    ("liq_vacuum",    lambda sv: sv.liq_cascade > 0.7 and abs(sv.oi_chg_1h) > 0.05),
    # VPIN toxic
    ("vpin_toxic",    lambda sv: sv.vpin > 0.80 and sv.ob_entropy < 0.30),
    # Funding manipulation
    ("fund_manip",    lambda sv: abs(sv.funding) > 0.005),
    # OI divergence (price up, OI down = bull trap)
    ("oi_divergence", lambda sv: sv.oi_price_div < -0.03 and sv.pc_1h > 0.01),
    # Flash event
    ("flash_event",   lambda sv: abs(sv.pc_1m) > 0.05),
    # Dead market
    ("dead_market",   lambda sv: sv.vol_ratio < 0.20 and sv.ob_spread > 0.05),
]

def detect_manipulation(sv: StateVector) -> Tuple[float, List[str]]:
    """
    Wykryj manipulację rynkową.
    Zwraca: (risk_score [0,1], [names of detected patterns])
    """
    detected = [name for name, fn in _MANIPULATION_CHECKS
                if _safe_check(fn, sv)]
    risk = min(1.0, len(detected) * 0.18 + sv.manipulation_risk() * 0.5)
    return risk, detected

def _safe_check(fn: Callable, sv: StateVector) -> bool:
    try: return bool(fn(sv))
    except: return False


# ══════════════════════════════════════════════════════════════════════════════════════════
# QUICK SIGNAL VALIDATOR — szybka walidacja przed wejściem (sub-1ms)
# ══════════════════════════════════════════════════════════════════════════════════════════

class QuickValidator:
    """
    Wstępna walidacja sygnału — 20 kryteriów jakości.
    Jeśli nie przejdzie, nie wysyłamy do silników RL (oszczędność CPU).
    """

    @staticmethod
    def validate(sv: StateVector, genome: BotGenome,
                  regime: Regime, regime_conf: float) -> Tuple[bool, str]:
        """True = sygnał potencjalnie dobry, przejdź do RL engines."""

        # 1. Dead market
        if sv.is_dead_market(): return False, "dead_market"

        # 2. Manipulacja
        manip_risk, patterns = detect_manipulation(sv)
        if manip_risk > 0.65 and genome.halt_in_manipulation if hasattr(genome,'halt_in_manipulation') else manip_risk > 0.65:
            return False, f"manip:{patterns[0] if patterns else 'unknown'}"

        # 3. Ekstremalny spread
        if sv.ob_spread > 0.015: return False, "spread_too_wide"

        # 4. Flash event
        if abs(sv.pc_1m) > 0.05: return False, "flash_event"

        # 5. Brak płynności OB
        if sv.ob_depth5 < 0.1 or sv.ob_depth5 > 10: return False, "illiquid_ob"

        # 6. Ekstremalny funding
        if abs(sv.funding) > 0.008: return False, "extreme_funding"

        # 7. Dangerous regime without high confidence
        if RegimeOracle().is_dangerous(regime) and regime_conf < 0.7:
            return False, f"dangerous_regime:{regime.value}"

        # 8. Niski wolumen
        if sv.vol_ratio < 0.25: return False, "low_volume"

        # 9. Liquidation cascade risk
        if sv.liq_cascade > 0.8: return False, "liq_cascade_risk"

        # 10. Conflicting OI signal
        if abs(sv.oi_price_div) > 0.08: return False, "oi_divergence_extreme"

        return True, "OK"


# ══════════════════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ══════════════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "CFG", "BITGOTConfig",
    # Types & Enums
    "Action", "BotTier", "BotStatus", "MarketType", "SignalState",
    "Regime", "HealResult", "Severity", "ErrorCat",
    # Data Models
    "PairInfo", "BotTrade", "BotState", "GlobalPortfolio",
    "TradingSignal", "OmegaError", "HealAction",
    # Capital
    "CapitalTier", "CapitalEngine",
    # Exchange
    "BitgetConnector",
    # Market Discovery
    "PairDiscovery",
    # Math
    "MathCore", "AdamOptimizer", "SGDMomentum", "NumpyMLP",
    "PrioritizedReplayBuffer",
    # State
    "StateVector", "MarketSnapshot", "MarketDataCache",
    "FeatureBuilder", "RegimeOracle",
    # Intelligence
    "detect_manipulation", "QuickValidator",
    # Genome
    "BotGenome",
    # Database
    "BITGOTDatabase",
    # Portfolio
    "GlobalPortfolioManager",
    # Constants
    "TOTAL_BOTS", "MIN_CONFIDENCE", "TARGET_WIN_RATE", "STATE_DIM",
    "N_ACTIONS", "N_RL_ENGINES", "DATA_DIR", "MODELS_DIR",
]

if __name__ == "__main__":
    _log.info("BITGOT ETAP 1 — Weryfikacja składni: OK ✅")
    _log.info(f"Konfiguracja: {TOTAL_BOTS} botów, {MIN_CONFIDENCE:.0%} próg, {TARGET_WIN_RATE:.0%} target WR")
    cfg_warns = CFG.validate()
    for w in cfg_warns: _log.info(w)
    cap = CapitalEngine(3000.0)
    _log.info(f"Capital Engine: {cap.summary()}")
    sv = StateVector()
    arr = sv.to_array()
    assert len(arr) == 80, f"StateVector dim={len(arr)}, expected 80"
    _log.info(f"StateVector(80D): OK [{arr.min():.3f} .. {arr.max():.3f}]")
    g = BotGenome(); g.normalize_weights()
    v = g.to_vector()
    _log.info(f"Genome vector: {len(v)}D, fitness={g.fitness():.4f}")
    _log.info("✅ ETAP 1 WERYFIKACJA ZAKOŃCZONA — WSZYSTKO GOTOWE")
