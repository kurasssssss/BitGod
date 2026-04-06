"""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║  ██████╗ ██╗████████╗ ██████╗  ██████╗ ████████╗   ███████╗██╗███╗   ██╗ █████╗ ██╗       ║
║  ██╔══██╗██║╚══██╔══╝██╔════╝ ██╔═══██╗╚══██╔══╝   ██╔════╝██║████╗  ██║██╔══██╗██║       ║
║  ██████╔╝██║   ██║   ██║  ███╗██║   ██║   ██║      █████╗  ██║██╔██╗ ██║███████║██║       ║
║  ██╔══██╗██║   ██║   ██║   ██║██║   ██║   ██║      ██╔══╝  ██║██║╚██╗██║██╔══██║██║       ║
║  ██████╔╝██║   ██║   ╚██████╔╝╚██████╔╝   ██║      ██║     ██║██║ ╚████║██║  ██║███████╗  ║
║  ╚═════╝ ╚═╝   ╚═╝    ╚═════╝  ╚═════╝    ╚═╝      ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝  ║
║                                                                                              ║
║  ██████╗     E T A P   4 / 4   —   O R C H E S T R A T O R                               ║
║                                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║  MODUŁY ETAPU 4 (FINALNEGO):                                                                ║
║                                                                                              ║
║  ▸ BITGOTOrchestrator  — łączy E1+E2+E3 w jeden żywy organizm                             ║
║    Sekwencja startu: Config → Connect → Discover → Build → Run                             ║
║    Graceful shutdown: pozycje → save → checkpoint → exit                                   ║
║                                                                                              ║
║  ▸ LiveDashboard       — terminal dashboard (30s refresh)                                   ║
║    Kolorowa konsola: PnL · WR · active positions · top bots · tier dist                   ║
║    ASCII art charts: equity curve · trades per hour                                        ║
║                                                                                              ║
║  ▸ RestAPI             — FastAPI server port 8888                                           ║
║    GET /status         — pełny snapshot systemu                                             ║
║    GET /bots           — lista 3000 botów ze statystykami                                  ║
║    GET /positions      — aktywne pozycje                                                    ║
║    GET /signals        — ostatnie 100 sygnałów                                             ║
║    GET /performance    — equity curve, drawdown, tier stats                               ║
║    GET /genome/{id}    — genome konkretnego bota                                           ║
║    POST /halt          — emergency halt trading                                             ║
║    POST /resume        — wznów po halcie                                                    ║
║    GET /health         — health check dla Replit                                            ║
║                                                                                              ║
║  ▸ MetricsEngine       — zbiera i eksportuje metryki co 30s                                ║
║    Equity curve · drawdown · tier WR · signals per hour                                   ║
║    Top/bottom 10 botów · manipulation alerts · healer XP                                  ║
║                                                                                              ║
║  ▸ CLI                 — argument parser                                                    ║
║    --paper / --live                                                                         ║
║    --capital 3000 (start capital $)                                                        ║
║    --bots 3000                                                                              ║
║    --api-port 8888                                                                          ║
║    --no-api                                                                                  ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal
import sys
import time
import threading
import traceback
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Importy z poprzednich etapów ──────────────────────────────────────────────
from bitgot_e1 import (
    CFG, BITGOTConfig, BotTier, BotState, BotGenome,
    BitgetConnector, PairDiscovery, PairInfo, CapitalEngine,
    BITGOTDatabase, GlobalPortfolioManager, MarketDataCache,
    MathCore, StateVector, FeatureBuilder,
    TOTAL_BOTS, MIN_CONFIDENCE, TARGET_WIN_RATE, DATA_DIR, MODELS_DIR,
    _TS, _MS, _NOW,
)
from bitgot_e2 import (
    BotBrain, SwarmIntelligence, GlobalMetaPool,
)
from bitgot_e3 import (
    BitgotSystemE3, BotScout, DataFetcher,
    CircuitBreakers, OmegaHealerDaemon, TierManager,
    GenomeEvolution, SignalManager,
)

_log = logging.getLogger("BITGOT·MAIN")

UTC = timezone.utc

# ══════════════════════════════════════════════════════════════════════════════════════════
# METRICS ENGINE — zbiera i eksportuje metryki systemowe
# ══════════════════════════════════════════════════════════════════════════════════════════

class MetricsEngine:
    """
    Zbiera metryki co 30s i przechowuje historię dla dashboardu i API.
    """

    HISTORY_POINTS = 2880    # 24h przy 30s interwale

    def __init__(self, system: "BITGOTOrchestrator"):
        self.sys = system
        self._equity_curve:  deque = deque(maxlen=self.HISTORY_POINTS)
        self._wr_curve:      deque = deque(maxlen=self.HISTORY_POINTS)
        self._trades_curve:  deque = deque(maxlen=self.HISTORY_POINTS)
        self._signals_curve: deque = deque(maxlen=self.HISTORY_POINTS)
        self._last_trades    = 0
        self._last_signals   = 0
        self._log = logging.getLogger("BITGOT·Metrics")
        self._running = False
        self._snapshots: deque = deque(maxlen=200)

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(CFG.metrics_interval_s)
            try:
                snap = self._collect()
                self._snapshots.append(snap)
                self.sys.db.q_metric(snap)
            except Exception as e:
                self._log.debug(f"Metrics: {e}")

    def _collect(self) -> Dict:
        e3  = self.sys.e3
        gp  = self.sys.portfolio.snapshot
        now = int(_TS())

        scouts = list(e3.scouts.values()) if e3 else []
        total_trades = sum(s._state.n_trades for s in scouts)
        total_pnl    = sum(s._state.total_pnl for s in scouts)
        all_wrs      = [s._state.wr for s in scouts if s._state.n_trades >= 10]
        global_wr    = float(np.mean(all_wrs)) if all_wrs else 0.0
        active_pos   = len(e3.positions) if e3 else 0

        # Tier stats
        tier_wrs = {}
        for tier in BotTier:
            tiers_scouts = [s for s in scouts if s.pair.tier == tier and s._state.n_trades >= 5]
            if tiers_scouts:
                tier_wrs[tier.value] = float(np.mean([s._state.wr for s in tiers_scouts]))
            else:
                tier_wrs[tier.value] = 0.0

        # Delta trades/signals since last collection
        trades_delta  = total_trades - self._last_trades
        self._last_trades = total_trades

        sig_mgr = e3.sig_manager if e3 else None
        signals_total = sig_mgr._n_received if sig_mgr else 0
        signals_delta = signals_total - self._last_signals
        self._last_signals = signals_total

        healer = e3.healer if e3 else None
        omega_lvl = healer._xp.level if healer else 1

        snap = {
            "ts":             now,
            "total_bots":     len(scouts),
            "active_positions": active_pos,
            "trades_1min":    trades_delta,
            "pnl_1min":       0.0,
            "total_pnl":      round(total_pnl, 6),
            "total_trades":   total_trades,
            "global_wr":      round(global_wr, 4),
            "portfolio":      round(gp.total_capital, 4),
            "tier_apex_wr":   round(tier_wrs.get("apex", 0), 4),
            "tier_elite_wr":  round(tier_wrs.get("elite", 0), 4),
            "tier_std_wr":    round(tier_wrs.get("standard", 0), 4),
            "tier_scout_wr":  round(tier_wrs.get("scout", 0), 4),
            "signals_1min":   signals_delta,
            "heals_1min":     healer._heals_this_min if healer else 0,
            "omega_level":    omega_lvl,
            "avg_confidence": round(float(np.mean(list(sig_mgr._conf_history))) if sig_mgr and sig_mgr._conf_history else 0, 4),
        }
        # Store curves
        self._equity_curve.append((now, round(total_pnl, 4)))
        self._wr_curve.append((now, round(global_wr, 4)))
        self._trades_curve.append((now, trades_delta))
        self._signals_curve.append((now, signals_delta))
        return snap

    def equity_curve(self, n: int = 100) -> List[Tuple]:
        return list(self._equity_curve)[-n:]

    def wr_curve(self, n: int = 100) -> List[Tuple]:
        return list(self._wr_curve)[-n:]

    def last_snapshot(self) -> Dict:
        if self._snapshots: return self._snapshots[-1]
        return {}

    def top_bots(self, n: int = 10) -> List[Dict]:
        e3 = self.sys.e3
        if not e3: return []
        scouts = [s for s in e3.scouts.values() if s._state.n_trades >= 20]
        scouts.sort(key=lambda s: s._state.wr, reverse=True)
        return [{
            "bot_id": s.bot_id, "symbol": s.pair.symbol,
            "tier": s.pair.tier.value, "wr": round(s._state.wr, 4),
            "pnl": round(s._state.total_pnl, 5), "trades": s._state.n_trades,
        } for s in scouts[:n]]

    def bottom_bots(self, n: int = 10) -> List[Dict]:
        e3 = self.sys.e3
        if not e3: return []
        scouts = [s for s in e3.scouts.values() if s._state.n_trades >= 20]
        scouts.sort(key=lambda s: s._state.wr)
        return [{
            "bot_id": s.bot_id, "symbol": s.pair.symbol,
            "tier": s.pair.tier.value, "wr": round(s._state.wr, 4),
            "pnl": round(s._state.total_pnl, 5), "trades": s._state.n_trades,
        } for s in scouts[:n]]

    def stop(self): self._running = False


# ══════════════════════════════════════════════════════════════════════════════════════════
# LIVE DASHBOARD — terminal dashboard (30s refresh)
# ══════════════════════════════════════════════════════════════════════════════════════════

class LiveDashboard:
    """
    Kolorowy terminal dashboard aktualizowany co 30s.
    Używa ANSI escape codes — działa w każdym terminalu.
    """

    REFRESH_S = 30.0

    # ANSI
    R  = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
    C  = "\033[96m"; M  = "\033[95m"; B  = "\033[94m"
    W  = "\033[97m"; DM = "\033[2m";  BO = "\033[1m"
    RS = "\033[0m"

    def __init__(self, orchestrator: "BITGOTOrchestrator",
                  metrics: MetricsEngine):
        self.orch    = orchestrator
        self.metrics = metrics
        self._start  = _TS()
        self._log    = logging.getLogger("BITGOT·Dashboard")
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(self.REFRESH_S)
            try:
                self._render()
            except Exception as e:
                self._log.debug(f"Dashboard: {e}")

    def _render(self):
        """Renderuj dashboard w terminalu."""
        snap  = self.metrics.last_snapshot()
        if not snap: return
        gp    = self.orch.portfolio.snapshot
        e3    = self.orch.e3
        healer= e3.healer if e3 else None
        circ  = e3.circuit if e3 else None

        uptime = _TS() - self._start
        h, rem = divmod(int(uptime), 3600)
        m, s   = divmod(rem, 60)

        pnl   = snap.get("total_pnl", 0)
        wr    = snap.get("global_wr", 0)
        port  = snap.get("portfolio", 0)
        apr   = gp.active_positions
        trades= snap.get("total_trades", 0)
        conf  = snap.get("avg_confidence", 0)

        pnl_c  = self.G if pnl >= 0 else self.R
        wr_c   = self.G if wr >= 0.85 else self.Y if wr >= 0.70 else self.R
        halt_c = self.R if gp.halted else self.G

        print(f"\n{self.BO}{self.C}{'═'*88}{self.RS}")
        print(f"{self.BO}  ██████╗ ██╗████████╗ ██████╗  ██████╗ ████████╗   "
              f"{self.Y}v∞   {h:02d}:{m:02d}:{s:02d}{self.RS}")
        print(f"{self.DM}{'─'*88}{self.RS}")

        # Row 1: Key metrics
        print(
            f"  Capital: {self.W}{self.BO}${port:>10,.2f}{self.RS}"
            f"  PnL: {pnl_c}{self.BO}{pnl:>+10.4f}${self.RS}"
            f"  WR: {wr_c}{self.BO}{wr:.1%}{self.RS}"
            f"  Active: {self.Y}{self.BO}{apr:>4d}{self.RS}"
            f"  Trades: {self.C}{trades:>7,}{self.RS}"
        )

        # Row 2: Tier WRs
        awr = snap.get("tier_apex_wr",  0); ewr = snap.get("tier_elite_wr",0)
        swr = snap.get("tier_std_wr",   0); scwr= snap.get("tier_scout_wr",0)
        print(
            f"  APEX:{self._wr_color(awr)}{awr:.0%}{self.RS}"
            f"  ELITE:{self._wr_color(ewr)}{ewr:.0%}{self.RS}"
            f"  STD:{self._wr_color(swr)}{swr:.0%}{self.RS}"
            f"  SCOUT:{self._wr_color(scwr)}{scwr:.0%}{self.RS}"
            f"  Conf:{self.M}{conf:.0%}{self.RS}"
            f"  Sigs/min:{self.B}{snap.get('signals_1min',0):>4}{self.RS}"
            f"  Heals/min:{self.M}{snap.get('heals_1min',0):>3}{self.RS}"
        )

        # Row 3: System status
        omega_lvl  = snap.get("omega_level", 1)
        halt_str   = f"{self.R}⚠ HALTED{self.RS}" if gp.halted else f"{self.G}✓ RUNNING{self.RS}"
        dd         = gp.drawdown_pct
        dd_c       = self.R if dd > 0.08 else self.Y if dd > 0.04 else self.G
        print(
            f"  Status:{halt_str}"
            f"  Drawdown:{dd_c}{dd:.2%}{self.RS}"
            f"  Daily PnL:{pnl_c}{gp.daily_pnl:+.4f}${self.RS}"
            f"  Omega Lv.{self.M}{omega_lvl}{self.RS}"
        )

        # Circuit breakers
        if circ:
            cs = circ.stats()
            l3 = f"{self.R}L3-HALT{self.RS}" if cs['l3_halted'] else f"{self.G}L3-OK{self.RS}"
            l4 = f"{self.R}L4-HALT{self.RS}" if cs['l4_halted'] else f"{self.G}L4-OK{self.RS}"
            print(
                f"  Circuit: {l3} {l4}"
                f"  L1-open:{self.Y}{cs['l1_open_bots']}{self.RS}"
                f"  Throttle:{self.Y}{cs['l2_throttle']:.0%}{self.RS}"
            )

        # Equity mini-chart (ASCII)
        curve = self.metrics.equity_curve(20)
        if len(curve) >= 5:
            vals = [v for _, v in curve]
            chart = self._mini_chart(vals, width=40, height=4)
            print(f"\n  Equity curve (last {len(curve)} pts):")
            for line in chart: print(f"  {self.DM}{line}{self.RS}")

        # Top 5 bots
        top = self.metrics.top_bots(5)
        if top:
            print(f"\n  {self.BO}TOP 5 BOTS:{self.RS}")
            for b in top:
                tw = b['wr']; tc = self._wr_color(tw)
                print(f"  #{b['bot_id']:04d} {b['symbol']:<22} "
                      f"[{b['tier'].upper():<8}] "
                      f"WR:{tc}{tw:.1%}{self.RS} "
                      f"PnL:{pnl_c if b['pnl']>=0 else self.R}"
                      f"{b['pnl']:>+.4f}${self.RS} "
                      f"T:{b['trades']:>4}")

        print(f"{self.BO}{self.C}{'═'*88}{self.RS}\n")

    def _wr_color(self, wr: float) -> str:
        if wr >= 0.85: return self.G
        if wr >= 0.70: return self.Y
        return self.R

    def _mini_chart(self, vals: List[float], width: int = 40, height: int = 4) -> List[str]:
        """ASCII line chart."""
        if not vals or len(vals) < 2: return []
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        if rng == 0: rng = 1e-10
        grid = [[" "] * width for _ in range(height)]
        n = len(vals)
        for i, v in enumerate(vals):
            x = int(i / n * (width - 1))
            y = int((v - mn) / rng * (height - 1))
            y = height - 1 - y
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = "█" if v >= 0 else "▄"
        lines = ["".join(row) for row in grid]
        # Axis labels
        lines[-1] = f"{mn:+.3f}" + lines[-1][8:] + f"{mx:+.3f}"[:8]
        return lines

    def stop(self): self._running = False


# ══════════════════════════════════════════════════════════════════════════════════════════
# REST API — FastAPI server port 8888
# ══════════════════════════════════════════════════════════════════════════════════════════

def build_api(orchestrator: "BITGOTOrchestrator", metrics: MetricsEngine):
    """Buduje FastAPI aplikację."""
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        _log.warning("FastAPI not installed — API disabled")
        return None

    app = FastAPI(
        title="BITGOT API",
        description="3000 Bot Bitget Trading System",
        version="∞.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        """Health check — Replit uptime monitor."""
        return {"status": "ok", "ts": _NOW(), "uptime_s": round(_TS() - orchestrator._start_ts, 1)}

    @app.get("/status")
    async def status():
        """Full system snapshot."""
        e3 = orchestrator.e3
        return JSONResponse({
            "system":    "BITGOT",
            "mode":      "PAPER" if CFG.paper_mode else "LIVE",
            "ts":        _NOW(),
            "uptime_s":  round(_TS() - orchestrator._start_ts, 1),
            "portfolio": orchestrator.portfolio.snapshot.__dict__,
            "metrics":   metrics.last_snapshot(),
            "e3":        e3.get_status() if e3 else {},
            "circuit":   e3.circuit.stats() if e3 else {},
            "healer":    e3.healer.stats() if e3 else {},
            "swarm":     orchestrator.swarm.global_stats(),
            "genome_evo":e3.genome_evo.stats() if e3 and e3.genome_evo else {},
        })

    @app.get("/bots")
    async def bots(tier: str = "", limit: int = 100, offset: int = 0):
        """Lista botów ze statystykami."""
        e3 = orchestrator.e3
        if not e3: return {"bots": []}
        scouts = list(e3.scouts.values())
        if tier: scouts = [s for s in scouts if s.pair.tier.value == tier]
        scouts.sort(key=lambda s: s._state.wr, reverse=True)
        page = scouts[offset:offset+limit]
        return {
            "total": len(scouts),
            "offset": offset,
            "limit": limit,
            "bots": [s.stats() for s in page],
        }

    @app.get("/positions")
    async def positions():
        """Aktywne otwarte pozycje."""
        e3 = orchestrator.e3
        if not e3: return {"positions": []}
        pos_list = []
        for bot_id, pos in e3.positions.items():
            snap = orchestrator.mdc.get(pos.symbol)
            cur_price = snap.price if snap else pos.entry_price
            if pos.side == "long":
                upnl = (cur_price - pos.entry_price) / pos.entry_price * pos.notional
            else:
                upnl = (pos.entry_price - cur_price) / pos.entry_price * pos.notional
            pos_list.append({
                "bot_id": pos.bot_id, "symbol": pos.symbol, "side": pos.side,
                "entry": pos.entry_price, "current": cur_price,
                "upnl": round(upnl, 5), "age_s": round(pos.age_s, 1),
                "leverage": pos.leverage, "margin": pos.margin,
                "sl": pos.sl_price, "tp": pos.tp_price,
                "trailing_armed": pos.trailing_armed,
            })
        pos_list.sort(key=lambda p: abs(p["upnl"]), reverse=True)
        return {"count": len(pos_list), "positions": pos_list}

    @app.get("/signals")
    async def signals_endpoint(limit: int = 100):
        """Ostatnie sygnały."""
        e3 = orchestrator.e3
        if not e3 or not e3.sig_manager: return {"signals": []}
        hist = list(e3.sig_manager._signal_history)[-limit:]
        return {
            "total_received": e3.sig_manager._n_received,
            "total_passed":   e3.sig_manager._n_passed,
            "pass_rate":      round(e3.sig_manager._n_passed / max(e3.sig_manager._n_received, 1), 4),
            "signals": [
                {
                    "symbol": r.signal.symbol, "side": r.signal.side,
                    "confidence": round(r.signal.confidence, 4),
                    "regime": r.signal.regime,
                    "leverage": r.signal.leverage,
                    "margin": round(r.signal.margin, 4),
                    "ts": r.received_ts,
                }
                for r in hist
            ]
        }

    @app.get("/performance")
    async def performance():
        """Equity curve, drawdown, Sharpe."""
        gp = orchestrator.portfolio.snapshot
        equity = metrics.equity_curve(288)   # last 24h
        wr_hist = metrics.wr_curve(100)
        vals = [v for _, v in equity]
        rets = np.diff(vals) if len(vals) > 1 else []
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(288)) if len(rets) > 5 else 0.0
        return {
            "total_pnl":      round(gp.total_pnl, 6),
            "daily_pnl":      round(gp.daily_pnl, 6),
            "drawdown_pct":   round(gp.drawdown_pct, 4),
            "peak_capital":   round(gp.peak_capital, 4),
            "global_wr":      round(gp.global_wr, 4),
            "total_trades":   gp.total_trades,
            "sharpe_30s":     round(sharpe, 4),
            "equity_curve":   equity,
            "wr_curve":       wr_hist,
            "top_bots":       metrics.top_bots(10),
            "bottom_bots":    metrics.bottom_bots(5),
            "tier_dist":      orchestrator.e3.tier_mgr.stats() if orchestrator.e3 and orchestrator.e3.tier_mgr else {},
        }

    @app.get("/genome/{bot_id}")
    async def genome(bot_id: int):
        """Genome konkretnego bota."""
        e3 = orchestrator.e3
        if not e3: raise HTTPException(404, "E3 not running")
        brain = e3.brains.get(bot_id)
        if not brain: raise HTTPException(404, f"Bot {bot_id} not found")
        g = brain.genome
        return {
            "bot_id": bot_id, "gid": g.gid,
            "generation": g.generation, "fitness": round(g.fitness(), 4),
            "wr": round(g.wr(), 4), "n_trades": g.n_trades,
            "genome": g.to_dict(),
        }

    @app.post("/halt")
    async def halt(reason: str = "manual"):
        """Emergency halt."""
        e3 = orchestrator.e3
        if e3 and e3.circuit:
            e3.circuit._l4_halted = True
        _log.critical(f"🚨 MANUAL HALT: {reason}")
        return {"halted": True, "reason": reason, "ts": _NOW()}

    @app.post("/resume")
    async def resume():
        """Wznów po halcie."""
        e3 = orchestrator.e3
        if e3 and e3.circuit:
            e3.circuit.resume_l4()
            e3.circuit.reset_daily()
        _log.info("✅ System resumed manually")
        return {"resumed": True, "ts": _NOW()}

    @app.get("/swarm")
    async def swarm_stats():
        """Swarm intelligence stats."""
        return orchestrator.swarm.global_stats()

    @app.get("/healer")
    async def healer_stats():
        """Omega Healer stats."""
        e3 = orchestrator.e3
        if not e3 or not e3.healer: return {}
        return e3.healer.stats()

    return app


# ══════════════════════════════════════════════════════════════════════════════════════════
# BITGOT ORCHESTRATOR — główny dyrygent systemu
# ══════════════════════════════════════════════════════════════════════════════════════════

class BITGOTOrchestrator:
    """
    Absolutny dyrygent systemu BITGOT.
    
    Sekwencja startu:
    1. Validate config (API keys, parameters)
    2. Connect to Bitget (test API)
    3. Discover 3000+ unique pairs
    4. Initialize MarketDataCache
    5. Build BotBrains (RL+Neural+Micro per bot)
    6. Build SwarmIntelligence + GlobalMetaPool
    7. Initialize BitgotSystemE3 (scouts, executors, monitors)
    8. Start REST API (background thread)
    9. Launch all async tasks (3000+ coroutines)
    10. Run LiveDashboard + MetricsEngine
    
    Graceful shutdown:
    - SIGINT/SIGTERM → drain signals → close positions → save models → exit
    """

    def __init__(self, cfg: BITGOTConfig = CFG):
        self.cfg      = cfg
        self._log     = logging.getLogger("BITGOT·Orch")
        self._start_ts = _TS()
        self._running  = False

        # Core components
        self.db         = BITGOTDatabase()
        self.portfolio  = GlobalPortfolioManager(cfg.start_capital)
        self.capital    = CapitalEngine(cfg.start_capital)
        self.mdc        = MarketDataCache()
        self.connector  = BitgetConnector(cfg)
        self.swarm      = SwarmIntelligence(cfg.n_bots)
        self.meta_pool  = GlobalMetaPool()

        # Built later
        self.pairs:  List[PairInfo]         = []
        self.brains: Dict[int, BotBrain]    = {}
        self.e3:     Optional[BitgotSystemE3] = None

        # Metrics + Dashboard + API
        self.metrics:   Optional[MetricsEngine]  = None
        self.dashboard: Optional[LiveDashboard]  = None
        self._api_thread: Optional[threading.Thread] = None
        self._api_app = None

        # Tasks
        self._main_tasks: list = []
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> bool:
        """Inicjalizuj wszystkie komponenty. Zwraca True jeśli sukces."""
        self._print_banner()

        # ── 1. Validate config ────────────────────────────────────────────────
        warnings = self.cfg.validate()
        for w in warnings: self._log.info(w)
        if not self.cfg.api_key and not self.cfg.paper_mode:
            self._log.error("❌ No API key — set BITGET_KEY or use --paper mode")
            return False

        # ── 2. Connect to Bitget ─────────────────────────────────────────────
        self._log.info("📡 Connecting to Bitget...")
        ok = await self.connector.connect()
        if not ok and not self.cfg.paper_mode:
            self._log.error("❌ Cannot connect to Bitget")
            return False
        if not ok:
            self._log.warning("⚠️  Bitget offline — running in full simulation mode")

        # ── 3. Discover pairs ─────────────────────────────────────────────────
        self._log.info(f"🔍 Discovering pairs for {self.cfg.n_bots} bots...")
        discovery = PairDiscovery(self.connector, self.cfg)
        if ok:
            await discovery.discover()
        self.pairs = discovery.assign_unique(self.cfg.n_bots)
        self._log.info(f"✅ {len(self.pairs)} unique pairs assigned")

        # ── 4. Build BotBrains ────────────────────────────────────────────────
        self._log.info("🧠 Building bot brains...")
        n = len(self.pairs)
        for i, pair in enumerate(self.pairs):
            genome = BotGenome(bot_id=i)
            genome.normalize_weights()
            meta   = self.meta_pool.get(pair.tier)
            brain  = BotBrain(pair.symbol, i, pair.tier, genome, self.swarm, meta)
            self.brains[i] = brain
            if i % 500 == 0:
                self._log.info(f"  Built {i}/{n} brains...")
        self._log.info(f"✅ {len(self.brains)} brains ready")

        # ── 5. Initialize E3 ──────────────────────────────────────────────────
        self._log.info("⚙️  Initializing E3 execution layer...")
        self.e3 = BitgotSystemE3(
            connector  = self.connector,
            pairs      = self.pairs,
            brains     = self.brains,
            swarm      = self.swarm,
            meta_pool  = self.meta_pool,
            capital    = self.capital,
            portfolio  = self.portfolio,
            db         = self.db,
            mdc        = self.mdc,
            cfg        = self.cfg,
        )
        await self.e3.initialize()

        # ── 6. Metrics + Dashboard ────────────────────────────────────────────
        self.metrics   = MetricsEngine(self)
        self.dashboard = LiveDashboard(self, self.metrics)

        # ── 7. REST API ───────────────────────────────────────────────────────
        self._api_app = build_api(self, self.metrics)

        self._log.info("✅ BITGOT fully initialized")
        return True

    async def run(self):
        """Główna pętla systemu."""
        if not await self.initialize():
            self._log.error("Initialization failed")
            return
        self._running = True
        self._register_signals()

        self._log.info("=" * 72)
        self._log.info("  🚀 BITGOT LAUNCH — ALL SYSTEMS GO")
        self._log.info(f"  Mode:    {'📄 PAPER' if self.cfg.paper_mode else '💰 LIVE'}")
        self._log.info(f"  Bots:    {len(self.pairs):,}")
        self._log.info(f"  Capital: ${self.cfg.start_capital:,.2f}")
        self._log.info(f"  Target:  {TARGET_WIN_RATE:.0%} WR · {MIN_CONFIDENCE:.0%} conf")
        self._log.info(f"  API:     http://0.0.0.0:{self.cfg.api_port}/status")
        self._log.info("=" * 72)

        # Start REST API in background thread
        if self._api_app:
            self._start_api_thread()

        # Collect all async tasks
        tasks = [
            asyncio.create_task(self.e3.run(),          name="E3-Core"),
            asyncio.create_task(self.metrics.run(),     name="Metrics"),
            asyncio.create_task(self.dashboard.run(),   name="Dashboard"),
            asyncio.create_task(self._meta_update_loop(),name="MetaUpdate"),
            asyncio.create_task(self._health_check_loop(),name="HealthCheck"),
            asyncio.create_task(self._shutdown_wait(),  name="ShutdownWait"),
        ]
        self._main_tasks = tasks
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown."""
        self._log.warning("🛑 BITGOT shutdown initiated...")
        self.metrics.stop()
        self.dashboard.stop()
        if self.e3: await self.e3.stop()
        # Final checkpoint
        saved = 0
        for bot_id, brain in list(self.brains.items())[:100]:
            try: brain.save_all(); self.db.save_genome(brain.genome); saved += 1
            except Exception: pass
        self._log.info(f"💾 Final checkpoint: {saved} brains saved")
        # Print final report
        self._final_report()
        # Close connector
        await self.connector.close()
        self._log.info("✅ BITGOT shutdown complete")

    async def _shutdown_wait(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def _register_signals(self):
        """Register SIGINT/SIGTERM."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(Exception):
                loop.add_signal_handler(sig, self._request_shutdown)

    def _request_shutdown(self):
        self._log.warning("Signal received — shutting down...")
        self._shutdown_event.set()

    async def _meta_update_loop(self):
        """Periodic MAML meta-update for all tiers."""
        while self._running:
            await asyncio.sleep(CFG.meta_update_interval * 60)
            try:
                self.meta_pool.update_all()
            except Exception as e:
                self._log.debug(f"Meta update: {e}")

    async def _health_check_loop(self):
        """Periodic health check — reconnect if needed."""
        while self._running:
            await asyncio.sleep(60)
            try:
                if not self.connector.is_healthy and not self.cfg.paper_mode:
                    self._log.warning("⚠️  Reconnecting to Bitget...")
                    await self.connector.connect()
            except Exception as e:
                self._log.debug(f"Health check: {e}")

    def _start_api_thread(self):
        """Start FastAPI in background thread."""
        try:
            import uvicorn

            config = uvicorn.Config(
                self._api_app,
                host="0.0.0.0",
                port=self.cfg.api_port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            self._api_thread = threading.Thread(
                target=server.run, daemon=True, name="FastAPI"
            )
            self._api_thread.start()
            self._log.info(f"🌐 REST API started on port {self.cfg.api_port}")
        except ImportError:
            self._log.warning("uvicorn not installed — API disabled")
        except Exception as e:
            self._log.warning(f"API start: {e}")

    def _final_report(self):
        """Print final statistics."""
        gp = self.portfolio.snapshot
        e3 = self.e3
        scouts = list(e3.scouts.values()) if e3 else []
        total_trades = sum(s._state.n_trades for s in scouts)
        total_pnl    = sum(s._state.total_pnl for s in scouts)
        all_wrs = [s._state.wr for s in scouts if s._state.n_trades >= 20]
        rt_h = (_TS() - self._start_ts) / 3600
        healer = e3.healer if e3 else None
        evo    = e3.genome_evo if e3 else None

        print(f"\n{'═'*72}")
        print(f"  BITGOT — SHUTDOWN REPORT")
        print(f"{'═'*72}")
        print(f"  Uptime:          {rt_h:.2f}h")
        print(f"  Total Bots:      {len(scouts):,}")
        print(f"  Total Trades:    {total_trades:,}")
        print(f"  Total PnL:       ${total_pnl:+,.4f}")
        print(f"  Portfolio:       ${gp.total_capital:,.4f}")
        print(f"  Global WR:       {float(np.mean(all_wrs)) if all_wrs else 0:.1%}")
        print(f"  Peak Capital:    ${gp.peak_capital:,.4f}")
        print(f"  Max Drawdown:    {gp.drawdown_pct:.2%}")
        if healer:
            print(f"  Omega Level:     {healer._xp.level} [{healer._xp.level_name}]")
            print(f"  Total Heals:     {healer._total_heals:,}")
        if evo:
            print(f"  Evo Generation:  {evo.gen}")
            print(f"  HOF Size:        {len(evo.hof)}")
        print(f"{'═'*72}\n")

    def _print_banner(self):
        C, Y, G, RS, BO = "\033[96m", "\033[93m", "\033[92m", "\033[0m", "\033[1m"
        print(f"\n{BO}{C}")
        print("  ██████╗ ██╗████████╗ ██████╗  ██████╗ ████████╗")
        print("  ██╔══██╗██║╚══██╔══╝██╔════╝ ██╔═══██╗╚══██╔══╝")
        print("  ██████╔╝██║   ██║   ██║  ███╗██║   ██║   ██║   ")
        print("  ██╔══██╗██║   ██║   ██║   ██║██║   ██║   ██║   ")
        print("  ██████╔╝██║   ██║   ╚██████╔╝╚██████╔╝   ██║   ")
        print(f"  ╚═════╝ ╚═╝   ╚═╝    ╚═════╝  ╚═════╝    ╚═╝   {RS}")
        print(f"{Y}  3000 Bot Bitget Supremacy System  v∞.0{RS}")
        print(f"{G}  85%+ WR Target · 200k Trades/Day · Self-Evolving{RS}\n")


# ══════════════════════════════════════════════════════════════════════════════════════════
# CLI — argument parser
# ══════════════════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BITGOT — 3000 Bot Bitget Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bitgot_main.py                   # paper mode (default)
  python bitgot_main.py --live            # LIVE trading (real money)
  python bitgot_main.py --capital 5000    # start with $5000
  python bitgot_main.py --bots 500        # only 500 bots (lighter)
  python bitgot_main.py --live --bots 100 # live with 100 bots
  python bitgot_main.py --no-api          # without REST API
        """
    )
    p.add_argument("--paper",    action="store_true", default=True,  help="Paper mode (default)")
    p.add_argument("--live",     action="store_true", default=False, help="⚠️ LIVE trading")
    p.add_argument("--capital",  type=float, default=3000.0,         help="Start capital USD (default: 3000)")
    p.add_argument("--bots",     type=int,   default=3000,           help="Number of bots (default: 3000)")
    p.add_argument("--api-port", type=int,   default=8888,           help="REST API port (default: 8888)")
    p.add_argument("--no-api",   action="store_true", default=False, help="Disable REST API")
    p.add_argument("--testnet",  action="store_true", default=False, help="Use Bitget testnet")
    p.add_argument("--verbose",  action="store_true", default=False, help="Verbose logging")
    p.add_argument("--conf",     type=float, default=0.80,           help="Min confidence threshold (default: 0.80)")
    return p.parse_args()


def apply_args(args: argparse.Namespace):
    """Zastosuj argumenty CLI do globalnej konfiguracji."""
    global CFG
    if args.live:
        CFG.paper_mode = False
        _log.warning("⚠️  LIVE MODE ACTIVE — real money at risk!")
    else:
        CFG.paper_mode = True
    CFG.start_capital = args.capital
    CFG.n_bots        = args.bots
    CFG.api_port      = args.api_port
    if args.testnet: CFG.testnet = True
    if args.conf:    CFG.confidence_threshold = args.conf
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# ══════════════════════════════════════════════════════════════════════════════════════════
# REQUIREMENTS CHECK — auto-install missing packages
# ══════════════════════════════════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = {
    "ccxt":     "ccxt>=4.3.0",
    "numpy":    "numpy>=1.26.0",
    "fastapi":  "fastapi>=0.111.0",
    "uvicorn":  "uvicorn>=0.29.0",
    "aiohttp":  "aiohttp>=3.9.0",
}

OPTIONAL_PACKAGES = {
    "psutil":   "psutil>=5.9.0",
    "websockets":"websockets>=12.0",
}

def check_requirements() -> bool:
    """Sprawdź i auto-instaluj wymagane pakiety."""
    import importlib
    missing = []
    for imp, pip in REQUIRED_PACKAGES.items():
        try: importlib.import_module(imp)
        except ImportError: missing.append(pip)
    if not missing:
        return True
    print(f"⚙️  Installing {len(missing)} required packages...")
    import subprocess
    for pkg in missing:
        print(f"  pip install {pkg}")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True, timeout=120
        )
        if r.returncode != 0:
            print(f"  ❌ Failed: {pkg}")
            return False
        print(f"  ✅ {pkg}")
    return True


# ══════════════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════════════════

async def _async_main():
    args = parse_args()
    apply_args(args)

    # Check requirements
    if not check_requirements():
        print("❌ Missing required packages. Install with: pip install ccxt numpy fastapi uvicorn aiohttp")
        sys.exit(1)

    # Safety confirmation for live mode
    if args.live and not args.paper:
        print("\n" + "⚠️ " * 20)
        print("  LIVE TRADING MODE — REAL MONEY WILL BE USED")
        print(f"  Capital: ${args.capital:,.2f}")
        print(f"  Bots: {args.bots:,}")
        print("⚠️ " * 20)
        confirm = input("\nType 'CONFIRM LIVE' to proceed: ").strip()
        if confirm != "CONFIRM LIVE":
            print("Aborted.")
            sys.exit(0)

    orchestrator = BITGOTOrchestrator(CFG)
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        pass


def main():
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\n👋 BITGOT stopped by user")


if __name__ == "__main__":
    main()
