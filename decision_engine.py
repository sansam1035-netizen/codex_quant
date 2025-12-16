# decision_engine_v3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import math


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class RegimeParams:
    spread_cap: float
    alpha_min: float
    ev1_floor: float
    win1_floor: float
    cvar1_floor: float
    psl_max: float
    event_cvar_r_floor: float
    lev_max: float
    cap_frac: float
    k_lev: float = 6.0  # real money: 3~5 권장


REGIME_PARAMS: Dict[str, RegimeParams] = {
    "bull": RegimeParams(
        spread_cap=0.0020, alpha_min=0.25,
        ev1_floor=0.0002, win1_floor=0.50, cvar1_floor=-0.010,
        psl_max=0.42, event_cvar_r_floor=-1.20,
        lev_max=20.0, cap_frac=0.25
    ),
    "bear": RegimeParams(
        spread_cap=0.0020, alpha_min=0.25,
        ev1_floor=0.0002, win1_floor=0.50, cvar1_floor=-0.011,
        psl_max=0.40, event_cvar_r_floor=-1.15,
        lev_max=18.0, cap_frac=0.22
    ),
    "chop": RegimeParams(
        spread_cap=0.0012, alpha_min=0.40,
        ev1_floor=0.0005, win1_floor=0.52, cvar1_floor=-0.008,
        psl_max=0.35, event_cvar_r_floor=-1.05,
        lev_max=10.0, cap_frac=0.10
    ),
    "volatile": RegimeParams(
        spread_cap=0.0008, alpha_min=0.55,
        ev1_floor=0.0008, win1_floor=0.53, cvar1_floor=-0.007,
        psl_max=0.32, event_cvar_r_floor=-0.95,
        lev_max=8.0, cap_frac=0.08
    ),
}


# -------------------------
# Lightweight OFI normalizer (no history storage)
# -------------------------

class EmaScale:
    """
    Very simple stable normalizer:
      mean EMA + abs(x) EMA as scale proxy
    Not a true std-dev; robust enough for gating/residual direction.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.mean = 0.0
        self.absmean = 1e-6
        self.inited = False

    def update(self, x: float) -> None:
        x = float(x)
        if not self.inited:
            self.mean = x
            self.absmean = abs(x) + 1e-6
            self.inited = True
            return
        a = self.alpha
        self.mean = (1 - a) * self.mean + a * x
        self.absmean = (1 - a) * self.absmean + a * (abs(x) + 1e-6)

    def z(self, x: float) -> float:
        return float((float(x) - self.mean) / self.absmean)


# -------------------------
# Public API
# -------------------------

@dataclass
class Decision:
    action: str  # LONG/SHORT/WAIT
    leverage: float = 0.0
    size_frac: float = 0.0
    reason: str = ""
    meta: Dict[str, Any] = None


class DecisionEngineV3:
    """
    New decision engine designed to be embedded with minimal coupling.

    Required inputs (in tick dict):
      - regime: str (bull/bear/chop/volatile)
      - session: str
      - bias: float in [-1,+1] (direction bias)
      - ofi: float
      - spread_pct: float
      - liquidity_score: float
      - sigma: float (short vol)
      - sl_pct: float
      - tp_pct: float
      - size_frac_suggested: float (optional)

    Required callbacks:
      - mc_runner(side: str, leverage: float, **mc_kwargs) -> dict
        must return keys: ev, win, cvar, event_p_sl, event_cvar_r, meta(optional)
        ev/win/cvar are net-% (already costs & leverage applied per your MC)
    """

    def __init__(self, mc_runner, ofi_alpha: float = 0.01):
        self.mc_runner = mc_runner
        self.ofi_norm = EmaScale(alpha=ofi_alpha)

    def decide(self, tick: Dict[str, Any]) -> Decision:
        meta: Dict[str, Any] = {}

        regime = str(tick.get("regime", "chop"))
        p = REGIME_PARAMS.get(regime, REGIME_PARAMS["chop"])

        spread_pct = float(tick.get("spread_pct", 0.0002))
        liq = float(tick.get("liquidity_score", 0.0))
        bias = float(tick.get("bias", 0.0))
        ofi = float(tick.get("ofi", 0.0))
        sigma = float(tick.get("sigma", 0.0))

        # -------------------------
        # Stage-1: Execution gate
        # -------------------------
        if spread_pct > p.spread_cap:
            return Decision("WAIT", reason="exec_gate_spread",
                            meta={"spread_pct": spread_pct, "cap": p.spread_cap, "regime": regime})

        # (optional) add liq floor if you have calibrated meaning
        # if liq < p.liq_floor: ...

        # -------------------------
        # Stage-2: Alpha candidate (bias + OFI)
        # -------------------------
        self.ofi_norm.update(ofi)
        ofi_z = self.ofi_norm.z(ofi)
        alpha_long = 0.55 * bias + 0.35 * ofi_z
        alpha_short = -0.55 * bias - 0.35 * ofi_z
        alpha_max = max(alpha_long, alpha_short)

        meta.update({
            "bias": bias, "ofi": ofi, "ofi_z": ofi_z,
            "alpha_long": alpha_long, "alpha_short": alpha_short,
            "alpha_min": p.alpha_min, "alpha_max": alpha_max,
            "regime": regime
        })

        if alpha_max < p.alpha_min:
            return Decision("WAIT", reason="alpha_gate", meta=meta)

        side = "LONG" if alpha_long >= alpha_short else "SHORT"
        meta["alpha_side"] = side

        # -------------------------
        # Stage-3: MC validate at leverage=1
        # -------------------------
        mc1 = self.mc_runner(
            side=side,
            leverage=1.0,
            **tick
        )
        ev1 = float(mc1.get("ev", -1e9))
        win1 = float(mc1.get("win", 0.0))
        cvar1 = float(mc1.get("cvar", -1e9))
        p_sl = float(mc1.get("event_p_sl", 1.0))
        event_cvar_r = float(mc1.get("event_cvar_r", -999.0))

        meta.update({
            "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
            "event_p_sl": p_sl, "event_cvar_r": event_cvar_r
        })

        if ev1 < p.ev1_floor:
            return Decision("WAIT", reason="mc1_ev_gate", meta=meta)
        if win1 < p.win1_floor:
            return Decision("WAIT", reason="mc1_win_gate", meta=meta)
        if cvar1 < p.cvar1_floor:
            return Decision("WAIT", reason="mc1_cvar_gate", meta=meta)
        if p_sl > p.psl_max:
            return Decision("WAIT", reason="mc1_psl_gate", meta=meta)
        if event_cvar_r < p.event_cvar_r_floor:
            return Decision("WAIT", reason="mc1_event_cvar_gate", meta=meta)

        # -------------------------
        # Stage-4: Risk-based leverage & sizing
        # -------------------------
        sl_pct = float(tick.get("sl_pct", 0.002))
        # Convert event tail from R to % to mix with CVaR(%)
        event_cvar_pct = event_cvar_r * sl_pct

        # Best-effort: if MC exposes slippage in meta, use it; else use tick estimate
        mc1_meta = mc1.get("meta", {}) or {}
        slippage_dyn = float(mc1_meta.get("slippage_dyn", tick.get("slippage_dyn", 0.0)))

        risk = max(abs(cvar1), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_dyn + 0.5 * sigma
        lev_raw = (max(ev1, 0.0) / (risk + 1e-9)) * p.k_lev
        lev = max(1.0, min(lev_raw, p.lev_max))

        cap_frac = p.cap_frac
        size_suggested = float(tick.get("size_frac_suggested", cap_frac))
        size_frac = min(size_suggested, cap_frac)

        meta.update({
            "risk": risk, "lev_raw": lev_raw, "lev": lev,
            "slippage_dyn": slippage_dyn, "cap_frac": cap_frac, "size_frac": size_frac
        })

        # Optional: MC at final leverage for reporting only
        mc_final = self.mc_runner(
            side=side,
            leverage=lev,
            **tick
        )
        meta.update({
            "EV_final": float(mc_final.get("ev", float("nan"))),
            "Win_final": float(mc_final.get("win", float("nan"))),
            "CVaR_final": float(mc_final.get("cvar", float("nan"))),
            "event_p_sl_final": float(mc_final.get("event_p_sl", float("nan"))),
            "event_cvar_r_final": float(mc_final.get("event_cvar_r", float("nan"))),
        })

        return Decision(action=side, leverage=lev, size_frac=size_frac, reason="v3_enter", meta=meta)
