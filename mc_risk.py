import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

def compute_cvar(pnl_samples, alpha=0.05):
    pnl = np.sort(np.asarray(pnl_samples))
    cutoff = int(len(pnl) * alpha)
    return float(np.mean(pnl[:cutoff])) if cutoff > 0 else 0.0


def kelly_fraction(p, b):
    q = 1 - p
    if b <= 0:
        return 0.0
    return max(0.0, min(1.0, (b*p - q) / b))


def kelly_with_cvar(win_rate, tp, sl, cvar):
    b = tp / sl if sl > 0 else 0.0
    raw = kelly_fraction(win_rate, b)

    # CVaR penalty
    penalty = min(1.0, abs(cvar) * 10)
    return max(0.0, raw * (1 - penalty))
from collections import deque

class PyramidTracker:
    def __init__(self, window=5):
        self.ev_hist = deque(maxlen=window)

    def update(self, ev):
        self.ev_hist.append(ev)

    def pyramid_factor(self):
        if len(self.ev_hist) < 3:
            return 1.0
        if all(self.ev_hist[i] < self.ev_hist[i+1] for i in range(len(self.ev_hist)-1)):
            return min(1.5, 1.0 + 0.1 * len(self.ev_hist))
        return 1.0


# -------------------------------------------------------------------
# Exit / Hold policy (MC event outputs -> real position decision)
# -------------------------------------------------------------------
@dataclass
class ExitPolicy:
    # --- tunable via WF ---
    min_event_ev_r: float = -0.0005     # event EV가 이보다 나쁘면 exit
    max_event_p_sl: float = 0.55        # SL 먼저 맞을 확률이 너무 크면 exit
    time_stop_mult: float = 2.2        # event_t_median * 배수 넘어가면 exit

    # --- fixed safety ---
    min_event_p_tp: float = 0.30        # TP 도달 확률이 너무 낮으면 exit(유예 후)
    grace_sec: int = 20                 # 진입 직후 흔들림 유예
    max_hold_sec: int = 600             # 절대 보유 제한(초)
    max_abs_event_cvar_r: float = 0.010 # |CVaR|이 크면 축소/청산


def should_exit_position(pos: Dict[str, Any], meta: Dict[str, Any], *, age_sec: float, policy: ExitPolicy) -> Tuple[bool, str]:
    """
    pos: orchestrator position dict
    meta: latest decision meta from MC engine (event_* fields expected)
    """
    def _f(x) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    ev_r = _f(meta.get("event_ev_r"))
    p_sl = _f(meta.get("event_p_sl"))
    p_tp = _f(meta.get("event_p_tp"))
    t_med = _f(meta.get("event_t_median"))
    cvar_r = _f(meta.get("event_cvar_r"))

    # 0) grace
    if age_sec < policy.grace_sec:
        return False, "grace"

    # 1) EV deteriorated
    if ev_r is not None and ev_r <= policy.min_event_ev_r:
        return True, f"event_ev_r<= {policy.min_event_ev_r:.5f}"

    # 2) SL-risk too high
    if p_sl is not None and p_sl >= policy.max_event_p_sl:
        return True, f"event_p_sl>= {policy.max_event_p_sl:.2f}"

    # 3) TP probability too low (after grace)
    if p_tp is not None and p_tp <= policy.min_event_p_tp:
        return True, f"event_p_tp<= {policy.min_event_p_tp:.2f}"

    # 4) CVaR ugly
    if cvar_r is not None and abs(cvar_r) >= policy.max_abs_event_cvar_r:
        return True, f"|event_cvar_r|>= {policy.max_abs_event_cvar_r:.4f}"

    # 5) time stop (event-based)
    if t_med is not None and t_med > 0 and age_sec >= policy.time_stop_mult * t_med:
        return True, f"time_stop>{policy.time_stop_mult:.1f}x_median"

    # 6) absolute max hold
    if age_sec >= policy.max_hold_sec:
        return True, "max_hold"

    return False, "hold"
