# engines/mc_engine.py
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from engines.base import BaseEngine
from regime import adjust_mu_sigma

from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

# NOTE:
# - meta에 멀티호라이즌(예: 60s/180s) 보조 필드를 넣어
#   상위 오케스트레이터에서 mid_boost/필터를 더 세밀하게 적용할 수 있게 한다.

# optional JAX acceleration
try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    from jax import random, lax  # type: ignore
    from jax import random as jrand  # type: ignore
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    random = None
    lax = None
    jrand = None

_JAX_OK = jax is not None
def _jax_mc_device():
    """Return a device to run MC kernels on.
    On Apple Metal backend, some ops/paths may crash with:
      UNIMPLEMENTED: default_memory_space is not supported.
    We keep using JAX, but run these kernels on CPU by default.
    Override with env JAX_MC_DEVICE=default to force default backend.
    """
    if jax is None:
        return None
    try:
        pref = os.environ.get("JAX_MC_DEVICE", "").strip().lower()
        if pref in ("default", "metal", "gpu"):
            return None

        # auto: if default backend is metal, prefer cpu for stability
        backend = None
        try:
            backend = jax.default_backend()
        except Exception:
            backend = None

        if pref in ("cpu",) or backend == "metal":
            devs = jax.devices("cpu")
            if devs:
                return devs[0]
    except Exception:
        return None
    return None


# -----------------------------
# CVaR Estimation (Ensemble)
# -----------------------------
def _cvar_empirical(pnl: np.ndarray, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(pnl, dtype=np.float64))
    k = max(1, int(alpha * len(x)))
    return float(x[:k].mean())


def _cvar_bootstrap(pnl: np.ndarray, alpha: float = 0.05, n_boot: int = 40, sample_frac: float = 0.7, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(pnl, dtype=np.float64)
    n = len(x)
    m = max(30, int(n * sample_frac))
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=m, replace=True)
        vals.append(_cvar_empirical(samp, alpha))
    return float(np.median(vals))


def _cvar_tail_inflate(pnl: np.ndarray, alpha: float = 0.05, inflate: float = 1.15) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    var = float(np.quantile(x, alpha))
    tail = x[x <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    if tail.size < 100:
        cvar *= float(inflate)
    return float(cvar)


def cvar_ensemble(pnl: Sequence[float], alpha: float = 0.05) -> float:
    x = np.asarray(pnl, dtype=np.float64)
    if x.size < 50:
        return float(_cvar_empirical(x, alpha))
    a = _cvar_empirical(x, alpha)
    b = _cvar_bootstrap(x, alpha)
    c = _cvar_tail_inflate(x, alpha)
    return float(0.60 * b + 0.25 * a + 0.15 * c)


# -----------------------------
# JAX helpers (optional)
# -----------------------------


def _cvar_jnp(x: "jnp.ndarray", alpha: float) -> "jnp.ndarray":  # type: ignore[name-defined]
    xs = jnp.sort(x)  # type: ignore[attr-defined]
    k = jnp.asarray(xs.shape[0] * alpha, dtype=jnp.int32)  # type: ignore[attr-defined]
    k = jnp.maximum(1, k)  # type: ignore[attr-defined]
    return jnp.mean(xs[:k])  # type: ignore[attr-defined]


def _sample_noise(key, shape, dist="gaussian", df=6.0, boot=None):
    if dist == "bootstrap" and boot is not None and boot.size > 16:
        idx = random.randint(key, shape, 0, boot.shape[0])  # type: ignore[attr-defined]
        return boot[idx]
    if dist == "student_t":
        k1, k2 = random.split(key)  # type: ignore[attr-defined]
        z = random.normal(k1, shape)  # type: ignore[attr-defined]
        u = 2.0 * random.gamma(k2, df / 2.0, shape)  # type: ignore[attr-defined]
        return z / jnp.sqrt(u / df)  # type: ignore[attr-defined]
    return random.normal(key, shape)  # type: ignore[attr-defined]


def mc_first_passage_tp_sl_jax(
    s0: float,
    tp_pct: float,
    sl_pct: float,
    mu: float,
    sigma: float,
    dt: float,
    max_steps: int,
    n_paths: int,
    seed: int,
    dist: str = "gaussian",
    df: float = 6.0,
    boot_rets: np.ndarray | None = None,
    cvar_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    JAX 가속 first-passage MC (TP/SL/timeout)
    """
    if jax is None or tp_pct <= 0 or sl_pct <= 0 or sigma <= 0:
        return {}

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    key = random.PRNGKey(seed & 0xFFFFFFFF)  # type: ignore[attr-defined]
    eps = _sample_noise(
        key,
        (n_paths, max_steps),
        dist=dist,
        df=df,
        boot=None if boot_rets is None else jnp.asarray(boot_rets),  # type: ignore[attr-defined]
    )

    log_inc = drift + vol * eps
    tp_price = s0 * (1.0 + tp_pct)
    sl_price = s0 * (1.0 - sl_pct)

    alive = jnp.ones(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_tp = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    hit_sl = jnp.zeros(n_paths, dtype=bool)  # type: ignore[attr-defined]
    t_hit = -jnp.ones(n_paths, dtype=jnp.int32)  # type: ignore[attr-defined]
    logp = jnp.zeros(n_paths)  # type: ignore[attr-defined]

    def step(carry, t):
        logp, alive, hit_tp, hit_sl, t_hit = carry
        logp2 = logp + log_inc[:, t]
        price = s0 * jnp.exp(logp2)  # type: ignore[attr-defined]
        tp_now = alive & (price >= tp_price)
        sl_now = alive & (price <= sl_price)
        hit = tp_now | sl_now
        t_hit = jnp.where(hit & (t_hit < 0), t, t_hit)  # type: ignore[attr-defined]
        hit_tp = hit_tp | tp_now
        hit_sl = hit_sl | sl_now
        alive = alive & (~hit)
        return (logp2, alive, hit_tp, hit_sl, t_hit), None

    (logp, alive, hit_tp, hit_sl, t_hit), _ = lax.scan(  # type: ignore[attr-defined]
        step,
        (logp, alive, hit_tp, hit_sl, t_hit),
        jnp.arange(max_steps),  # type: ignore[attr-defined]
    )

    p_tp = jnp.mean(hit_tp)  # type: ignore[attr-defined]
    p_sl = jnp.mean(hit_sl)  # type: ignore[attr-defined]
    p_to = jnp.mean(alive)  # type: ignore[attr-defined]

    r_tp = tp_pct / sl_pct
    r = jnp.where(hit_tp, r_tp, jnp.where(hit_sl, -1.0, 0.0))  # type: ignore[attr-defined]

    ev_r = jnp.mean(r)  # type: ignore[attr-defined]
    cvar_r = _cvar_jnp(r, cvar_alpha)

    t_vals = jnp.where(t_hit >= 0, t_hit.astype(jnp.float32), jnp.nan)  # type: ignore[attr-defined]

    p_tp_f = float(p_tp)
    p_sl_f = float(p_sl)
    p_to_f = float(p_to)
    prob_sum = p_tp_f + p_sl_f + p_to_f
    if abs(prob_sum - 1.0) > 1e-3 and prob_sum > 0:
        p_tp_f /= prob_sum
        p_sl_f /= prob_sum
        p_to_f = max(0.0, 1.0 - p_tp_f - p_sl_f)

    t_median = float(jnp.nanmedian(t_vals))  # type: ignore[attr-defined]
    t_mean = float(jnp.nanmean(t_vals))  # type: ignore[attr-defined]

    return {
        "event_p_tp": p_tp_f,
        "event_p_sl": p_sl_f,
        "event_p_timeout": p_to_f,
        "event_ev_r": float(ev_r),
        "event_cvar_r": float(cvar_r),
        "event_t_median": t_median if math.isfinite(t_median) else None,
        "event_t_mean": t_mean if math.isfinite(t_mean) else None,
    }


# -----------------------------
# Helpers
# -----------------------------
def ema(values: Sequence[float], period: int) -> Optional[float]:
    if values is None or len(values) < 2:
        return None
    v = np.asarray(values, dtype=np.float64)
    period = max(2, int(period))
    alpha = 2.0 / (period + 1.0)
    e = float(v[0])
    for x in v[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return float(e)


@dataclass
class MCParams:
    min_win: float
    profit_target: float
    ofi_weight: float
    max_kelly: float
    cvar_alpha: float
    cvar_scale: float
    n_paths: int


DEFAULT_PARAMS = {
    "bull": MCParams(min_win=0.55, profit_target=0.0012, ofi_weight=0.0015, max_kelly=0.25, cvar_alpha=0.05, cvar_scale=6.0, n_paths=16000),
    "bear": MCParams(min_win=0.57, profit_target=0.0012, ofi_weight=0.0018, max_kelly=0.20, cvar_alpha=0.05, cvar_scale=7.0, n_paths=16000),
    "chop": MCParams(min_win=0.60, profit_target=0.0010, ofi_weight=0.0022, max_kelly=0.10, cvar_alpha=0.05, cvar_scale=8.0, n_paths=16000),
}


class MonteCarloEngine(BaseEngine):
    """
    - ctx에서 regime_params를 받으면 그것을 우선 사용
    - 아니면 DEFAULT_PARAMS[regime]
    - 시뮬레이션은 numpy로 결정론 seed 사용(튜닝 안정)
    """
    name = "mc_barrier"
    weight = 1.0

    def __init__(self):
        # horizons(초) - GPU 실행 가정, 더 촘촘하게 확장
        self.horizons = (15, 30, 60, 120, 180, 300, 600)
        self.dt = 1.0 / 31536000.0  # seconds/year
        # Bybit taker round-trip(0.06% * 2) 기준, 레버리지와 무관한 고정 비용
        self.fee_roundtrip_base = 0.0012
        self.slippage_perc = 0.0003
        # tail mode defaults
        self.default_tail_mode = "student_t"  # "gaussian" | "student_t" | "bootstrap"
        self.default_student_t_df = 6.0
        self._use_jax = True
        self._tail_mode = self.default_tail_mode
        self._student_t_df = self.default_student_t_df
        self._bootstrap_returns = None
        self._ofi_hist: Dict[Tuple[str, str], List[float]] = {}

    # -----------------------------
    # First-passage TP/SL Monte Carlo (event-based)
    # -----------------------------
    def mc_first_passage_tp_sl(
        self,
        s0: float,
        tp_pct: float,
        sl_pct: float,
        mu: float,
        sigma: float,
        dt: float,
        max_steps: int,
        n_paths: int,
        cvar_alpha: float = 0.05,
        timeout_mode: str = "flat",
        seed: Optional[int] = None,
        side: str = "LONG",
    ) -> Dict[str, Any]:
        tp_pct = float(tp_pct)
        sl_pct = float(sl_pct)
        if tp_pct <= 0 or sl_pct <= 0 or sigma <= 0 or s0 <= 0:
            return {
                "event_p_tp": None,
                "event_p_sl": None,
                "event_p_timeout": None,
                "event_ev_r": None,
                "event_cvar_r": None,
                "event_t_median": None,
                "event_t_mean": None,
            }

        rng = np.random.default_rng(seed)
        max_steps = int(max(1, max_steps))
        # -----------------------------
        # Direction handling
        # LONG: 그대로, SHORT: log-return 반전
        # -----------------------------
        direction = 1.0
        if str(side).upper() == "SHORT":
            direction = -1.0

        drift = direction * (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices_np: np.ndarray
        if use_jax:
            cpu_dev = _jax_mc_device()
            try:
                if cpu_dev is None:
                    key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                    key, z_j = self._sample_increments_jax(
                        key,
                        (int(n_paths), int(max_steps)),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                    if z_j is None:
                        use_jax = False
                    else:
                        z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                        logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        prices_j = float(s0) * jnp.exp(direction * logret_j)  # type: ignore[attr-defined]
                        prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    # Run on CPU device (keeps JAX, avoids Metal XLA memory-space crashes)
                    with jax.default_device(cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed or 0) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                        key, z_j = self._sample_increments_jax(
                            key,
                            (int(n_paths), int(max_steps)),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                        )
                        if z_j is None:
                            use_jax = False
                        else:
                            z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                            logret_j = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            prices_j = float(s0) * jnp.exp(direction * logret_j)  # type: ignore[attr-defined]
                            prices_np = np.asarray(jax.device_get(prices_j), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                # Any JAX/XLA backend failure -> fall back to NumPy path simulation
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            steps = drift + diffusion * z
            log_prices = np.cumsum(steps, axis=1) + math.log(s0)
            prices_np = np.exp(direction * log_prices)

        if str(side).upper() == "SHORT":
            tp_level = s0 * (1.0 - tp_pct)
            sl_level = s0 * (1.0 + sl_pct)
        else:
            tp_level = s0 * (1.0 + tp_pct)
            sl_level = s0 * (1.0 - sl_pct)

        hit_tp = prices_np >= tp_level
        hit_sl = prices_np <= sl_level

        tp_hit_idx = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1) + 1, max_steps + 1)
        sl_hit_idx = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1) + 1, max_steps + 1)

        first_hit_idx = np.minimum(tp_hit_idx, sl_hit_idx)
        hit_type = np.full(n_paths, "timeout", dtype=object)
        hit_type[(tp_hit_idx < sl_hit_idx) & (tp_hit_idx <= max_steps)] = "tp"
        hit_type[(sl_hit_idx < tp_hit_idx) & (sl_hit_idx <= max_steps)] = "sl"

        tp_R = float(tp_pct / sl_pct)
        returns_r = np.zeros(n_paths, dtype=np.float64)
        returns_r[hit_type == "tp"] = tp_R
        returns_r[hit_type == "sl"] = -1.0
        if timeout_mode in ("mark_to_market", "mtm"):
            # mark-to-market: use end-of-horizon return
            end_prices = prices_np[:, -1]
            returns_r[hit_type == "timeout"] = (end_prices[hit_type == "timeout"] - s0) / (s0 * sl_pct)

        event_p_tp = float(np.mean(hit_type == "tp"))
        event_p_sl = float(np.mean(hit_type == "sl"))
        event_p_timeout = float(np.mean(hit_type == "timeout"))
        event_ev_r = float(np.mean(returns_r))
        event_cvar_r = float(cvar_ensemble(returns_r, alpha=cvar_alpha))

        hit_mask = (hit_type == "tp") | (hit_type == "sl")
        hit_times = first_hit_idx[hit_mask]
        event_t_median = float(np.median(hit_times)) if hit_times.size > 0 else None
        event_t_mean = float(np.mean(hit_times)) if hit_times.size > 0 else None

        # sanity check
        prob_sum = event_p_tp + event_p_sl + event_p_timeout
        if abs(prob_sum - 1.0) > 1e-3:
            # normalize softly
            event_p_tp /= prob_sum
            event_p_sl /= prob_sum
            event_p_timeout = max(0.0, 1.0 - event_p_tp - event_p_sl)

        return {
            "event_p_tp": event_p_tp,
            "event_p_sl": event_p_sl,
            "event_p_timeout": event_p_timeout,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_t_median": event_t_median,
            "event_t_mean": event_t_mean,
        }

    @staticmethod
    def _annualize(mu_bar: float, sigma_bar: float, bar_seconds: float) -> Tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = float(mu_bar) * bars_per_year
        sigma_ann = float(sigma_bar) * math.sqrt(bars_per_year)
        return float(mu_base), float(max(sigma_ann, 1e-6))

    @staticmethod
    def _trend_direction(price: float, closes: Sequence[float]) -> int:
        # EMA200 없으면 EMA50/20로 대체
        if closes is None or len(closes) < 30:
            return 1
        p = float(price)
        e_slow = ema(closes, 200) if len(closes) >= 200 else ema(closes, min(50, len(closes)))
        if e_slow is None:
            return 1
        return 1 if p >= float(e_slow) else -1

    def _get_params(self, regime: str, ctx: Dict[str, Any]) -> MCParams:
        rp = ctx.get("regime_params")
        if isinstance(rp, dict):
            # dict → MCParams로 안전 변환
            base = DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])
            return MCParams(
                min_win=float(rp.get("min_win", base.min_win)),
                profit_target=float(rp.get("profit_target", base.profit_target)),
                ofi_weight=float(rp.get("ofi_weight", base.ofi_weight)),
                max_kelly=float(rp.get("max_kelly", base.max_kelly)),
                cvar_alpha=float(rp.get("cvar_alpha", base.cvar_alpha)),
                cvar_scale=float(rp.get("cvar_scale", base.cvar_scale)),
                n_paths=int(rp.get("n_paths", base.n_paths)),
            )
        return DEFAULT_PARAMS.get(regime, DEFAULT_PARAMS["chop"])

    # -----------------------------
    # Regime clustering (lightweight K-Means)
    # -----------------------------
    @staticmethod
    def _cluster_regime(closes: Sequence[float]) -> str:
        if closes is None or len(closes) < 40:
            return "chop"
        x = np.asarray(closes, dtype=np.float64)
        rets = np.diff(np.log(x))
        if rets.size < 30:
            return "chop"
        # 특징: 단기 추세, 변동성
        slope = float(x[-1] - x[-10]) / max(1e-6, float(x[-10]))
        vol = float(rets[-30:].std())
        feats = np.array([[slope, vol]], dtype=np.float64)
        # 초기 중심 (bear/chop/bull 가정)
        centers = np.array([
            [-0.002, vol * 1.2],
            [0.0, vol],
            [0.002, max(vol * 0.8, 1e-6)]
        ], dtype=np.float64)
        # 미니 k-means 3회
        for _ in range(3):
            d = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            for k in range(3):
                mask = labels == k
                if mask.any():
                    centers[k] = feats[mask].mean(axis=0)
        label = int(labels[0])
        if label == 0:
            return "bear"
        if label == 2:
            return "bull"
        return "volatile" if vol > 0.01 else "chop"

    # -----------------------------
    # Slippage model
    # -----------------------------
    def _estimate_slippage(self, leverage: float, sigma: float, liq_score: float, ofi_z_abs: float = 0.0) -> float:
        base = self.slippage_perc
        vol_term = 1.0 + float(sigma) * 0.5
        liq_term = 1.0 if liq_score <= 0 else min(2.0, 1.0 + 1.0 / max(liq_score, 1.0))
        lev_term = max(1.0, abs(leverage) / 5.0)
        adv_k = 1.0 + 0.6 * min(2.0, max(0.0, ofi_z_abs))
        return base * vol_term * liq_term * lev_term * adv_k

    # -----------------------------
    # Tail samplers
    # -----------------------------
    def _sample_increments_np(self, rng: np.random.Generator, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            idx = rng.integers(0, bootstrap_returns.size, size=shape)
            return bootstrap_returns[idx].astype(np.float64)
        if mode == "student_t":
            z = rng.standard_t(df=df, size=shape).astype(np.float64)
            if df > 2:
                z = z / np.sqrt(df / (df - 2.0))
            return z
        return rng.standard_normal(size=shape).astype(np.float64)

    def _sample_increments_jax(self, key, shape, *, mode: str, df: float, bootstrap_returns: Optional[np.ndarray]):
        if jrand is None:
            return key, None
        if mode == "bootstrap" and bootstrap_returns is not None and bootstrap_returns.size >= 16:
            br = jnp.asarray(bootstrap_returns, dtype=jnp.float32)  # type: ignore[attr-defined]
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            idx = jrand.randint(k1, shape=shape, minval=0, maxval=br.shape[0])  # type: ignore[attr-defined]
            return key, br[idx]
        if mode == "student_t":
            key, k1 = jrand.split(key)  # type: ignore[attr-defined]
            z = jrand.t(k1, df=df, shape=shape)  # type: ignore[attr-defined]
            if df > 2:
                z = z / jnp.sqrt(df / (df - 2.0))  # type: ignore[attr-defined]
            return key, z
        key, k1 = jrand.split(key)  # type: ignore[attr-defined]
        return key, jrand.normal(k1, shape=shape)  # type: ignore[attr-defined]

    def simulate_paths_netpnl(
        self,
        seed: int,
        s0: float,
        mu: float,
        sigma: float,
        direction: int,
        leverage: float,
        n_paths: int,
        horizons: Sequence[int],
        dt: float,
        fee_roundtrip: float,
    ) -> Dict[int, np.ndarray]:
        """
        horizon별 net_pnl paths 반환
        """
        rng = np.random.default_rng(seed)
        max_steps = int(max(horizons))
        drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion = sigma * math.sqrt(dt)

        mode = str(getattr(self, "_tail_mode", self.default_tail_mode))
        df = float(getattr(self, "_student_t_df", self.default_student_t_df))
        br = getattr(self, "_bootstrap_returns", None)
        use_jax = bool(getattr(self, "_use_jax", True)) and _JAX_OK

        prices: np.ndarray
        if use_jax:
            cpu_dev = _jax_mc_device()
            try:
                if cpu_dev is None:
                    key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                    key, z_j = self._sample_increments_jax(
                        key,
                        (int(n_paths), int(max_steps)),
                        mode=mode,
                        df=df,
                        bootstrap_returns=br,
                    )
                    if z_j is None:
                        use_jax = False
                    else:
                        z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                        logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                        prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                        prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
                else:
                    with jax.default_device(cpu_dev):  # type: ignore[attr-defined]
                        key = jrand.PRNGKey(int(seed) & 0xFFFFFFFF)  # type: ignore[attr-defined]
                        key, z_j = self._sample_increments_jax(
                            key,
                            (int(n_paths), int(max_steps)),
                            mode=mode,
                            df=df,
                            bootstrap_returns=br,
                        )
                        if z_j is None:
                            use_jax = False
                        else:
                            z_j = z_j.astype(jnp.float32)  # type: ignore[attr-defined]
                            logret = jnp.cumsum((drift + diffusion * z_j), axis=1)  # type: ignore[attr-defined]
                            prices = float(s0) * jnp.exp(logret)  # type: ignore[attr-defined]
                            prices = np.asarray(jax.device_get(prices), dtype=np.float64)  # type: ignore[attr-defined]
            except Exception:
                use_jax = False


        if not use_jax:
            z = self._sample_increments_np(rng, (n_paths, max_steps), mode=mode, df=df, bootstrap_returns=br)
            logret = np.cumsum(drift + diffusion * z, axis=1)
            prices = s0 * np.exp(logret)

        out = {}
        for h in horizons:
            idx = int(h) - 1
            tp = prices[:, idx]
            gross = direction * (tp - s0) / s0 * float(leverage)
            net = gross - fee_roundtrip
            out[int(h)] = net.astype(np.float64)
        return out

    def evaluate_entry_metrics(self, ctx: Dict[str, Any], params: MCParams, seed: int) -> Dict[str, Any]:
        """
        튜너가 과거 ctx로 candidate 파라미터 평가할 때도 사용.
        """
        def _s(val, default=0.0) -> float:
            try:
                if val is None:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)

        symbol = str(ctx.get("symbol", ""))
        price = _s(ctx.get("price"), 0.0)
        mu_base = ctx.get("mu_base")
        sigma = ctx.get("sigma")
        closes = ctx.get("closes")
        liq_score = _s(ctx.get("liquidity_score"), 1.0)
        bar_seconds = _s(ctx.get("bar_seconds", 60.0), 60.0)
        # tail mode plumbing
        self._use_jax = bool(ctx.get("use_jax", True))
        self._tail_mode = str(ctx.get("tail_mode", self.default_tail_mode))
        self._student_t_df = _s(ctx.get("student_t_df", self.default_student_t_df), self.default_student_t_df)
        br = ctx.get("bootstrap_returns")
        if br is not None:
            try:
                self._bootstrap_returns = np.asarray(br, dtype=np.float64)
            except Exception:
                self._bootstrap_returns = None
        else:
            if self._tail_mode == "bootstrap" and closes is not None and len(closes) >= 64:
                x = np.asarray(closes, dtype=np.float64)
                rets = np.diff(np.log(np.maximum(x, 1e-12)))
                self._bootstrap_returns = rets[-512:].astype(np.float64) if rets.size >= 32 else None
            else:
                self._bootstrap_returns = None
        regime_ctx = str(ctx.get("regime", "chop"))

        if (mu_base is None or sigma is None) and closes is not None and len(closes) >= 10:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            if rets.size >= 5:
                mu_bar = float(rets.mean())
                sigma_bar = float(rets.std())
                mu_base, sigma = self._annualize(mu_bar, sigma_bar, bar_seconds=bar_seconds)

        # 다중 지평 블렌딩으로 추정치 안정화
        if closes is not None and len(closes) >= 30:
            rets = np.diff(np.log(np.asarray(closes, dtype=np.float64)))
            windows = [30, 90, 180]
            mu_blend = []
            sigma_blend = []
            for w in windows:
                if rets.size >= w:
                    rw = rets[-w:]
                    mu_blend.append(float(rw.mean()))
                    sigma_blend.append(float(rw.std()))
            if mu_blend and sigma_blend:
                mu_bar_mix = float(np.mean(mu_blend))
                sigma_bar_mix = float(np.mean(sigma_blend))
                mu_mix, sigma_mix = self._annualize(mu_bar_mix, sigma_bar_mix, bar_seconds=bar_seconds)
                if mu_base is None:
                    mu_base = mu_mix
                else:
                    mu_base = 0.6 * float(mu_base) + 0.4 * float(mu_mix)
                if sigma is None:
                    sigma = sigma_mix
                else:
                    sigma = 0.5 * float(sigma) + 0.5 * float(sigma_mix)

        if mu_base is None or sigma is None or price <= 0:
            return {"can_enter": False, "ev": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": 1, "kelly": 0.0, "size_frac": 0.0}

        mu_base = float(mu_base)
        sigma = float(max(sigma, 1e-6))
        # 레짐 기반 기대수익/변동성 조정
        mu_base, sigma = adjust_mu_sigma(mu_base, sigma, regime_ctx)

        ofi_score = _s(ctx.get("ofi_score"), 0.0)
        direction = int(ctx.get("direction") or self._trend_direction(price, closes or []))
        regime_ctx_for_cluster = ctx.get("regime")
        if regime_ctx_for_cluster == "chop" and closes:
            regime_ctx_for_cluster = self._cluster_regime(closes)

        # OFI 반영(방향 반대면 더 강하게)
        ofi_impact = ofi_score * params.ofi_weight
        if (direction == 1 and ofi_score < 0) or (direction == -1 and ofi_score > 0):
            ofi_impact *= 2.0
        mu_adj = mu_base + ofi_impact

        leverage = _s(ctx.get("leverage", 5.0), 5.0)

        # OFI z-score by regime/session (abs) for slippage adverse factor
        key = (str(ctx.get("regime", "chop")), str(ctx.get("session", "OFF")))
        hist = self._ofi_hist.setdefault(key, [])
        hist.append(ofi_score)
        if len(hist) > 500:
            hist.pop(0)
        ofi_z_abs = 0.0
        if len(hist) >= 5:
            arr = np.asarray(hist, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())
            std = std if std > 1e-6 else 0.05
            ofi_z_abs = abs(ofi_score - mean) / std

        # 레버리지/변동성/유동성 기반 슬리피지 모델
        slippage_dyn = self._estimate_slippage(leverage, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        # 수수료는 레버리지와 무관하게 고정, 슬리피지는 노출(lev) 가중
        fee_rt = self.fee_roundtrip_base + slippage_dyn
        # gate용 baseline(lev=1)
        slippage_base = self._estimate_slippage(1.0, sigma, liq_score, ofi_z_abs=ofi_z_abs)
        fee_rt_base = self.fee_roundtrip_base + slippage_base
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # 2bp fallback
        try:
            spread_pct = float(spread_pct)
        except Exception:
            spread_pct = 0.0002
        expected_spread_cost = 0.5 * float(spread_pct) * 1.0  # adverse selection factor=1.0
        execution_cost = expected_spread_cost + slippage_dyn + self.fee_roundtrip_base

        net_by_h = self.simulate_paths_netpnl(
            seed=seed,
            s0=price,
            mu=mu_adj,
            sigma=sigma,
            direction=direction,
            leverage=leverage,
            n_paths=int(params.n_paths),
            horizons=self.horizons,
            dt=self.dt,
            fee_roundtrip=fee_rt,
        )
        net_by_h_base = self.simulate_paths_netpnl(
            seed=seed ^ 0xABCDEF,
            s0=price,
            mu=mu_adj,
            sigma=sigma,
            direction=direction,
            leverage=1.0,
            n_paths=int(params.n_paths),
            horizons=self.horizons,
            dt=self.dt,
            fee_roundtrip=fee_rt_base,
        )

        # horizon별 요약 + 가중 앙상블
        best_h = None
        best_ev = -1e18
        ev_list: List[float] = []
        win_list: List[float] = []
        cvar_list: List[float] = []
        h_list: List[int] = []
        mid_h = []
        for h, net in net_by_h.items():
            ev_h = float(net.mean())
            win_h = float((net > 0).mean())
            cvar_h = float(cvar_ensemble(net, alpha=params.cvar_alpha))
            ev_list.append(ev_h)
            win_list.append(win_h)
            cvar_list.append(cvar_h)
            h_list.append(int(h))
            if ev_h > best_ev:
                best_ev = ev_h
                best_h = h
            if h >= 180:
                mid_h.append(net)

        if not h_list:
            return {"can_enter": False, "ev": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        # exp-decay weights (half-life=120s)
        h_arr = np.asarray(h_list, dtype=np.float64)
        w = np.exp(-h_arr * math.log(2) / 120.0)
        w = w / np.sum(w)
        evs = np.asarray(ev_list, dtype=np.float64)
        wins = np.asarray(win_list, dtype=np.float64)
        cvars = np.asarray(cvar_list, dtype=np.float64)

        ev_agg = float(np.sum(w * evs))
        win_agg = float(np.sum(w * wins))
        cvar_agg = float(np.quantile(cvars, 0.25))
        horizon_weights = {int(h): float(w_i) for h, w_i in zip(h_list, w)}

        if best_h is None:
            return {"can_enter": False, "ev": 0.0, "win": 0.0, "cvar": 0.0, "best_h": 0, "direction": direction, "kelly": 0.0, "size_frac": 0.0}

        ev = ev_agg - float(execution_cost)
        win = win_agg
        cvar = cvar_agg
        # gate용(base lev=1)
        ev1 = float(ev_agg - (expected_spread_cost + slippage_base + self.fee_roundtrip_base))
        win1 = win_agg  # 승률은 레버리지와 무관
        cvar1 = cvar_agg
        ev_mid = None
        win_mid = None
        cvar_mid = None
        if mid_h:
            mid_concat = np.concatenate(mid_h, axis=0)
            ev_mid = float(mid_concat.mean())
            win_mid = float((mid_concat > 0).mean())
            cvar_mid = float(cvar_ensemble(mid_concat, alpha=params.cvar_alpha))

        # entry gating: 비용을 충분히 이길 때만, 승률/꼬리/중기 필터
        cost_floor = float(fee_rt)
        # 더 높은 EV 기준: 비용 + 추가 버퍼
        ev_floor = max(params.profit_target, cost_floor * 1.0 + 0.0008)
        win_floor = max(params.min_win - 0.02, 0.53)
        cvar_floor_abs = cost_floor * 3.0  # 요구: cvar_agg > -cvar_floor_abs

        can_enter = False
        if ev > ev_floor and win >= win_floor and cvar > -cvar_floor_abs:
            mid_ok = True
            if ev_mid is not None:
                mid_ok = ev_mid >= 0.0
            if win_mid is not None and mid_ok:
                mid_ok = win_mid >= 0.50
            can_enter = mid_ok

        # Kelly raw (EV / variance proxy)
        variance_proxy = float(sigma * sigma)
        kelly_raw = max(0.0, ev / max(variance_proxy, 1e-6))

        # CVaR 기반 축소 (레버리지 고려)
        leverage_penalty = max(1.0, abs(leverage) / 5.0)
        cvar_penalty = max(0.05, 1.0 - params.cvar_scale * abs(cvar) * leverage_penalty)

        # 고레버리지일수록 Kelly 상한을 자동 축소
        kelly_cap = params.max_kelly / leverage_penalty

        kelly = min(kelly_raw * cvar_penalty, kelly_cap)

        confidence = float(win)  # hub confidence는 win 기반 유지
        size_frac = float(max(0.0, kelly * confidence))

        # Event-based MC (first passage TP/SL)
        tp_pct = float(max(params.profit_target, 0.0005))
        sl_pct = float(max(tp_pct * 0.8, 0.0008))
        if jax is not None:
            event_metrics = mc_first_passage_tp_sl_jax(
                s0=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                mu=mu_adj,
                sigma=sigma,
                dt=1.0,
                max_steps=int(max(self.horizons)),
                n_paths=int(params.n_paths),
                cvar_alpha=params.cvar_alpha,
                seed=seed,
            )
            if not event_metrics:
                event_metrics = self.mc_first_passage_tp_sl(
                    s0=price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    mu=mu_adj,
                    sigma=sigma,
                    dt=1.0,
                    max_steps=int(max(self.horizons)),
                    n_paths=int(params.n_paths),
                    cvar_alpha=params.cvar_alpha,
                    seed=seed,
                )
        else:
            event_metrics = self.mc_first_passage_tp_sl(
                s0=price,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                mu=mu_adj,
                sigma=sigma,
                dt=1.0,
                max_steps=int(max(self.horizons)),
                n_paths=int(params.n_paths),
                cvar_alpha=params.cvar_alpha,
                seed=seed,
            )

        # event EV/CVaR를 % 수익률 단위로 환산 (R * SL%)
        event_ev_pct = None
        event_cvar_pct = None
        try:
            if event_metrics.get("event_ev_r") is not None:
                event_ev_pct = float(event_metrics["event_ev_r"]) * sl_pct
        except Exception:
            event_ev_pct = None
        try:
            if event_metrics.get("event_cvar_r") is not None:
                event_cvar_pct = float(event_metrics["event_cvar_r"]) * sl_pct
        except Exception:
            event_cvar_pct = None

        return {
            "can_enter": bool(can_enter),
            "ev": ev,
            "win": win,
            "cvar": cvar,
            "best_h": int(best_h),
            "direction": int(direction),
            "kelly": float(kelly),
            "size_frac": float(size_frac),
            "ev1": ev1,
            "win1": win1,
            "cvar1": cvar1,
            "mu_adj": float(mu_adj),
            "fee_rt": float(fee_rt),
            "spread_pct": float(spread_pct),
            "expected_spread_cost": float(expected_spread_cost),
            "execution_cost": float(execution_cost),
            "ev_floor": float(ev_floor),
            "win_floor": float(win_floor),
            "cvar_floor": float(cvar_floor_abs),
            "ev_mid": ev_mid,
            "win_mid": win_mid,
            "cvar_mid": cvar_mid,
            "slippage_pct": float(slippage_dyn),
            "ev_by_horizon": [float(x) for x in ev_list],
            "win_by_horizon": [float(x) for x in win_list],
            "cvar_by_horizon": [float(x) for x in cvar_list],
            "horizon_seq": [int(h) for h in h_list],
            "event_p_tp": event_metrics.get("event_p_tp"),
            "event_p_sl": event_metrics.get("event_p_sl"),
            "event_p_timeout": event_metrics.get("event_p_timeout"),
            "event_ev_r": event_metrics.get("event_ev_r"),
            "event_cvar_r": event_metrics.get("event_cvar_r"),
            "event_ev_pct": event_ev_pct,
            "event_cvar_pct": event_cvar_pct,
            "event_t_median": event_metrics.get("event_t_median"),
            "event_t_mean": event_metrics.get("event_t_mean"),
            "horizon_weights": horizon_weights,
        }

    def decide(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(ctx.get("symbol", ""))
        price = float(ctx.get("price", 0.0))
        regime_ctx = str(ctx.get("regime", "chop"))
        params = self._get_params(regime_ctx, ctx)

        # deterministic seed for stability (per symbol, per minute-ish)
        seed = int((hash(symbol) ^ int(time.time() // 3)) & 0xFFFFFFFF)

        metrics = self.evaluate_entry_metrics(ctx, params, seed=seed)

        action = "WAIT"
        if metrics["can_enter"]:
            action = "LONG" if metrics["direction"] == 1 else "SHORT"

        best_desc = f"{metrics['best_h']}초" if metrics["best_h"] else "-"

        # 멀티호라이즌 보조 메타(예: 60s/180s) 추출/생성
        ev_60 = None
        win_60 = None
        cvar_60 = None
        ev_180 = metrics.get("ev_mid")
        win_180 = metrics.get("win_mid")
        cvar_180 = metrics.get("cvar_mid")
        ensemble_mid_boost = 1.0
        if win_180 is not None:
            try:
                ensemble_mid_boost = 0.85 + 0.6 * max(0.0, min(1.0, (win_180 - 0.52) / 0.18))
            except Exception:
                ensemble_mid_boost = 1.0

        return {
            "action": action,
            "ev": float(metrics["ev"]),
            "confidence": float(metrics["win"]),
            "reason": f"MC({best_desc}) {regime_ctx} EV {metrics['ev']*100:.2f}% Win {metrics['win']*100:.1f}% CVaR {metrics['cvar']*100:.2f}%",
            "meta": {
                "regime": regime_ctx,
                "best_horizon_desc": best_desc,
                "best_horizon_steps": int(metrics["best_h"]),
                "ev": float(metrics["ev"]),
                "ev1": float(metrics.get("ev1", metrics["ev"])),
                "win_rate": float(metrics["win"]),
                "win_rate1": float(metrics.get("win1", metrics["win"])),
                "cvar05": float(metrics["cvar"]),
                "cvar05_lev1": float(metrics.get("cvar1", metrics["cvar"])),
                "ev_mid": metrics.get("ev_mid"),
                "win_mid": metrics.get("win_mid"),
                "cvar_mid": metrics.get("cvar_mid"),
                "ev_by_horizon": metrics.get("ev_by_horizon"),
                "win_by_horizon": metrics.get("win_by_horizon"),
                "cvar_by_horizon": metrics.get("cvar_by_horizon"),
                "horizon_seq": metrics.get("horizon_seq"),
                "horizon_weights": metrics.get("horizon_weights"),
                "ev_60": ev_60,
                "win_60": win_60,
                "cvar_60": cvar_60,
                "ev_180": ev_180,
                "win_180": win_180,
                "cvar_180": cvar_180,
                "ensemble_mid_boost": ensemble_mid_boost,
                "ev_entry_threshold": metrics.get("ev_floor"),
                "win_entry_threshold": metrics.get("win_floor"),
                "cvar_entry_threshold": metrics.get("cvar_floor"),
                "kelly": float(metrics["kelly"]),
                "size_fraction": float(metrics["size_frac"]),
                "direction": int(metrics["direction"]),
                "mu_adjusted": float(metrics.get("mu_adj", 0.0)),
                "event_p_tp": metrics.get("event_p_tp"),
                "event_p_sl": metrics.get("event_p_sl"),
                "event_p_timeout": metrics.get("event_p_timeout"),
                "event_ev_r": metrics.get("event_ev_r"),
                "event_cvar_r": metrics.get("event_cvar_r"),
                "event_ev_pct": metrics.get("event_ev_pct"),
                "event_cvar_pct": metrics.get("event_cvar_pct"),
                "event_t_median": metrics.get("event_t_median"),
                "event_t_mean": metrics.get("event_t_mean"),
                "params": {
                    "min_win": params.min_win,
                    "profit_target": params.profit_target,
                    "ofi_weight": params.ofi_weight,
                    "max_kelly": params.max_kelly,
                    "cvar_alpha": params.cvar_alpha,
                    "cvar_scale": params.cvar_scale,
                    "n_paths": params.n_paths,
                },
            },
            # EngineHub가 그대로 받게
            "size_frac": float(metrics["size_frac"]),
        }
