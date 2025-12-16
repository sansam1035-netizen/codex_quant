from __future__ import annotations
import asyncio
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional
from aiohttp import web
import ccxt.async_support as ccxt

from engines.engine_hub import EngineHub

# -------------------------------------------------------------------
# aiohttp 일부 macOS 환경에서 TCP keepalive 설정 시 OSError(22)가 날 수 있다.
# (dashboard 접속 불가 증상). best-effort로 무시 처리.
# -------------------------------------------------------------------
try:
    import aiohttp.tcp_helpers as _aiohttp_tcp_helpers

    _orig_tcp_keepalive = _aiohttp_tcp_helpers.tcp_keepalive

    def _tcp_keepalive_safe(transport):
        try:
            return _orig_tcp_keepalive(transport)
        except OSError:
            return None

    _aiohttp_tcp_helpers.tcp_keepalive = _tcp_keepalive_safe
except Exception:
    pass
from engines.mc_engine import mc_first_passage_tp_sl_jax
from engines.mc_risk import compute_cvar, kelly_with_cvar, PyramidTracker, ExitPolicy, should_exit_position
from regime import adjust_mu_sigma, time_regime, get_regime_mu_sigma
from engines.running_stats import RunningStats

PORT = 9999

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "DOGE/USDT:USDT",
    "AVAX/USDT:USDT", "DOT/USDT:USDT", "LINK/USDT:USDT"
]

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

# Prefer the actively edited template inside templates/, but keep fallbacks
# for legacy copies that might still exist.
_dashboard_candidates = [
    TEMPLATE_DIR / "dashboard_v2.html",
    TEMPLATE_DIR / "dashboard.html",
    BASE_DIR / "dashboard_v2.html",
    BASE_DIR / "dashboard.html",
]
DASHBOARD_FILE = next((p for p in _dashboard_candidates if p.exists()), None)
if DASHBOARD_FILE is None:
    raise FileNotFoundError("Dashboard template not found in templates/ or base directory.")

# ---- OHLCV settings
TIMEFRAME = "1m"
OHLCV_PRELOAD_LIMIT = 240   # 시작 시 한 번에 채울 캔들 수
OHLCV_REFRESH_LIMIT = 2     # 이후 갱신은 최소로
OHLCV_SLEEP_SEC = 30        # 더 촘촘한 1분봉 갱신 주기

# ---- Orderbook settings (대시보드 "Orderbook Ready" 해결)
ORDERBOOK_DEPTH = 5
ORDERBOOK_SLEEP_SEC = 2.0     # 심볼 전체를 한 바퀴 도는 주기(대략)

# ---- Networking / Retry (Bybit public endpoints are sensitive)
CCXT_TIMEOUT_MS = 20000
MAX_RETRY = 4
RETRY_BASE_SEC = 0.5
MAX_INFLIGHT_REQ = 1

# ---- Risk / Execution settings
ENABLE_LIVE_ORDERS = False          # 실제 주문 호출 토글
DEFAULT_LEVERAGE = 5.0
MAX_LEVERAGE = 50.0
DEFAULT_SIZE_FRAC = 0.10            # balance 대비 기본 진입 비중 (더 공격적)
MAX_POSITION_HOLD_SEC = 600         # 10분 이상 보유 시 강제 청산 (더 공격적)
POSITION_CAP_ENABLED = False        # 포지션 개수 제한 비활성화(무제한 진입)
EXPOSURE_CAP_ENABLED = True         # 노출 한도 사용
MAX_CONCURRENT_POSITIONS = 99999
MAX_NOTIONAL_EXPOSURE = 5.0         # 총 노출을 잔고 대비 500%까지 허용
REBALANCE_THRESHOLD_FRAC = 0.02     # 목표 노출 대비 2% 이상 차이 나면 리밸런싱(더 잦게)
EV_DROP_THRESHOLD = 0.0003          # EV 급락 exit 감지 임계
K_LEV = 4.0                         # 레버리지 스케일(실계좌는 3~5 권장)
EV_EXIT_FLOOR = {"bull": -0.0003, "bear": -0.0003, "chop": -0.0002, "volatile": -0.0002}
EV_DROP = {"bull": 0.0010, "bear": 0.0010, "chop": 0.0008, "volatile": 0.0008}
PSL_RISE = {"bull": 0.05, "bear": 0.05, "chop": 0.03, "volatile": 0.03}

# ---- Consensus / Spread settings
CONSENSUS_THRESHOLD = 1.0           # 합의 점수 임계값(가중 득표 합)
RSI_PERIOD = 14
RSI_LONG = 60.0
RSI_SHORT = 40.0
SPREAD_LOOKBACK = 60
SPREAD_Z_ENTRY = 2.0
SPREAD_Z_EXIT = 0.5
SPREAD_SIZE_FRAC = 0.02
SPREAD_HOLD_SEC = 600
SPREAD_ENABLED = True   # 스프레드 활성화
SPREAD_PAIRS = [
    ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ("SOL/USDT:USDT", "BNB/USDT:USDT"),
]
# 스프레드 상한 (entry gate)
SPREAD_PCT_MAX = 0.0005  # 0.05%

# Bybit USDT-Perp 기본 수수료(보수적으로 Taker 가정)
BYBIT_TAKER_FEE = 0.0006  # 0.06% per side
BYBIT_MAKER_FEE = 0.0001  # 0.01% per side

# EV auto-tuning (p95 of recent EVs)
EV_TUNE_WINDOW_SEC = 30 * 60   # 30 minutes
EV_TUNE_PCTL = 95
EV_TUNE_MIN_SAMPLES = 40
EV_ENTER_FLOOR_MIN = 0.0008   # 조금 완화
EV_ENTER_FLOOR_MAX = 0.0025   # 상한도 소폭 완화

# 엔트리 히스테리시스 / 쿨다운
COOLDOWN_SEC = 60
ENTRY_STREAK_MIN = 1
# cooldown presets used by _mark_exit_and_cooldown()
COOLDOWN_TP_SEC = 30
COOLDOWN_RISK_SEC = 120

def now_ms() -> int:
    return int(time.time() * 1000)


class LiveOrchestrator:
    def __init__(self, exchange):
        self.hub = EngineHub()
        self.exchange = exchange
        self._net_sem = asyncio.Semaphore(MAX_INFLIGHT_REQ)
        self._last_ok = {"tickers": 0, "ohlcv": {s: 0 for s in SYMBOLS}, "ob": {s: 0 for s in SYMBOLS}}

        self.clients = set()
        self.logs = deque(maxlen=300)

        self.balance = 10_000.0
        self.positions = {}  # sym -> position dict (demo/paper)
        self.leverage = DEFAULT_LEVERAGE
        self.max_leverage = MAX_LEVERAGE
        self.enable_orders = ENABLE_LIVE_ORDERS
        self.max_positions = MAX_CONCURRENT_POSITIONS
        self.max_notional_frac = MAX_NOTIONAL_EXPOSURE
        self.position_cap_enabled = POSITION_CAP_ENABLED
        self.exposure_cap_enabled = EXPOSURE_CAP_ENABLED
        self.default_size_frac = DEFAULT_SIZE_FRAC
        # 수수료 설정 (Bybit taker/maker)
        self.fee_taker = BYBIT_TAKER_FEE
        self.fee_maker = BYBIT_MAKER_FEE
        self.fee_mode = "taker"  # 기본 taker 가정
        self._decision_log_every = 10
        self._decision_cycle = 0
        self.spread_pairs = SPREAD_PAIRS
        self.spread_enabled = SPREAD_ENABLED
        self._last_actions = {}
        self._cooldown_until = {s: 0.0 for s in SYMBOLS}
        self._entry_streak = {s: 0 for s in SYMBOLS}
        self._last_exit_kind = {s: "NONE" for s in SYMBOLS}
        self._streak = {s: 0 for s in SYMBOLS}
        self._ev_tune_hist = {s: deque(maxlen=2000) for s in SYMBOLS}  # (ts, ev)
        self._ev_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._cvar_hist = {s: deque(maxlen=400) for s in SYMBOLS}
        self._ofi_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist: dict[tuple[str, str], deque] = {}
        self._ev_thr_ema: dict[tuple[str, str], float] = {}
        self._ev_thr_ema_ts: dict[tuple[str, str], int] = {}
        self._ev_drop_state: dict[str, dict] = {s: {"prev": None, "streak": 0} for s in SYMBOLS}
        self._spread_regime_hist: dict[tuple[str, str], deque] = {}
        self._liq_regime_hist: dict[tuple[str, str], deque] = {}
        self._ofi_resid_hist: dict[tuple[str, str], deque] = {}
        self._ev_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._cvar_regime_hist_rs: dict[tuple[str, str], deque] = {}
        self._ema_ev: dict[str, tuple[float, int]] = {}
        self._ema_psl: dict[str, tuple[float, int]] = {}
        self._exit_bad_ticks: dict[str, int] = {s: 0 for s in SYMBOLS}
        self._dyn_leverage = {s: self.leverage for s in SYMBOLS}
        self.trade_tape = deque(maxlen=20_000)
        self.eval_history = deque(maxlen=5_000)  # 예측 vs 실제 품질 평가용
        self.exit_policy = ExitPolicy(
            min_event_ev_r=-0.0005,
            max_event_p_sl=0.55,
            min_event_p_tp=0.30,
            grace_sec=20,
            max_hold_sec=600,
            time_stop_mult=2.2,
            max_abs_event_cvar_r=0.010,
        )
        self.mc_cache = {}  # (sym, side, regime, price_bucket) -> (ts, meta)
        self.mc_cache_ttl = 2.0  # seconds

        self.market = {s: {"price": None, "ts": 0} for s in SYMBOLS}

        # 1m close buffer (preload로 한 번에 채움)
        self.ohlcv_buffer = {s: deque(maxlen=OHLCV_PRELOAD_LIMIT) for s in SYMBOLS}

        # orderbook 상태(대시보드 표기용)
        self.orderbook = {s: {"ts": 0, "ready": False, "bids": [], "asks": []} for s in SYMBOLS}

        # OHLCV freshness / dedupe
        self._last_kline_ts = {s: 0 for s in SYMBOLS}      # 마지막 캔들 timestamp(ms)
        self._last_kline_ok_ms = {s: 0 for s in SYMBOLS}   # 마지막으로 buffer 갱신 성공한 시각(ms)
        self._preloaded = {s: False for s in SYMBOLS}

        self._last_feed_ok_ms = 0
        self._equity_history = deque(maxlen=20_000)
        # persistence
        self.state_dir = BASE_DIR / "state"
        self.state_dir.mkdir(exist_ok=True)
        self.state_files = {
            "equity": self.state_dir / "equity_history.json",
            "trade": self.state_dir / "trade_tape.json",
            "eval": self.state_dir / "eval_history.json",
            "positions": self.state_dir / "positions.json",
            "balance": self.state_dir / "balance.json",
        }
        self._last_state_persist_ms = 0
        self._load_persistent_state()
        # 러닝 통계
        self.stats = RunningStats(maxlen=5000)

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "INFO", "msg": text})
        print(text)

    def _log_err(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.logs.append({"time": ts, "level": "ERROR", "msg": text})
        print(text)

    async def _ccxt_call(self, label: str, fn, *args, **kwargs):
        """
        Best-effort CCXT call with:
          - concurrency cap (semaphore)
          - retry with exponential backoff + jitter
        """
        for attempt in range(1, MAX_RETRY + 1):
            try:
                async with self._net_sem:
                    return await fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                is_retryable = any(k in msg for k in [
                    "RequestTimeout", "DDoSProtection", "ExchangeNotAvailable",
                    "NetworkError", "ETIMEDOUT", "ECONNRESET", "502", "503", "504"
                ])
                if (attempt >= MAX_RETRY) or (not is_retryable):
                    raise
                backoff = (RETRY_BASE_SEC * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                self._log_err(f"[WARN] {label} retry {attempt}/{MAX_RETRY} err={msg} sleep={backoff:.2f}s")
                await asyncio.sleep(backoff)

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def _load_json(self, path: Path, default):
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            self._log_err(f"[ERR] load {path.name}: {e}")
        return default

    def _load_persistent_state(self):
        balance_loaded = False
        # balance
        bal = self._load_json(self.state_files.get("balance"), None)
        if isinstance(bal, (int, float)):
            self.balance = float(bal)
            balance_loaded = True

        # equity history
        eq = self._load_json(self.state_files["equity"], [])
        for item in eq:
            try:
                t = int(item.get("time", 0))
                v = float(item.get("equity", 0.0))
                self._equity_history.append({"time": t, "equity": v})
            except Exception:
                continue

        # trade tape
        trades = self._load_json(self.state_files["trade"], [])
        for t in trades:
            self.trade_tape.append(t)

        # eval history
        evals = self._load_json(self.state_files["eval"], [])
        for e in evals:
            self.eval_history.append(e)

        # positions
        poss = self._load_json(self.state_files.get("positions"), [])
        if isinstance(poss, list):
            for p in poss:
                try:
                    sym = str(p.get("symbol"))
                    if not sym:
                        continue
                    p["entry_price"] = float(p.get("entry_price", 0.0))
                    p["quantity"] = float(p.get("quantity", 0.0))
                    p["notional"] = float(p.get("notional", 0.0))
                    p["leverage"] = float(p.get("leverage", self.leverage))
                    p["cap_frac"] = float(p.get("cap_frac", 0.0))
                    p["entry_time"] = int(p.get("entry_time", now_ms()))
                    p["hold_limit"] = int(p.get("hold_limit", MAX_POSITION_HOLD_SEC * 1000))
                    self.positions[sym] = p
                except Exception:
                    continue

        # fallback balance from equity history if not loaded and no positions
        if (not balance_loaded) and self._equity_history and not self.positions:
            self.balance = float(self._equity_history[-1]["equity"])

    def _persist_state(self, force: bool = False):
        now = now_ms()
        if not force and (now - self._last_state_persist_ms < 10_000):
            return
        self._last_state_persist_ms = now
        try:
            with self.state_files["equity"].open("w", encoding="utf-8") as f:
                json.dump(list(self._equity_history), f, ensure_ascii=False)
            with self.state_files["trade"].open("w", encoding="utf-8") as f:
                json.dump(list(self.trade_tape), f, ensure_ascii=False)
            with self.state_files["eval"].open("w", encoding="utf-8") as f:
                json.dump(list(self.eval_history), f, ensure_ascii=False)
            with self.state_files["positions"].open("w", encoding="utf-8") as f:
                json.dump(list(self.positions.values()), f, ensure_ascii=False)
            with self.state_files["balance"].open("w", encoding="utf-8") as f:
                json.dump(self.balance, f, ensure_ascii=False)
        except Exception as e:
            self._log_err(f"[ERR] persist state: {e}")

    def _compute_returns_and_vol(self, prices):
        if prices is None or len(prices) < 10:
            return None, None

        log_returns = []
        for i in range(1, len(prices)):
            p0, p1 = prices[i - 1], prices[i]
            if p0 and p1 and p0 > 0 and p1 > 0:
                log_returns.append(math.log(p1 / p0))

        if len(log_returns) < 5:
            return None, None

        mu = sum(log_returns) / len(log_returns)
        var = sum((r - mu) ** 2 for r in log_returns) / len(log_returns)
        sigma = math.sqrt(var)
        return mu, sigma

    @staticmethod
    def _annualize_mu_sigma(mu_bar: float, sigma_bar: float, bar_seconds: float) -> tuple[float, float]:
        bars_per_year = (365.0 * 24.0 * 3600.0) / float(bar_seconds)
        mu_base = mu_bar * bars_per_year
        sigma_annual = sigma_bar * math.sqrt(bars_per_year)
        return float(mu_base), float(sigma_annual)

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _row(self, sym, price, ts, decision, candles, ctx=None):
        status = "WAIT"
        ai = "-"
        mc = "-"
        conf = 0.0

        mc_meta = {}
        if decision:
            status = decision.get("action", "WAIT")  # LONG/SHORT/WAIT
            conf = self._safe_float(decision.get("confidence", 0.0), 0.0)
            mc = decision.get("reason", "") or "-"
            # details에서 mc 메타 뽑기
            for d in decision.get("details", []):
                if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                    mc_meta = d.get("meta", {}) or {}
                    break
            # details에 없으면 decision.meta 사용 (엔진이 meta를 직접 주는 경우)
            if not mc_meta:
                mc_meta = decision.get("meta", {}) or {}

        trade_age = ts - self.market[sym]["ts"] if self.market[sym]["ts"] else None
        kline_age = ts - self._last_kline_ok_ms[sym] if self._last_kline_ok_ms[sym] else None

        ob_age = ts - self.orderbook[sym]["ts"] if self.orderbook[sym]["ts"] else None
        ob_ready = bool(self.orderbook[sym]["ready"])

        meta = decision.get("meta", {}) if decision else {}
        ev = self._safe_float(decision.get("ev", meta.get("ev", 0.0)) if decision else 0.0, 0.0)
        kelly = self._safe_float(meta.get("kelly", 0.0), 0.0)
        regime = (ctx or {}).get("regime") or meta.get("regime") or "-"
        mode = "alpha"
        if self.positions.get(sym, {}).get("tag") == "spread":
            mode = "spread"
        elif decision and any(d.get("_engine") == "consensus" for d in decision.get("details", [])):
            mode = "consensus"
        action_type = self.positions.get(sym, {}).get("tag") == "spread" and "SPREAD" or (self.positions.get(sym) and "HOLD") or self._last_actions.get(sym, "-")

        pos = self.positions.get(sym)
        pos_side = pos.get("side") if pos else "-"
        pos_pnl = None
        pos_roe = None
        pos_lev = pos.get("leverage") if pos else self._dyn_leverage.get(sym)
        pos_sf = pos.get("size_frac") if pos else None
        pos_cap_frac = pos.get("cap_frac") if pos else None
        if pos and price:
            entry = float(pos.get("entry_price", price))
            qty = float(pos.get("quantity", 0.0))
            pnl = ((price - entry) * qty) if pos_side == "LONG" else ((entry - price) * qty)
            notional = float(pos.get("notional", 0.0))
            lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
            base_notional = notional / max(lev_safe, 1e-6)
            pos_pnl = pnl
            pos_roe = pnl / base_notional if base_notional else 0.0

        def _opt_float(val):
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        event_p_tp = _opt_float(meta.get("event_p_tp"))
        event_p_timeout = _opt_float(meta.get("event_p_timeout"))
        event_t_median = _opt_float(meta.get("event_t_median"))
        event_ev_r = _opt_float(meta.get("event_ev_r"))
        event_cvar_r = _opt_float(meta.get("event_cvar_r"))
        event_ev_pct = _opt_float(meta.get("event_ev_pct"))
        event_cvar_pct = _opt_float(meta.get("event_cvar_pct"))
        horizon_weights = meta.get("horizon_weights")
        ev_by_h = meta.get("ev_by_horizon")
        win_by_h = meta.get("win_by_horizon")
        cvar_by_h = meta.get("cvar_by_horizon")
        horizon_seq = meta.get("horizon_seq")

        return {
            "symbol": sym,
            "price": price,
            "status": status,
            "ai": ai,
            "mc": mc,
            "conf": conf,
            "ev": ev,
            "kelly": kelly,
            "regime": regime,
            "mode": mode,
            "action_type": action_type,
            "candles": candles,
            "event_p_tp": event_p_tp,
            "event_p_timeout": event_p_timeout,
            "event_t_median": event_t_median,
            "event_ev_r": event_ev_r,
            "event_cvar_r": event_cvar_r,
            "event_ev_pct": event_ev_pct,
            "event_cvar_pct": event_cvar_pct,
            "horizon_weights": horizon_weights,
            "ev_by_horizon": ev_by_h,
            "win_by_horizon": win_by_h,
            "cvar_by_horizon": cvar_by_h,
            "horizon_seq": horizon_seq,

            # freshness
            "trade_age": trade_age,
            "kline_age": kline_age,
            "orderbook_age": ob_age,
            "orderbook_ready": ob_ready,

            "pos_side": pos_side,
            "pos_pnl": pos_pnl,
            "pos_roe": pos_roe,
            "pos_tag": pos.get("tag") if pos else None,
            "pos_leverage": pos_lev,
            "pos_size_frac": pos_sf,
            "pos_cap_frac": pos_cap_frac,

            # MC diagnostics
            "mc_h_desc": mc_meta.get("best_horizon_desc") or mc_meta.get("best_horizon") or "-",
            "mc_ev": self._safe_float(mc_meta.get("ev", 0.0), 0.0),
            "mc_win_rate": self._safe_float(mc_meta.get("win_rate", 0.0), 0.0),
            "mc_be_win_rate": self._safe_float(mc_meta.get("be_win_rate", 0.0), 0.0),
            "mc_tp": self._safe_float(mc_meta.get("tp", 0.0), 0.0),
            "mc_sl": self._safe_float(mc_meta.get("sl", 0.0), 0.0),
            "mc_hit_rate": self._safe_float(mc_meta.get("hit_rate", 0.0), 0.0),

            "details": (decision.get("details", []) if decision else []),
        }

    def _total_open_notional(self) -> float:
        return sum(float(pos.get("notional", 0.0)) for pos in self.positions.values())

    def _can_enter_position(self, notional: float) -> tuple[bool, str]:
        if self.position_cap_enabled and len(self.positions) >= self.max_positions:
            return False, "max positions reached"
        if self.exposure_cap_enabled and (self._total_open_notional() + notional) > (self.balance * self.max_notional_frac):
            return False, "exposure capped"
        return True, ""

    def _entry_permit(self, sym: str, decision: dict, ts_ms: int) -> tuple[bool, str]:
        now_sec = time.time()
        if now_sec < self._cooldown_until.get(sym, 0):
            self._entry_streak[sym] = 0
            return False, "cooldown"
        meta = (decision.get("meta") or {}) if decision else {}
        ev = float(decision.get("ev", 0.0) or 0.0)
        win = float(decision.get("confidence", 0.0) or 0.0)
        ev_thr = float(meta.get("ev_entry_threshold", 0.0) or 0.0)
        win_thr = float(meta.get("win_entry_threshold", 0.55) or 0.55)
        ev_thr_dyn = meta.get("ev_entry_threshold_dyn")
        if ev_thr_dyn is not None:
            try:
                ev_thr = max(ev_thr, float(ev_thr_dyn))
            except Exception:
                pass
        ev_mid = meta.get("ev_mid")
        win_mid = meta.get("win_mid")

        # 중기 필터: EV_mid가 음수이거나 win_mid<0.50인 경우만 차단(완화)
        if ev_mid is not None and ev_mid < 0:
            self._entry_streak[sym] = 0
            return False, "mid_ev_neg"
        if win_mid is not None and win_mid < 0.50:
            self._entry_streak[sym] = 0
            return False, "mid_win_low"

        # EV/Win 충족 여부
        if ev >= ev_thr and win >= win_thr:
            strong = ev >= ev_thr * 1.5
            needed = 1 if strong else ENTRY_STREAK_MIN
            self._entry_streak[sym] = self._entry_streak.get(sym, 0) + 1
            if self._entry_streak[sym] < needed:
                return False, "streak"
        else:
            self._entry_streak[sym] = 0
            return False, "threshold"
        return True, ""

    def _calc_position_size(self, decision: dict, price: float, leverage: float, size_frac_override: float | None = None) -> tuple[float, float, float]:
        meta = (decision or {}).get("meta", {}) or {}
        size_frac = size_frac_override if size_frac_override is not None else decision.get("size_frac") or meta.get("size_fraction") or self.default_size_frac
        cap_frac = meta.get("regime_cap_frac")
        if cap_frac is not None:
            try:
                size_frac = min(size_frac, float(cap_frac))
            except Exception:
                pass
        # 상한 제거: 신호가 강하면 엔진이 제시한 비중을 그대로 사용
        size_frac = float(max(0.0, size_frac))
        notional = float(max(0.0, self.balance * size_frac * leverage))
        qty = float(notional / price) if price and notional > 0 else 0.0
        return size_frac, notional, qty

    #def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
    def _dynamic_leverage_risk(self, decision: dict, ctx: dict) -> float:
        def _f(x, default=0.0):
            try:
                if x is None:
                    return float(default)
                return float(x)
            except Exception:
                return float(default)

        regime = ctx.get("regime") or "chop"
        sigma = _f(ctx.get("sigma"), 0.0)
        meta = decision.get("meta") or {}
        ev = _f(decision.get("ev"), 0.0)
        cvar = _f(meta.get("cvar05", decision.get("cvar")), 0.0)
        event_p_sl = _f(meta.get("event_p_sl"), 0.0)
        spread_pct = _f(meta.get("spread_pct", ctx.get("spread_pct")), 0.0002)
        execution_cost = _f(meta.get("execution_cost", meta.get("fee_rt")), 0.0)
        slippage_pct = _f(meta.get("slippage_pct", 0.0), 0.0)
        event_cvar_pct = _f(meta.get("event_cvar_pct"), 0.0)

        # risk = max(|CVaR|, |event_cvar_pct|) + 0.7*spread + 0.5*slippage + 0.5*sigma (+ p_sl 가중)
        risk_score = max(abs(cvar), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma + 0.2 * event_p_sl
        if risk_score <= 1e-6:
            risk_score = 1e-6

        lev_max_map = {"bull": self.max_leverage, "bear": self.max_leverage, "chop": min(self.max_leverage, 30.0), "volatile": min(self.max_leverage, 20.0)}
        lev_max = lev_max_map.get(regime, self.max_leverage)

        # EV를 위험 대비 비례 반영 (음수 EV면 최소 1배로 수렴)
        lev_raw = (max(ev - execution_cost, 0.0) / risk_score) * K_LEV
        lev = float(max(1.0, min(lev_max, lev_raw)))
        sym = ctx.get("symbol")
        if sym:
            self._dyn_leverage[sym] = lev
        return lev

    def _direction_bias(self, closes) -> int:
        if not closes:
            return 1
        price_now = closes[-1]
        fast_period = min(10, len(closes))
        slow_period = min(40, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        bias = 1 if fast >= slow else -1
        # 레짐 힌트: 상승장에서는 롱 우선, 하락장에서는 숏 우선 (더 민감하게)
        if len(closes) >= 20:
            slope_short = closes[-1] - closes[-20]
            pct_short = slope_short / max(price_now, 1e-6)
            if pct_short > 0.002:
                bias = 1
            elif pct_short < -0.002:
                bias = -1
        if len(closes) >= 60:
            slope_long = closes[-1] - closes[-60]
            pct_long = slope_long / max(price_now, 1e-6)
            if pct_long > 0.004:
                bias = 1
            elif pct_long < -0.004:
                bias = -1
        # 최근 방향 비율로 편향 교정
        if len(closes) >= 15:
            rets = [1 if closes[i] >= closes[i - 1] else -1 for i in range(len(closes) - 14, len(closes))]
            up_ratio = sum(1 for r in rets if r > 0) / len(rets)
            if up_ratio >= 0.6:
                bias = 1
            elif up_ratio <= 0.4:
                bias = -1
        return bias

    def _infer_regime(self, closes) -> str:
        if not closes or len(closes) < 30:
            return "chop"
        fast_period = min(80, len(closes))
        slow_period = min(200, len(closes))
        fast = sum(closes[-fast_period:]) / fast_period
        slow = sum(closes[-slow_period:]) / slow_period
        slope_short = closes[-1] - closes[max(0, len(closes) - 6)]
        slope_long = closes[-1] - closes[max(0, len(closes) - 40)]
        rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, min(len(closes), 180))]
        vol = float(np.std(rets)) if rets else 0.0
        # 변동성 높고 추세 약하면 volatile
        if vol > 0.01 and abs(slope_short) < closes[-1] * 0.0015:
            return "volatile"
        # 강한 상승/하락을 더 민감하게
        if fast > slow and slope_long > 0 and slope_short > 0:
            return "bull"
        if fast < slow and slope_long < 0 and slope_short < 0:
            return "bear"
        return "chop"

    def _compute_ofi_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym)
        if not ob:
            return 0.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        # 심도별 가중(상위호가 가중치 ↑)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        bid_vol = sum(float(b[1]) * weights[i] for i, b in enumerate(bids[: len(weights)]) if len(b) >= 2)
        ask_vol = sum(float(a[1]) * weights[i] for i, a in enumerate(asks[: len(weights)]) if len(a) >= 2)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return float((bid_vol - ask_vol) / denom)

    def _liquidity_score(self, sym: str) -> float:
        ob = self.orderbook.get(sym)
        if not ob:
            return 1.0
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2) + sum(float(a[1]) for a in asks[:5] if len(a) >= 2)
        return float(max(vol, 1.0))

    def _maybe_place_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False):
        if not self.enable_orders or quantity <= 0:
            return
        asyncio.create_task(self._execute_order(symbol, side, quantity, reduce_only))

    async def _execute_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False):
        """
        실 주문 호출. ENABLE_LIVE_ORDERS가 True일 때만 create_order를 호출한다.
        """
        order_side = "buy" if side == "LONG" else "sell"
        params = {"reduceOnly": reduce_only}
        try:
            await self.exchange.create_order(symbol, "market", order_side, quantity, None, params)
            self._log(f"[ORDER] {symbol} {order_side} {quantity:.6f} reduce_only={reduce_only}")
        except Exception as e:
            self._log_err(f"[ERR] order {symbol} {order_side}: {e}")

    def _enter_position(
        self,
        sym: str,
        side: str,
        price: float,
        decision: dict,
        ts: int,
        ctx: dict | None = None,
        *,
        size_frac_override: float | None = None,
        hold_limit_override: int | None = None,
        tag: str | None = None,
        leverage_override: float | None = None,
    ):
        lev = leverage_override if leverage_override is not None else self.leverage
        size_frac, notional, qty = self._calc_position_size(decision, price, lev, size_frac_override=size_frac_override)
        can_enter, reason = self._can_enter_position(notional)
        if not can_enter or qty <= 0:
            self._log(f"[{sym}] skip entry ({reason})")
            return

        ctx = ctx or {}

        meta = (decision.get("meta") or {}) if decision else {}
        horizon = int(meta.get("best_horizon_steps") or MAX_POSITION_HOLD_SEC)
        hold_limit = min(max(horizon, 120), MAX_POSITION_HOLD_SEC)
        if hold_limit_override is not None:
            hold_limit = hold_limit_override
        hold_ms = hold_limit * 1000
        pos = {
            "symbol": sym,
            "side": side,
            "entry_price": float(price),
            "entry_time": ts,
            "hold_limit": hold_ms,
            "quantity": qty,
            "notional": notional,
            "size_frac": size_frac,
            "tag": tag,
            "leverage": lev,
            "cap_frac": float(notional / self.balance) if self.balance else 0.0,
            "fee_paid": notional * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker),
            "regime": (ctx or {}).get("regime") or (decision.get("meta") or {}).get("regime"),
            "session": (ctx or {}).get("session") or (decision.get("meta") or {}).get("session"),
            # 예측 스냅샷
            "pred_win": decision.get("confidence") if decision else None,
            "pred_ev": decision.get("ev") if decision else None,
            "pred_event_ev_r": (decision.get("meta") or {}).get("event_ev_r") if decision else None,
            "pred_event_p_tp": (decision.get("meta") or {}).get("event_p_tp") if decision else None,
            "pred_event_p_sl": (decision.get("meta") or {}).get("event_p_sl") if decision else None,
            "consensus_used": self._consensus_used_flag(decision),
        }
        # 진입 수수료 선반영
        fee_entry = pos["fee_paid"]
        self.balance -= fee_entry
        self.positions[sym] = pos
        self._log(f"[{sym}] ENTER {side} qty={qty:.4f} notional={notional:.2f} fee={fee_entry:.4f} size={size_frac:.2%} tag={tag or '-'}")
        self._last_actions[sym] = "ENTER"
        self._maybe_place_order(sym, side, qty, reduce_only=False)
        entry_type = "SPREAD" if tag == "spread" else "ENTER"
        self._record_trade(entry_type, sym, side, price, qty, pos, fee=fee_entry)
        self._persist_state(force=True)

    def _close_position(self, sym: str, price: float, reason: str, exit_kind: str = "MANUAL"):
        pos = self.positions.pop(sym, None)
        if not pos or price is None:
            return
        qty = float(pos.get("quantity", 0.0))
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        notional_entry = float(pos.get("notional", 0.0))
        fee_exit = price * qty * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker)
        pnl = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        pnl_net = pnl - fee_exit
        self.balance += pnl_net
        self._log(f"[{sym}] EXIT {side} qty={qty:.4f} pnl={pnl_net:.2f} fee={fee_exit:.4f} ({reason})")
        self._last_actions[sym] = "EXIT"
        exit_side = "SHORT" if side == "LONG" else "LONG"
        self._maybe_place_order(sym, exit_side, qty, reduce_only=True)
        realized_r = pnl_net / notional_entry if notional_entry else 0.0
        hit = 1 if pnl > 0 else 0
        self._record_trade(
            "EXIT",
            sym,
            side,
            price,
            qty,
            pos,
            pnl=pnl_net,
            fee=fee_exit,
            reason=reason,
            realized_r=realized_r,
            hit=hit,
        )
        # 재진입 쿨다운 (종류별)
        self._mark_exit_and_cooldown(sym, exit_kind=exit_kind, ts_ms=now_ms())
        # 예측 vs 실제 기록
        pred_win = pos.get("pred_win")
        pred_ev = pos.get("pred_ev")
        pred_event_ev_r = pos.get("pred_event_ev_r")
        pred_event_p_tp = pos.get("pred_event_p_tp")
        pred_event_p_sl = pos.get("pred_event_p_sl")
        brier_sl = None
        if pred_event_p_sl is not None:
            try:
                brier_sl = (float(pred_event_p_sl) - (1 - hit)) ** 2
            except Exception:
                brier_sl = None
        self.eval_history.append({
            "hit": hit,
            "pred_win": pred_win,
            "pred_ev": pred_ev,
            "pred_event_ev_r": pred_event_ev_r,
            "pred_event_p_tp": pred_event_p_tp,
            "pred_event_p_sl": pred_event_p_sl,
            "realized_r": realized_r,
            "brier_sl": brier_sl,
        })
        try:
            log_payload = {
                "regime": pos.get("regime"),
                "session": pos.get("session"),
                "EV": pred_ev,
                "event_p_tp": pred_event_p_tp,
                "event_p_sl": pred_event_p_sl,
                "realized_r": realized_r,
                "consensus_used": pos.get("consensus_used"),
            }
            self._log(f"[METRIC] {json.dumps(log_payload)}")
        except Exception:
            pass
        self._persist_state(force=True)

    def _rebalance_position(self, sym: str, price: float, decision: dict, leverage_override: float | None = None):
        """
        기존 포지션이 있을 때 목표 비중과 현 포지션이 크게 다르면 수량/노출을 조정한다.
        실제 주문은 ENABLE_LIVE_ORDERS에 따라 별도 처리.
        """
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        lev = leverage_override if leverage_override is not None else pos.get("leverage", self.leverage)
        target_size_frac, target_notional, target_qty = self._calc_position_size(decision, price, lev)
        if target_notional <= 0:
            # 목표 노출이 0이면 전량 청산
            self._close_position(sym, price, "rebalance to zero")
            return
        curr_notional = float(pos.get("notional", 0.0))
        delta = abs(target_notional - curr_notional) / curr_notional if curr_notional else 1.0
        # 항상 레버리지/메타 업데이트
        if leverage_override is not None:
            pos["leverage"] = leverage_override
        if delta < REBALANCE_THRESHOLD_FRAC:
            return
        entry = float(pos.get("entry_price", price))
        side = pos.get("side")
        curr_qty = float(pos.get("quantity", 0.0))

        if target_notional < curr_notional and curr_notional > 0:
            # 부분 청산: 줄이는 비율만큼 실현 손익을 balance에 반영
            reduce_ratio = 1.0 - (target_notional / curr_notional)
            close_qty = curr_qty * reduce_ratio
            close_notional = curr_notional * reduce_ratio
            pnl_realized = (price - entry) * close_qty if side == "LONG" else (entry - price) * close_qty
            fee_partial = close_notional * (self.fee_taker if self.fee_mode == "taker" else self.fee_maker)
            pnl_realized_net = pnl_realized - fee_partial
            self.balance += pnl_realized_net

            # 남은 포지션 업데이트
            pos["quantity"] = max(curr_qty - close_qty, 0.0)
            pos["notional"] = max(target_notional, 0.0)
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(pos["notional"] / self.balance) if self.balance else 0.0

            # 부분 청산 기록
            close_pos = dict(pos)
            close_pos["notional"] = close_notional
            close_pos["quantity"] = close_qty
            close_pos["leverage"] = lev
            self._log(f"[{sym}] PARTIAL EXIT by REBAL qty={close_qty:.4f} pnl={pnl_realized_net:.2f} fee={fee_partial:.4f}")
            self._last_actions[sym] = "REBAL_EXIT"
            self._record_trade("REBAL_EXIT", sym, side, price, close_qty, close_pos, pnl=pnl_realized_net, fee=fee_partial, reason="rebalance partial")
            self._cooldown_until[sym] = time.time() + COOLDOWN_SEC
        else:
            # 노출 확대/동일: 포지션만 갱신
            pos["notional"] = target_notional
            pos["quantity"] = target_qty
            pos["size_frac"] = target_size_frac
            pos["cap_frac"] = float(target_notional / self.balance) if self.balance else 0.0
            self._log(f"[{sym}] REBALANCE qty={target_qty:.4f} notional={target_notional:.2f} size={target_size_frac:.2%}")
            self._last_actions[sym] = "REBAL"
            # 기록용 스냅샷
            pnl_now = (price - entry) * target_qty if side == "LONG" else (entry - price) * target_qty
            self._record_trade("REBAL", sym, side, price, target_qty, pos, pnl=pnl_now, reason="rebalance")

        # 실제 리밸런싱 주문 경로 (옵션)
        adj_qty = max(0.0, target_qty - float(pos.get("quantity", 0.0)))
        if adj_qty > 0:
            self._maybe_place_order(sym, pos["side"], adj_qty, reduce_only=False)
        self._persist_state(force=True)

    def _maybe_exit_position(self, sym: str, price: float, decision: dict, ts: int):
        pos = self.positions.get(sym)
        if not pos or price is None:
            return
        action = (decision or {}).get("action") or "WAIT"
        hold_limit_ms = pos.get("hold_limit", MAX_POSITION_HOLD_SEC * 1000)
        age_ms = ts - pos.get("entry_time", ts)
        entry = float(pos.get("entry_price", price))
        qty = float(pos.get("quantity", 0.0))
        side = pos.get("side")
        notional = float(pos.get("notional", 0.0))
        pnl_unreal = (price - entry) * qty if side == "LONG" else (entry - price) * qty
        lev_safe = float(pos.get("leverage", self.leverage) or 1.0)
        base_notional = notional / max(lev_safe, 1e-6) if notional else 0.0
        roe_unreal = pnl_unreal / base_notional if base_notional else 0.0
        exit_reasons = []
        if action == "WAIT":
            exit_reasons.append("engine WAIT")
        elif action in ("LONG", "SHORT") and action != pos["side"]:
            exit_reasons.append("signal flip")
        if age_ms >= hold_limit_ms:
            exit_reasons.append("hold timeout")
        # 공격적 손실 컷 (미실현 ROE 기준)
        if roe_unreal <= -0.02:
            exit_reasons.append("unrealized_dd")
        if exit_reasons:
            self._close_position(sym, price, ", ".join(exit_reasons))

    def _record_trade(
        self,
        ttype: str,
        sym: str,
        side: str,
        price: float,
        qty: float,
        pos: dict,
        pnl: float | None = None,
        fee: float | None = None,
        reason: str | None = None,
        realized_r: float | None = None,
        hit: int | None = None,
        **_ignored,
    ):
        ts = time.strftime("%H:%M:%S")
        # normalize numeric fields
        pnl_val = None if pnl is None else float(pnl)
        notional = pos.get("notional")
        lev = pos.get("leverage")
        base_notional = None
        if notional is not None and lev not in (None, 0):
            try:
                base_notional = float(notional) / float(max(lev, 1e-6))
            except Exception:
                base_notional = None
        roe_val = None
        if pnl_val is not None and base_notional:
            try:
                roe_val = pnl_val / base_notional
            except Exception:
                roe_val = None

        entry = {
            "time": ts,
            "type": ttype,
            "ttype": ttype,
            "symbol": sym,
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "pnl": pnl_val,
            "roe": roe_val,
            "notional": notional,
            "leverage": lev,
            "fee": None if fee is None else float(fee),
            "tag": pos.get("tag"),
            "reason": reason,
            "pred_win": pos.get("pred_win"),
            "pred_ev": pos.get("pred_ev"),
            "pred_event_ev_r": pos.get("pred_event_ev_r"),
            "pred_event_p_tp": pos.get("pred_event_p_tp"),
            "pred_event_p_sl": pos.get("pred_event_p_sl"),
            "consensus_used": pos.get("consensus_used"),
            "realized_r": float(realized_r) if realized_r is not None else (pnl_val / base_notional if (pnl_val is not None and base_notional) else None),
            "hit": int(hit) if hit is not None else None,
        }
        self.trade_tape.append(entry)
        self._persist_state(force=True)

    def _consensus_used_flag(self, decision: dict) -> bool:
        meta = decision.get("meta") or {}
        if meta.get("consensus_used"):
            return True
        for d in decision.get("details", []) or []:
            m = d.get("meta") or {}
            if m.get("consensus_used"):
                return True
        return False

    def _ema_update(self, store: dict, key, x: float, half_life_sec: float, ts_ms: int) -> float:
        prev = store.get(key)
        if prev is None:
            store[key] = (float(x), ts_ms)
            return float(x)
        prev_val, prev_ts = prev
        dt_sec = max(1.0, (ts_ms - prev_ts) / 1000.0)
        alpha = 1.0 - math.exp(-math.log(2) * dt_sec / max(half_life_sec, 1e-6))
        new_val = alpha * float(x) + (1.0 - alpha) * float(prev_val)
        store[key] = (new_val, ts_ms)
        return new_val

    @staticmethod
    def _calc_rsi(closes, period: int = RSI_PERIOD) -> float | None:
        if closes is None or len(closes) <= period:
            return None
        gains = []
        losses = []
        for i in range(1, period + 1):
            delta = closes[-i] - closes[-i - 1]
            if delta >= 0:
                gains.append(delta)
            else:
                losses.append(abs(delta))
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _consensus_action(self, decision: dict, ctx: dict) -> tuple[str, float]:
        """
        여러 지표를 투표/가중합하여 방향 합의 점수와 액션을 결정.
        score > CONSENSUS_THRESHOLD => LONG, score < -CONSENSUS_THRESHOLD => SHORT, else WAIT
        """
        score = 0.0
        base_action = decision.get("action") if decision else "WAIT"
        ev = float(decision.get("ev", 0.0)) if decision else 0.0
        meta = (decision.get("meta") or {}) if decision else {}
        cvar = float(meta.get("cvar05", decision.get("cvar", 0.0)) if decision else 0.0)
        win = float(decision.get("confidence", 0.0)) if decision else 0.0
        direction = int(ctx.get("direction", 1))
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))
        ofi = float(ctx.get("ofi_score", 0.0))
        rsi = self._calc_rsi(ctx.get("closes"), period=RSI_PERIOD)
        closes = ctx.get("closes") or []
        sym = ctx.get("symbol")
        spread_pct = float(meta.get("spread_pct", ctx.get("spread_pct", 0.0) or 0.0))
        liq = self._liquidity_score(sym) if sym else 1.0
        p_sl = float(meta.get("event_p_sl", 0.0) or 0.0)

        if ev <= 0:
            return "WAIT", score

        # --- history update for z-scores (regime/session residual 기반) ---
        def _rz(val, hist_deque: deque | None, default_scale: float = 0.001):
            try:
                if hist_deque is None or len(hist_deque) < 5:
                    return float(val / max(default_scale, 1e-6))
                arr = np.asarray(hist_deque, dtype=np.float64)
                mean = float(arr.mean())
                std = float(arr.std())
                std = std if std > 1e-6 else default_scale
                return float((val - mean) / std)
            except Exception:
                return float(val / max(default_scale, 1e-6))

        key = (regime, session)
        ev_hist_rs = self._ev_regime_hist_rs.setdefault(key, deque(maxlen=500))
        cvar_hist_rs = self._cvar_regime_hist_rs.setdefault(key, deque(maxlen=500))
        ev_hist_rs.append(ev)
        cvar_hist_rs.append(cvar)

        ofi_hist_base = self._ofi_regime_hist.setdefault(key, deque(maxlen=500))
        ofi_hist_base.append(ofi)
        ofi_mean = float(np.mean(ofi_hist_base)) if ofi_hist_base else 0.0
        ofi_residual = ofi - ofi_mean
        ofi_resid_hist = self._ofi_resid_hist.setdefault(key, deque(maxlen=500))
        ofi_resid_hist.append(ofi_residual)

        spread_hist = self._spread_regime_hist.setdefault(key, deque(maxlen=500))
        spread_hist.append(spread_pct)
        liq_hist = self._liq_regime_hist.setdefault(key, deque(maxlen=500))
        liq_hist.append(liq)

        z_ev = _rz(ev, ev_hist_rs)
        z_cvar = _rz(-cvar, cvar_hist_rs, default_scale=0.0005)
        z_ofi = _rz(ofi_residual, ofi_resid_hist, default_scale=0.0005)
        z_spread = _rz(spread_pct, spread_hist, default_scale=0.0002)
        z_liq = _rz(liq, liq_hist, default_scale=1.0)

        bias = float(direction)

        score = (
            1.2 * z_ev
            + 0.9 * z_cvar  # -CVaR를 넣어 양수일수록 좋게
            - 0.7 * p_sl
            + 0.5 * z_ofi
            + 0.5 * bias
            + 0.3 * z_liq
            - 0.6 * z_spread
        )

        # RSI as mild tie-breaker
        if rsi is not None:
            if rsi >= RSI_LONG:
                score += 0.2
            elif rsi <= RSI_SHORT:
                score -= 0.2

        action = "WAIT"
        if score >= CONSENSUS_THRESHOLD:
            action = "LONG"
        elif score <= -CONSENSUS_THRESHOLD:
            action = "SHORT"
        else:
            action = base_action if base_action in ("LONG", "SHORT") else "WAIT"
        return action, score

    def _spread_signal(self, base: str, quote: str) -> tuple[str, str, float] | None:
        """
        단순 비율 mean-reversion 스프레드.
        ratio = base/quote. z > entry: short base / long quote, z < -entry: long base / short quote.
        """
        base_closes = list(self.ohlcv_buffer.get(base) or [])
        quote_closes = list(self.ohlcv_buffer.get(quote) or [])
        if len(base_closes) < SPREAD_LOOKBACK + 1 or len(quote_closes) < SPREAD_LOOKBACK + 1:
            return None
        ratio = [b / q for b, q in zip(base_closes[-SPREAD_LOOKBACK:], quote_closes[-SPREAD_LOOKBACK:]) if q]
        if len(ratio) < SPREAD_LOOKBACK:
            return None
        mean = sum(ratio) / len(ratio)
        var = sum((x - mean) ** 2 for x in ratio) / len(ratio)
        std = math.sqrt(max(var, 1e-9))
        latest = ratio[-1]
        z = (latest - mean) / std if std > 0 else 0.0
        if abs(z) < SPREAD_Z_ENTRY:
            return None
        if z > 0:
            return ("SHORT", "LONG", z)
        return ("LONG", "SHORT", z)

    def _manage_spreads(self, ts: int):
        """
        스프레드 진입/청산. 페어 양쪽 포지션이 모두 없을 때만 진입.
        """
        if not self.spread_enabled:
            return
        for base, quote in self.spread_pairs:
            base_px = self.market.get(base, {}).get("price")
            quote_px = self.market.get(quote, {}).get("price")
            if base_px is None or quote_px is None:
                continue
            signal = self._spread_signal(base, quote)
            base_pos = self.positions.get(base)
            quote_pos = self.positions.get(quote)

            # 청산 조건: 태그가 spread이고 z가 수렴하거나 보유 초과
            if (base_pos and base_pos.get("tag") == "spread") or (quote_pos and quote_pos.get("tag") == "spread"):
                # 재계산 z
                zinfo = self._spread_signal(base, quote)
                z_now = zinfo[2] if zinfo else 0.0
                should_exit = abs(z_now) <= SPREAD_Z_EXIT
                age_ok = False
                if base_pos:
                    age_ok = age_ok or (ts - base_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if quote_pos:
                    age_ok = age_ok or (ts - quote_pos.get("entry_time", ts) >= SPREAD_HOLD_SEC * 1000)
                if should_exit or age_ok:
                    if base_pos:
                        self._close_position(base, float(base_px), f"spread exit z={z_now:.2f}")
                    if quote_pos:
                        self._close_position(quote, float(quote_px), f"spread exit z={z_now:.2f}")
                    continue

            # 진입: 두 심볼 모두 포지션 없고 스프레드 신호가 명확할 때
            if signal and not base_pos and not quote_pos:
                base_side, quote_side, z = signal
                hold_override = min(SPREAD_HOLD_SEC, MAX_POSITION_HOLD_SEC)
                ctx_spread = {"regime": "spread", "session": time_regime()}
                meta_spread = {"regime": "spread", "session": time_regime()}
                self._enter_position(
                    base, base_side, float(base_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._enter_position(
                    quote, quote_side, float(quote_px), {"meta": meta_spread}, ts,
                    ctx=ctx_spread, size_frac_override=SPREAD_SIZE_FRAC,
                    hold_limit_override=hold_override, tag="spread"
                )
                self._log(f"[SPREAD] {base}/{quote} z={z:.2f} -> {base_side}/{quote_side}")
                self._last_actions[base] = "SPREAD_ENTER"
                self._last_actions[quote] = "SPREAD_ENTER"
    def _decide_v3(self, ctx: dict) -> dict:
        """
        v3 flow:
          1) Execution gate (spread)
          2) Alpha candidate (direction + OFI) -> choose side or WAIT
          3) MC validate at leverage=1 using EngineHub.decide(ctx)
          4) Risk-based leverage compute
          5) (optional) Re-run hub.decide with final leverage for reporting/meta
        Returns a decision dict compatible with existing downstream code.
        """

        sym = ctx.get("symbol")
        regime = str(ctx.get("regime", "chop"))
        session = str(ctx.get("session", "OFF"))

        # -------------------------
        # Stage-1: Execution gate
        # -------------------------
        spread_pct = ctx.get("spread_pct")
        if spread_pct is None:
            spread_pct = 0.0002  # fallback (2bp)
        spread_pct = float(spread_pct)

        spread_cap_map = {
            "bull": 0.0020,
            "bear": 0.0020,
            "chop": 0.0012,
            "volatile": 0.0008,
        }
        spread_cap = spread_cap_map.get(regime, 0.0012)
        if spread_pct > spread_cap:
            return {
                "action": "WAIT",
                "reason": "v3_exec_gate_spread",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {"regime": regime, "session": session, "spread_pct": spread_pct, "spread_cap": spread_cap},
                "details": [],
            }

        # -------------------------
        # Stage-2: Alpha candidate (direction + OFI residual-ish)
        # -------------------------
        direction = float(ctx.get("direction", 1))  # +1 / -1
        ofi = float(ctx.get("ofi_score", 0.0))
        ofi_mean = self.stats.ema_update("ofi_mean_v3", (regime, session), ofi, half_life_sec=900)
        ofi_res = ofi - float(ofi_mean)

        # simple scale proxy using recent abs-mean from stats buffer if available
        # (if not enough samples, it'll behave like "raw")
        try:
            z_ofi = self.stats.robust_z("ofi_res", (regime, session), ofi_res, fallback=ofi_res / 0.001)
        except Exception:
            z_ofi = ofi_res / 0.001

        alpha_long = 0.55 * direction + 0.35 * z_ofi
        alpha_short = -0.55 * direction - 0.35 * z_ofi
        alpha_max = max(alpha_long, alpha_short)

        alpha_min_map = {"bull": 0.25, "bear": 0.25, "chop": 0.40, "volatile": 0.55}
        alpha_min = alpha_min_map.get(regime, 0.40)
        if alpha_max < alpha_min:
            return {
                "action": "WAIT",
                "reason": "v3_alpha_gate",
                "ev": 0.0,
                "confidence": 0.0,
                "cvar": 0.0,
                "meta": {
                    "regime": regime, "session": session,
                    "alpha_long": alpha_long, "alpha_short": alpha_short,
                    "alpha_min": alpha_min, "z_ofi": z_ofi,
                    "spread_pct": spread_pct,
                },
                "details": [],
            }

        alpha_side = "LONG" if alpha_long >= alpha_short else "SHORT"

        # -------------------------
        # Stage-3: MC validate at leverage=1 (using existing EngineHub)
        # -------------------------
        ctx1 = dict(ctx)
        ctx1["leverage"] = 1.0
        decision1 = self.hub.decide(ctx1)

        ev1 = float(decision1.get("ev", 0.0) or 0.0)
        win1 = float(decision1.get("confidence", 0.0) or 0.0)
        cvar1 = float(decision1.get("cvar", 0.0) or 0.0)
        meta1 = decision1.get("meta") or {}

        p_sl = float(meta1.get("event_p_sl", 0.0) or 0.0)
        event_cvar_r = meta1.get("event_cvar_r")
        event_cvar_r = float(event_cvar_r) if event_cvar_r is not None else -999.0

        # v3 lev=1 gates (완화된 버전)
        ev1_floor = {"bull": 0.0002, "bear": 0.0002, "chop": 0.0005, "volatile": 0.0008}.get(regime, 0.0005)
        win1_floor = {"bull": 0.50, "bear": 0.50, "chop": 0.52, "volatile": 0.53}.get(regime, 0.52)
        cvar1_floor = {"bull": -0.010, "bear": -0.011, "chop": -0.008, "volatile": -0.007}.get(regime, -0.010)
        psl_max = {"bull": 0.42, "bear": 0.40, "chop": 0.35, "volatile": 0.32}.get(regime, 0.40)
        event_cvar_r_floor = {"bull": -1.20, "bear": -1.15, "chop": -1.05, "volatile": -0.95}.get(regime, -1.10)

        if ev1 < ev1_floor or win1 < win1_floor or cvar1 < cvar1_floor or p_sl > psl_max or event_cvar_r < event_cvar_r_floor:
            # decision1 포맷 유지 + reason만 v3로 덮어쓰기
            d = dict(decision1)
            d["action"] = "WAIT"
            d["reason"] = "v3_mc1_gate"
            m = dict(meta1)
            m.update({
                "v3_alpha_side": alpha_side,
                "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
                "ev1_floor": ev1_floor, "win1_floor": win1_floor, "cvar1_floor": cvar1_floor,
                "psl_max": psl_max, "event_cvar_r_floor": event_cvar_r_floor,
                "spread_pct": spread_pct,
            })
            d["meta"] = m
            return d

        # -------------------------
        # Stage-4: Risk-based leverage (use lev=1 metrics)
        # -------------------------
        sl_pct = float(meta1.get("mc_sl") or meta1.get("sl_pct") or 0.002)
        event_cvar_pct = event_cvar_r * sl_pct

        # best-effort slippage from meta if present
        slippage_pct = float(meta1.get("slippage_pct", 0.0) or 0.0)
        sigma = float(ctx.get("sigma", 0.0) or 0.0)

        risk = max(abs(cvar1), abs(event_cvar_pct)) + 0.7 * spread_pct + 0.5 * slippage_pct + 0.5 * sigma
        if risk < 1e-9:
            risk = 1e-9

        # leverage bounds per regime
        lev_max = {"bull": 20.0, "bear": 18.0, "chop": 10.0, "volatile": 8.0}.get(regime, 10.0)
        lev_raw = (max(ev1, 0.0) / risk) * K_LEV
        lev = float(max(1.0, min(lev_max, lev_raw)))

        # -------------------------
        # Stage-5: Re-run hub.decide with final leverage (reporting & size_frac)
        # -------------------------
        ctxF = dict(ctx)
        ctxF["leverage"] = lev
        decisionF = self.hub.decide(ctxF)

        # enforce alpha side: if hub returns opposite, prefer alpha side unless you want flip logic
        decisionF = dict(decisionF)
        decisionF["action"] = alpha_side if decisionF.get("action") in ("LONG", "SHORT") else "WAIT"

        metaF = dict(decisionF.get("meta") or {})
        metaF.update({
            "v3_used": True,
            "v3_alpha_side": alpha_side,
            "EV1": ev1, "Win1": win1, "CVaR1": cvar1,
            "risk": risk, "lev_raw": lev_raw, "lev": lev,
            "event_cvar_pct": event_cvar_pct,
            "spread_pct": spread_pct,
        })
        decisionF["meta"] = metaF

        # expose leverage/size_frac for your existing sizing code
        decisionF["leverage"] = lev

        # size cap by regime (your code already applies regime_cap_frac later, but we can prefill)
        cap_map = {"bull": 0.25, "bear": 0.22, "chop": 0.10, "volatile": 0.08}
        metaF.setdefault("regime_cap_frac", cap_map.get(regime, 0.10))

        return decisionF

    async def fetch_prices_loop(self):
        while True:
            try:
                tickers = await self._ccxt_call("fetch_tickers", self.exchange.fetch_tickers, SYMBOLS)
                ts = now_ms()
                ok_any = False
                for s in SYMBOLS:
                    last = (tickers.get(s) or {}).get("last")
                    if last is not None:
                        self.market[s]["price"] = float(last)
                        self.market[s]["ts"] = ts
                        ok_any = True
                if ok_any:
                    self._last_feed_ok_ms = ts
            except Exception as e:
                self._log_err(f"[ERR] fetch_tickers: {e}")
            await asyncio.sleep(1.0)  # ✅ 너무 촘촘하면 Bybit timeout 잦음

    async def preload_all_ohlcv(self, limit: int = OHLCV_PRELOAD_LIMIT):
        """
        서버 시작 전에 OHLCV를 미리 채워서 'candles 부족 / 준비중' 시간을 없앰.
        """
        for sym in SYMBOLS:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit)
                if not ohlcv:
                    continue
                self.ohlcv_buffer[sym].clear()
                last_ts = 0
                for c in ohlcv:
                    ts_ms = int(c[0])
                    close_price = float(c[4])
                    self.ohlcv_buffer[sym].append(close_price)
                    last_ts = ts_ms
                self._last_kline_ts[sym] = last_ts
                self._last_kline_ok_ms[sym] = now_ms()
                self._preloaded[sym] = True
                self._log(f"[PRELOAD] {sym} candles={len(self.ohlcv_buffer[sym])}")
            except Exception as e:
                self._log_err(f"[ERR] preload_ohlcv {sym}: {e}")

    async def fetch_ohlcv_loop(self):
        """
        - preload은 main()에서 한 번에 수행
        - 이후 최신 1분봉만 dedupe 갱신
        """
        while True:
            start = now_ms()
            try:
                for sym in SYMBOLS:
                    try:
                        ohlcv = await self._ccxt_call(
                            f"fetch_ohlcv {sym}",
                            self.exchange.fetch_ohlcv,
                            sym, timeframe=TIMEFRAME, limit=OHLCV_REFRESH_LIMIT
                        )
                        if not ohlcv:
                            continue
                        last = ohlcv[-1]
                        ts_ms = int(last[0])
                        close_price = float(last[4])

                        if ts_ms != self._last_kline_ts[sym]:
                            self.ohlcv_buffer[sym].append(close_price)
                            self._last_kline_ts[sym] = ts_ms
                            self._last_kline_ok_ms[sym] = now_ms()
                    except Exception as e_sym:
                        self._log_err(f"[ERR] fetch_ohlcv {sym}: {e_sym}")
            except Exception as e:
                self._log_err(f"[ERR] fetch_ohlcv(loop): {e}")

            elapsed = (now_ms() - start) / 1000.0
            sleep_left = max(1.0, OHLCV_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    async def fetch_orderbook_loop(self):
        """
        대시보드 Orderbook Ready = NO 해결용.
        - 심볼별 fetch_order_book 수행
        """
        while True:
            start = now_ms()
            for sym in SYMBOLS:
                try:
                    ob = await self._ccxt_call(
                        f"fetch_orderbook {sym}",
                        self.exchange.fetch_order_book,
                        sym, limit=ORDERBOOK_DEPTH
                    )
                    bids = (ob.get("bids") or [])[:ORDERBOOK_DEPTH]
                    asks = (ob.get("asks") or [])[:ORDERBOOK_DEPTH]
                    ready = bool(bids) and bool(asks)
                    self.orderbook[sym]["bids"] = bids
                    self.orderbook[sym]["asks"] = asks
                    self.orderbook[sym]["ready"] = ready
                    self.orderbook[sym]["ts"] = now_ms()
                except Exception as e_sym:
                    # rate limit/권한 이슈 등은 ready False로 유지
                    self.orderbook[sym]["ready"] = False
                    msg = str(e_sym)
                    if ("RequestTimeout" not in msg) and ("DDoSProtection" not in msg) and ("RateLimit" not in msg):
                        extra = ""
                        try:
                            status = getattr(e_sym, "status", None)
                            body = getattr(e_sym, "body", None)
                            if status is not None:
                                extra += f" status={status}"
                            if body:
                                extra += f" body={str(body)[:300]}"
                        except Exception:
                            pass
                        self._log_err(f"[ERR] fetch_orderbook {sym}: {repr(e_sym)}{extra}")
                await asyncio.sleep(0.35)  # 심볼 간 간격(레이트리밋 완화)

            elapsed = (now_ms() - start) / 1000.0
            # 전체 한바퀴 주기를 ORDERBOOK_SLEEP_SEC에 맞춤
            sleep_left = max(0.0, ORDERBOOK_SLEEP_SEC - elapsed)
            await asyncio.sleep(sleep_left)

    def _mark_exit_and_cooldown(self, sym: str, exit_kind: str, ts_ms: int):
        """
        exit_kind: "TP" | "TIMEOUT" | "SL" | "KILL" | "MANUAL"
        """
        k = (exit_kind or "MANUAL").upper()
        self._last_exit_kind[sym] = k
        if k in ("SL", "KILL"):
            self._cooldown_until[sym] = ts_ms + int(COOLDOWN_RISK_SEC * 1000)
        else:
            self._cooldown_until[sym] = ts_ms + int(COOLDOWN_TP_SEC * 1000)
        self._streak[sym] = 0

    def _compute_portfolio(self):
        unreal = 0.0
        total_notional = 0.0
        pos_list = []
        for sym, pos in self.positions.items():
            px = self.market[sym]["price"]
            if px is None:
                continue
            entry = float(pos["entry_price"])
            side = pos["side"]
            qty = float(pos.get("quantity", 0.0))
            notional = float(pos.get("notional", 0.0))
            pnl = ((px - entry) * qty) if side == "LONG" else ((entry - px) * qty)
            unreal += pnl
            total_notional += notional
            lev = float(pos.get("leverage", self.leverage))
            base_notional = notional / max(lev, 1e-6)
            pos_list.append({
                "symbol": sym,
                "side": side,
                "entry_price": entry,
                "current": float(px),
                "pnl": float(pnl),
                "roe": float(pnl / base_notional) if base_notional else 0.0,
                "leverage": lev,
                "quantity": qty,
                "notional": notional,
                "cap_frac": float(notional / self.balance) if self.balance else 0.0,
                "size_frac": pos.get("size_frac"),
                "tag": pos.get("tag"),
                "age_sec": max(0.0, (now_ms() - pos.get("entry_time", now_ms())) / 1000.0),
            })
        equity = float(self.balance) + float(unreal)
        self._equity_history.append({"time": now_ms(), "equity": equity})
        util = float(total_notional / self.balance) if self.balance else 0.0
        self._persist_state(force=False)
        return equity, unreal, util, pos_list

    def _compute_eval_metrics(self):
        # Brier score, hit rate, avg EV vs realized R
        if not self.eval_history:
            return {"brier": None, "hit_rate": None, "avg_ev": None, "avg_realized_r": None, "avg_event_ev_r": None}
        briers = []
        hits = []
        evs = []
        evr = []
        realized = []
        for e in self.eval_history:
            hit = e.get("hit")
            pred_win = e.get("pred_win")
            if pred_win is not None and hit is not None:
                try:
                    briers.append((float(pred_win) - float(hit)) ** 2)
                except Exception:
                    pass
            if hit is not None:
                hits.append(float(hit))
            if e.get("pred_ev") is not None:
                evs.append(float(e.get("pred_ev")))
            if e.get("pred_event_ev_r") is not None:
                evr.append(float(e.get("pred_event_ev_r")))
            if e.get("realized_r") is not None:
                realized.append(float(e.get("realized_r")))
        return {
            "brier": float(np.mean(briers)) if briers else None,
            "hit_rate": float(np.mean(hits)) if hits else None,
            "avg_ev": float(np.mean(evs)) if evs else None,
            "avg_realized_r": float(np.mean(realized)) if realized else None,
            "avg_event_ev_r": float(np.mean(evr)) if evr else None,
        }

    async def broadcast(self, rows):
        try:
            equity, unreal, util, pos_list = self._compute_portfolio()
            eval_metrics = self._compute_eval_metrics()
            ts = now_ms()

            feed_connected = (self._last_feed_ok_ms > 0) and (ts - self._last_feed_ok_ms < 10_000)
            feed = {
                "connected": bool(feed_connected),
                "last_msg_age": (ts - self._last_feed_ok_ms) if self._last_feed_ok_ms else None
            }

            payload = {
                "type": "full_update",
                "server_time": ts,
                "engine": {"modules_ok": True, "ws_clients": len(self.clients), "loop_ms": None},
                "feed": feed,
                "market": rows,
                "portfolio": {
                    "balance": float(self.balance),
                    "equity": float(equity),
                    "unrealized_pnl": float(unreal),
                    "utilization": util,
                    "utilization_cap": MAX_NOTIONAL_EXPOSURE if self.exposure_cap_enabled else None,
                    "positions": pos_list,
                    "history": list(self._equity_history),
                },
                "eval_metrics": eval_metrics,
                "logs": list(self.logs),
                "trade_tape": list(self.trade_tape),
            }

            msg = json.dumps(payload, ensure_ascii=False)
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)
        except Exception as e:
            import traceback
            self._log_err(f"[ERR] broadcast: {e} {traceback.format_exc()}")

    async def decision_loop(self):
        while True:
            rows = []
            ts = now_ms()
            self._decision_cycle = (self._decision_cycle + 1) % self._decision_log_every
            log_this_cycle = (self._decision_cycle == 0)
            try:
                for sym in SYMBOLS:
                    price = self.market[sym]["price"]
                    closes = list(self.ohlcv_buffer[sym])
                    candles = len(closes)

                    if price is None:
                        rows.append(self._row(sym, None, ts, None, candles))
                        continue

                    mu_bar, sigma_bar = self._compute_returns_and_vol(closes)
                    regime = self._infer_regime(closes)

                    spread_pct = None
                    ob = self.orderbook.get(sym)
                    if ob and ob.get("ready"):
                        bids = ob.get("bids") or []
                        asks = ob.get("asks") or []
                        if bids and asks and len(bids[0]) >= 2 and len(asks[0]) >= 2:
                            bid = float(bids[0][0])
                            ask = float(asks[0][0])
                            mid = (bid + ask) / 2.0 if (bid and ask) else 0.0
                            if mid > 0:
                                spread_pct = (ask - bid) / mid

                    # 1) realized (annualized)
                    mu_base, sigma = (0.0, 0.0)
                    if mu_bar is not None and sigma_bar is not None:
                        mu_base, sigma = self._annualize_mu_sigma(mu_bar, sigma_bar, bar_seconds=60.0)

                    # 2) regime μ/σ table blend (session aware)
                    session = time_regime()
                    mu_tab, sig_tab = get_regime_mu_sigma(regime, session, symbol=sym)
                    if mu_tab is not None and sig_tab is not None:
                        w = 0.35
                        mu_base = float((1.0 - w) * float(mu_base) + w * float(mu_tab))
                        sigma = float(max(1e-6, (1.0 - w) * float(sigma) + w * float(sig_tab)))

                    ofi_score = float(self._compute_ofi_score(sym))
                    tuner = getattr(self, "tuner", None)
                    if tuner and hasattr(tuner, "get_params"):
                        try:
                            regime_params = tuner.get_params(regime)
                        except Exception:
                            regime_params = None
                    else:
                        regime_params = None

                    # bootstrap returns for tail (optional)
                    bootstrap_returns = None
                    if closes is not None and len(closes) >= 64:
                        try:
                            x = np.asarray(closes, dtype=np.float64)
                            bootstrap_returns = np.diff(np.log(np.maximum(x, 1e-12))).astype(np.float64)[-512:]
                        except Exception:
                            bootstrap_returns = None

                    ctx = {
                        "symbol": sym,
                        "price": float(price),
                        "bar_seconds": 60.0,
                        "closes": closes,
                        "direction": self._direction_bias(closes),
                        "regime": regime,
                        "ofi_score": float(ofi_score),
                        "liquidity_score": self._liquidity_score(sym),
                        "leverage": None,  # placeholder; set below
                        "mu_base": float(mu_base),
                        "sigma": float(max(sigma, 0.0)),
                        "regime_params": regime_params,
                        "session": session,
                        "spread_pct": spread_pct,
                        # ---- MC tail + JAX knobs ----
                        "use_jax": True,
                        "tail_mode": "student_t",
                        "tail_model": "student_t",
                        "tail_df": 6.0,
                        "student_t_df": 6.0,
                        "bootstrap_returns": bootstrap_returns,
                        "ev": None,  # filled post decision
                    }

                    decision = self.hub.decide(ctx)

                    # ---- DEBUG: dump TP/SL-ish keys from decision.meta (once per N cycles)
                    if log_this_cycle and decision:
                        meta = decision.get("meta") or {}

                        keys = [
                            "mc_tp", "mc_sl", "tp", "sl",
                            "profit_target", "stop_loss",
                            "tp_pct", "sl_pct",
                            "params",
                        ]

                        picked = {}
                        for k in keys:
                            if k in meta:
                                picked[k] = meta.get(k)

                        params = meta.get("params") or {}
                        if isinstance(params, dict):
                            for k in ["profit_target", "stop_loss", "tp_pct", "sl_pct", "tp", "sl", "n_paths"]:
                                if k in params:
                                    picked[f"params.{k}"] = params.get(k)

                        self._log(f"[DBG_META_TPSL] {sym} action={decision.get('action')} picked={json.dumps(picked, ensure_ascii=False)}")

                    ctx["ev"] = decision.get("ev", 0.0)
                    side = decision.get("action_type") or decision.get("action")

                    price_bucket = round(float(price), 3)
                    cache_key = (sym, side, regime, price_bucket)
                    now_cache = time.time()
                    cached = self.mc_cache.get(cache_key)
                    if cached and (now_cache - cached[0] <= self.mc_cache_ttl):
                        # 메타 재사용
                        decision_meta = decision.get("meta") or {}
                        decision_meta.update(cached[1])
                        decision["meta"] = decision_meta
                    else:
                        self.mc_cache[cache_key] = (now_cache, decision.get("meta", {}))

                    # 러닝 통계 업데이트 (regime/session)
                    def _s(val, default=0.0):
                        try:
                            if val is None:
                                return float(default)
                            return float(val)
                        except Exception:
                            return float(default)

                    self.stats.push("ev", (regime, session), _s(decision.get("ev")))
                    self.stats.push("cvar", (regime, session), _s(decision.get("cvar")))
                    spread_val = _s(decision.get("meta", {}).get("spread_pct", ctx.get("spread_pct")), 0.0)
                    self.stats.push("spread", (regime, session), spread_val)
                    self.stats.push("liq", (regime, session), _s(self._liquidity_score(sym)))
                    self.stats.push("ofi", (regime, session), _s(ofi_score))
                    ofi_mean = self.stats.ema_update("ofi_mean", (regime, session), ofi_score, half_life_sec=900)
                    ofi_res = ofi_score - ofi_mean
                    self.stats.push("ofi_res", (regime, session), ofi_res)

                    # 동적 레버리지(리스크 기반)
                    dyn_leverage = float(decision.get("leverage") or decision.get("meta", {}).get("lev") or self.leverage)
                    ctx["leverage"] = dyn_leverage


                    # 레짐별 사이즈 상한 적용
                    cap_map = {"bull": 0.25, "bear": 0.25, "chop": 0.10, "volatile": 0.08}
                    cap_frac_regime = cap_map.get(regime, 0.10)
                    decision = dict(decision)
                    decision_meta = dict(decision.get("meta") or {})
                    decision_meta["regime_cap_frac"] = cap_frac_regime
                    decision["meta"] = decision_meta
                    sz = decision.get("size_frac") or decision_meta.get("size_fraction") or self.default_size_frac
                    decision["size_frac"] = float(min(max(0.0, sz), cap_frac_regime))

                    # 추가 하드 게이트: event_cvar_r, spread_pct 상한
                    spread_pct_now = decision_meta.get("spread_pct", ctx.get("spread_pct"))
                    ev_cvar_r = decision_meta.get("event_cvar_r")
                    # 레짐별 스프레드 상한
                    spread_cap_map = {
                        "bull": 0.0020,
                        "bear": 0.0020,
                        "chop": 0.0012,
                        "volatile": 0.0008,
                    }
                    spread_cap = spread_cap_map.get(regime, SPREAD_PCT_MAX)
                    if spread_pct_now is not None and spread_cap is not None and spread_pct_now > spread_cap:
                        decision["action"] = "WAIT"
                        decision["reason"] = f"{decision.get('reason','')} | spread_cap"
                    # 레짐별 event_cvar_r 하한
                    cvar_floor_map = {
                        "bull": -1.2,
                        "bear": -1.2,
                        "chop": -1.0,
                        "volatile": -0.8,
                    }
                    cvar_floor_regime = cvar_floor_map.get(regime, -1.0)
                    if ev_cvar_r is not None and ev_cvar_r < cvar_floor_regime:
                        decision["action"] = "WAIT"
                        decision["reason"] = f"{decision.get('reason','')} | event_cvar_floor"

                    # EV 동적 진입 문턱: 최근 30분 EV의 p80 (레짐별) EMA(half-life 10m)
                    ev_net_now = float(decision.get("ev", 0.0) or 0.0)
                    hist_sym = self._ev_tune_hist[sym]
                    hist_sym.append((ts, ev_net_now))
                    cutoff = ts - int(EV_TUNE_WINDOW_SEC * 1000)
                    while hist_sym and hist_sym[0][0] < cutoff:
                        hist_sym.popleft()

                    regime_key = (regime or "chop", session or "OFF")
                    hist_reg = self._ev_regime_hist.setdefault(regime_key, deque(maxlen=4000))
                    hist_reg.append((ts, ev_net_now))
                    while hist_reg and hist_reg[0][0] < cutoff:
                        hist_reg.popleft()

                    dyn_enter_floor = None
                    ev_vals = [x[1] for x in hist_reg]
                    if len(ev_vals) >= EV_TUNE_MIN_SAMPLES:
                        try:
                            raw_thr = float(np.percentile(ev_vals, 80))
                            # EMA half-life 10m (600s)
                            prev = self._ev_thr_ema.get(regime_key)
                            prev_ts = self._ev_thr_ema_ts.get(regime_key, ts)
                            dt_sec = max(1.0, (ts - prev_ts) / 1000.0)
                            alpha = 1.0 - math.exp(-math.log(2) * dt_sec / 600.0)
                            ema = raw_thr if prev is None else (alpha * raw_thr + (1 - alpha) * prev)
                            ema = float(max(EV_ENTER_FLOOR_MIN, min(EV_ENTER_FLOOR_MAX, ema)))
                            self._ev_thr_ema[regime_key] = ema
                            self._ev_thr_ema_ts[regime_key] = ts
                            dyn_enter_floor = ema
                        except Exception:
                            dyn_enter_floor = None

                    if dyn_enter_floor is not None:
                        decision = dict(decision)
                        meta_tmp = dict(decision.get("meta") or {})
                        meta_tmp["ev_entry_threshold_dyn"] = dyn_enter_floor
                        decision["meta"] = meta_tmp

                    consensus_action, consensus_score = self._consensus_action(decision, ctx)
                    if decision.get("action") == "WAIT" and consensus_action in ("LONG", "SHORT") and decision.get("ev",0.0) > 0 and decision.get("confidence",0.0) >= 0.60:
                        decision = dict(decision)
                        decision["action"] = consensus_action
                        decision["reason"] = f"{decision.get('reason', '')} | consensus {consensus_action} score={consensus_score:.2f}"
                        decision["details"] = decision.get("details", [])
                        decision["details"].append({
                            "_engine": "consensus",
                            "_weight": 0.5,
                            "action": consensus_action,
                            "ev": decision.get("ev", 0.0),
                            "confidence": decision.get("confidence", 0.0),
                            "reason": f"consensus {consensus_score:.2f}",
                            "meta": {"consensus_score": consensus_score, "consensus_used": True},
                        })
                        decision["size_frac"] = decision.get("size_frac") or decision.get("meta", {}).get("size_fraction") or self.default_size_frac

                    # ---- HOLD / EXIT decision using event-based MC (remaining TP/SL) ----
                    exited_by_event = False
                    pos = self.positions.get(sym)
                    if pos and decision and ctx.get("mu_base") is not None and ctx.get("sigma", 0.0) > 0:
                        mu_evt, sigma_evt = adjust_mu_sigma(
                            float(ctx.get("mu_base", 0.0)),
                            float(ctx.get("sigma", 0.0)),
                            str(ctx.get("regime", "chop")),
                        )
                        seed_evt = int(time.time()) ^ hash(sym)
                        entry = float(pos.get("entry_price", price))
                        price_now = float(price)

                        meta = decision.get("meta") or {}
                        if not meta:
                            for d in decision.get("details", []):
                                if d.get("_engine") in ("mc_barrier", "mc_engine", "mc"):
                                    meta = d.get("meta") or {}
                                    break
                        params_meta = meta.get("params") or {}
                        tp_pct = decision.get("mc_tp") or meta.get("mc_tp") or params_meta.get("profit_target") or 0.001
                        sl_pct = decision.get("mc_sl") or meta.get("mc_sl") or (tp_pct * 0.8)
                        tp_pct = float(max(tp_pct, 1e-6))
                        sl_pct = float(max(sl_pct, 1e-6))

                        tp_rem = max((entry * (1 + tp_pct) / price_now) - 1.0, 1e-6)
                        sl_rem = max(1.0 - (entry * (1 - sl_pct) / price_now), 1e-6)

                        m_evt = mc_first_passage_tp_sl_jax(
                            s0=price_now,
                            tp_pct=tp_rem,
                            sl_pct=sl_rem,
                            mu=mu_evt,
                            sigma=sigma_evt,
                            dt=1.0,
                            max_steps=240,
                            n_paths=int(decision.get("n_paths", meta.get("params", {}).get("n_paths", 2048))),
                            seed=seed_evt,
                            dist=ctx.get("tail_model", "student_t"),
                            df=ctx.get("tail_df", 6.0),
                            boot_rets=ctx.get("bootstrap_returns"),
                        )

                        if m_evt:
                            ev_r_evt = float(m_evt.get("event_ev_r", 0.0) or 0.0)
                            cvar_r_evt = float(m_evt.get("event_cvar_r", 0.0) or 0.0)
                            p_sl_evt = float(m_evt.get("event_p_sl", 0.0) or 0.0)
                            ev_pct_evt = ev_r_evt * sl_rem
                            cvar_pct_evt = cvar_r_evt * sl_rem
                        else:
                            ev_pct_evt = 0.0
                            cvar_pct_evt = 0.0
                            p_sl_evt = 0.0

                        if m_evt and (
                            (ev_pct_evt < -0.0005)
                            or (cvar_pct_evt < -0.006)
                            or p_sl_evt >= 0.55
                        ):
                            self._log(
                                f"[{sym}] EXIT by MC "
                                f"(EV%={ev_pct_evt*100:.2f}%, "
                                f"CVaR%={cvar_pct_evt*100:.2f}%, "
                                f"P_SL={p_sl_evt:.2f})"
                            )
                            self._close_position(sym, price_now, "event_mc_exit")
                            exited_by_event = True

                    # 정책 기반 event MC exit (메타값 기반)
                    if (not exited_by_event) and sym in self.positions and decision:
                        pos = self.positions.get(sym) or {}
                        meta = decision.get("meta") or {}
                        age_sec = (ts - int(pos.get("entry_time", ts))) / 1000.0
                        do_exit, reason = should_exit_position(pos, meta, age_sec=age_sec, policy=self.exit_policy)
                        if do_exit:
                            self._close_position(sym, float(price), f"MC_EXIT:{reason}")
                            exited_by_event = True
                        else:
                            # EMA 기반 EV/PSL 악화 감지
                            ev_now = float(decision.get("ev", 0.0) or 0.0)
                            p_sl_now = float(meta.get("event_p_sl", 0.0) or 0.0)
                            ev_ema = self._ema_update(self._ema_ev, sym, ev_now, half_life_sec=30, ts_ms=ts)
                            psl_ema = self._ema_update(self._ema_psl, sym, p_sl_now, half_life_sec=30, ts_ms=ts)
                            d_ev = ev_now - ev_ema
                            d_psl = p_sl_now - psl_ema
                            ev_floor_reg = EV_EXIT_FLOOR.get(regime, -0.0002)
                            ev_drop_reg = EV_DROP.get(regime, 0.0008)
                            psl_rise_reg = PSL_RISE.get(regime, 0.03)
                            persist_need = 1 if regime in ("bull", "bear") else 2
                            if (ev_now < ev_floor_reg) and (d_ev < -ev_drop_reg) and (d_psl > psl_rise_reg):
                                self._exit_bad_ticks[sym] = self._exit_bad_ticks.get(sym, 0) + 1
                            else:
                                self._exit_bad_ticks[sym] = 0
                            if self._exit_bad_ticks[sym] >= persist_need:
                                self._close_position(sym, float(price), "ev_psl_ema_exit", exit_kind="RISK")
                                exited_by_event = True

                    self._maybe_exit_position(sym, float(price), decision, ts)

                    # 열린 포지션에도 최신 동적 레버리지 반영
                    if sym in self.positions:
                        self.positions[sym]["leverage"] = dyn_leverage

                    if not exited_by_event:
                        if decision.get("action") in ("LONG", "SHORT") and sym in self.positions:
                            self._rebalance_position(sym, float(price), decision, leverage_override=dyn_leverage)

                        # EV 급락 기반 추가 exit
                        if sym in self.positions:
                            ev_now = float(decision.get("ev", 0.0) or 0.0)
                            meta_now = decision.get("meta") or {}
                            ev_floor = float(meta_now.get("ev_entry_threshold_dyn") or meta_now.get("ev_entry_threshold") or 0.0)
                            prev_ev = self._ev_drop_state[sym].get("prev")
                            delta_ev = None
                            if prev_ev is not None:
                                delta_ev = ev_now - prev_ev
                            # 강신호 판정
                            strong_signal = decision.get("confidence", 0.0) >= 0.65
                            needed_ticks = 1 if strong_signal else 2
                            if ev_now < ev_floor and (delta_ev is not None) and (delta_ev < -EV_DROP_THRESHOLD):
                                self._ev_drop_state[sym]["streak"] += 1
                            else:
                                self._ev_drop_state[sym]["streak"] = 0
                            self._ev_drop_state[sym]["prev"] = ev_now
                            if self._ev_drop_state[sym]["streak"] >= needed_ticks:
                                self._close_position(sym, float(price), "ev_drop_exit", exit_kind="RISK")
                                exited_by_event = True

                        if decision.get("action") in ("LONG", "SHORT") and sym not in self.positions:
                            permit, deny_reason = self._entry_permit(sym, decision, ts)
                            if permit:
                                self._enter_position(sym, decision["action"], float(price), decision, ts, ctx=ctx, leverage_override=dyn_leverage)
                            else:
                                if log_this_cycle:
                                    self._log(f"[{sym}] skip entry (permit: {deny_reason}) ev={decision.get('ev',0):.4f} win={decision.get('confidence',0):.2f}")

                    if log_this_cycle and decision:
                        meta = decision.get("meta") or {}
                        self._log(
                            f"[DECISION] {sym} action={decision.get('action')} "
                            f"ev={decision.get('ev', 0.0):.4f} "
                            f"win={decision.get('confidence', 0.0):.2f} "
                            f"size={decision.get('size_frac', meta.get('size_fraction', 0.0)):.3f} "
                            f"reason={decision.get('reason', '')}"
                        )

                    rows.append(self._row(sym, float(price), ts, decision, candles))
            except Exception as e:
                import traceback
                self._log_err(f"[ERR] decision_loop: {e} {traceback.format_exc()}")

            try:
                if self.spread_enabled:
                    self._manage_spreads(ts)
                await self.broadcast(rows)
            except Exception as e2:
                self._log_err(f"[ERR] broadcast: {e2}")

            await asyncio.sleep(0.1)


async def index_handler(request):
    return web.FileResponse(str(DASHBOARD_FILE))


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    orch: LiveOrchestrator = request.app["orchestrator"]
    orch.clients.add(ws)
    await ws.send_str(json.dumps({"type": "init", "msg": "connected"}, ensure_ascii=False))

    async for _ in ws:
        pass

    orch.clients.discard(ws)
    return ws


async def main():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "timeout": CCXT_TIMEOUT_MS,
    })
    orchestrator = LiveOrchestrator(exchange)

    runner = None
    site = None
    try:
        # ✅ 서버 올리기 전에 OHLCV 먼저 채우기
        await orchestrator.preload_all_ohlcv(limit=OHLCV_PRELOAD_LIMIT)
        orchestrator._persist_state(force=True)

        app = web.Application()
        app["orchestrator"] = orchestrator
        app.add_routes([web.get("/", index_handler), web.get("/ws", ws_handler)])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()

        asyncio.create_task(orchestrator.fetch_prices_loop())
        asyncio.create_task(orchestrator.fetch_ohlcv_loop())
        asyncio.create_task(orchestrator.fetch_orderbook_loop())
        asyncio.create_task(orchestrator.decision_loop())

        print(f"🚀 Dashboard: http://localhost:{PORT}")
        print(f"📄 Serving: {DASHBOARD_FILE.name}")
        await asyncio.Future()
    except OSError as e:
        print(f"[ERR] Failed to bind on port {PORT}: {e}")
    finally:
        try:
            await exchange.close()
        except Exception:
            pass
        if site is not None:
            try:
                await site.stop()
            except Exception:
                pass
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
