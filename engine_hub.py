# engines/engine_hub.py

from engines.dummy_engine import DummyEngine


class EngineHub:
    """
    실전용 엔진 허브
    - 엔진 안전 로딩
    - EV 중심 통합
    """

    # =========================
    # sanitize (핵심!)
    # =========================
    @staticmethod
    def _sanitize(obj):
        # JAX -> host
        try:
            from jax import device_get  # type: ignore
            obj = device_get(obj)
        except Exception:
            pass

        # dict / list 재귀
        if isinstance(obj, dict):
            return {str(k): EngineHub._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [EngineHub._sanitize(v) for v in obj]

        # numpy / array-like
        try:
            import numpy as np
            arr = np.asarray(obj)

            # scalar
            if arr.ndim == 0:
                item = arr.item()
                if isinstance(item, (int, float, np.floating)):
                    return float(item)
                return str(item)

            # vector/matrix -> flat list
            return [
                float(x) if isinstance(x, (int, float, np.floating)) else str(x)
                for x in arr.reshape(-1).tolist()
            ]
        except Exception:
            pass

        # primitive
        if isinstance(obj, (int, float, bool)) or obj is None:
            return obj

        # fallback
        return str(obj)

    # =========================
    # lifecycle
    # =========================
    def __init__(self):
        self.engines = []
        self._load_engines()

        if not self.engines:
            self.engines.append(DummyEngine())

    def _load_engines(self):
        self._safe_load(self._load_mc_engine)

    def _safe_load(self, loader_fn):
        try:
            engine = loader_fn()
            self.engines.append(engine)
            print(f"✅ Engine loaded: {engine.name}")
        except Exception as e:
            print(f"⚠️ Engine skipped: {e}")

    def _load_mc_engine(self):
        from engines.mc_engine import MonteCarloEngine
        return MonteCarloEngine()

    # =========================
    # decision
    # =========================
    def decide(self, ctx: dict) -> dict:
        results = []

        for engine in self.engines:
            try:
                res = engine.decide(ctx)
                res["_engine"] = engine.name
                res["_weight"] = engine.weight
                # pass through event-based MC metrics
                meta = res.get("meta") or {}
                for k in (
                    "event_p_tp",
                    "event_p_sl",
                    "event_p_timeout",
                    "event_ev_r",
                    "event_cvar_r",
                    "event_t_median",
                    "event_t_mean",
                ):
                    if k in meta:
                        res[k] = meta[k]
                results.append(res)
            except Exception as e:
                results.append({
                    "action": "WAIT",
                    "ev": 0.0,
                    "confidence": 0.0,
                    "reason": f"{engine.name} error: {e}",
                    "_engine": engine.name,
                    "_weight": engine.weight,
                })

        total_ev = sum(r["ev"] * r["_weight"] for r in results)
        best = max(results, key=lambda r: r["ev"])

        final_action = best["action"] if total_ev > 0 else "WAIT"

        final = {
            "action": final_action,
            "ev": total_ev,
            "confidence": max(r.get("confidence", 0.0) for r in results),
            "reason": " | ".join(r.get("reason", "") for r in results),
            "details": results,
        }

        # 🔥 최종 경계에서 무조건 sanitize
        return EngineHub._sanitize(final)
