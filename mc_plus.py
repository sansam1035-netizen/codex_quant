# mc_plus.py 전체를 이 코드로 덮어쓰세요.

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import jit
from typing import Dict, List, Tuple

class KalmanFilter1D:
    def __init__(self, R=0.01, Q=1e-5):
        self.R = R
        self.Q = Q
        self.P = 1.0
        self.x = None
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        x_pred = self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

class OUProcess:
    def __init__(self, window=20):
        self.window = window
    def get_z_score(self, prices):
        if len(prices) < self.window: return 0.0
        log_prices = np.log(prices[-self.window:])
        mu = np.mean(log_prices)
        sigma = np.std(log_prices)
        if sigma < 1e-9: return 0.0
        return (log_prices[-1] - mu) / sigma

class LSMModel:
    @staticmethod
    @jit
    def calculate_values(paths, entry_price, direction, leverage, discount=0.9999):
        # paths shape: (sims, steps)
        current_price = jnp.mean(paths[:, 0])
        exercise_value = (current_price - entry_price) / entry_price * direction * leverage
        
        future_prices = paths[:, 1:]
        future_pnl = (future_prices - entry_price) / entry_price * direction * leverage
        
        n_steps = future_pnl.shape[1]
        discount_factors = jnp.power(discount, jnp.arange(1, n_steps + 1))
        discounted_pnl = future_pnl * discount_factors[None, :]
        
        continuation_value = jnp.mean(jnp.sum(discounted_pnl, axis=1))
        return exercise_value, continuation_value

class LeverageOptimizer:
    def __init__(self, max_leverage=10.0, kelly_fraction=0.5):
        self.max_leverage = max_leverage
        self.kelly_fraction = kelly_fraction
        
    def calculate_optimal_leverage(self, win_rate, avg_win, avg_loss, z_score, volatility):
        if avg_loss < 1e-9: avg_loss = 1e-9
        b_ratio = avg_win / abs(avg_loss)
        kelly_pct = win_rate - ((1 - win_rate) / b_ratio)
        if kelly_pct <= 0: return 1.0
        
        safe_kelly = kelly_pct * self.kelly_fraction
        target_vol = 0.02
        vol_leverage = target_vol / (volatility + 1e-9)
        
        raw_leverage = min(safe_kelly * 10, vol_leverage)
        
        ou_penalty = 1.0
        if abs(z_score) > 1.0:
            ou_penalty = max(0.0, 1.0 - (abs(z_score) - 1.0) * 0.5)
            
        final_leverage = np.clip(raw_leverage * ou_penalty, 1.0, self.max_leverage)
        return float(round(final_leverage * 2) / 2)

class QuantDecisionEngine:
    def __init__(self):
        self.kalman = KalmanFilter1D()
        self.ou = OUProcess(window=20)
        self.lsm = LSMModel()
        
    def decide(self, mc_engine, symbol, current_price, position, historical_prices, market_data, win_probability=0.5):
        # 1. 절대 손절 (-1.5%)
        pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * position['direction'] * position['leverage']
        if pnl_pct < -0.015: return "CLOSE", f"🛑 Stop Loss (-1.5%)"
        
        # 2. 최소 보유 시간 (스캘핑이므로 3분으로 단축)
        import time
        if time.time() - position['entry_time'] < 180: # 3분
             if pnl_pct > -0.005: # 큰 손실 아니면 좀 더 지켜봄
                 return "HOLD", "⏳ Min Hold (3m)"

        # 3. [수정] LSMC Horizon 동기화 (15분 예측)
        # 기존: 48시간 -> 수정: 15분 (1분봉 기준 15개)
        mc_paths = mc_engine.generate_raw_paths(
            symbol=symbol,
            current_price=current_price,
            mu=market_data['predicted_mu'],
            sigma=market_data['volatility'],
            n_steps=15,    # [변경] 15 steps (15분)
            dt=1/525600,   # [변경] 1분 단위 (1년=525600분)
            n_paths=5000
        )
        
        exercise_val, continue_val = self.lsm.calculate_values(
            mc_paths, position['entry_price'], position['direction'], position['leverage']
        )
        
        # 판결 로직 (Scalping에 맞게 민감도 조절)
        score_close = 0
        
        # 미래가치보다 현재가치가 더 크면 (즉, 고점 찍고 내려갈 것 같으면)
        if exercise_val > continue_val * 1.01: # 1% 더 버는 것보다 지금 파는게 낫다
            score_close += 50
            
        # OU 과열 (Z-Score > 2.5)
        z_score = self.ou.get_z_score(np.array(historical_prices))
        if abs(z_score) > 2.5:
            score_close += 30
            
        if score_close >= 50:
            return "CLOSE", f"📉 Optimal Exit (Val:{exercise_val:.4f} > Fut:{continue_val:.4f})"
            
        return "HOLD", f"💎 Holding (Upside Left)"