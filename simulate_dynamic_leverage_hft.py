#!/usr/bin/env python3
# simulate_dynamic_leverage_hft.py
# ================================================================
# Dynamic Leverage in High Frequency Trading Simulation
# ================================================================

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import copy

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.unified_backtester import UnifiedBacktester
from engines.crypto_data_provider import CryptoDataProvider
from engines.unified_leverage_optimizer import UnifiedLeverageOptimizer

# ----------------------------------------------------------------
# 1. Custom Backtester for Dynamic Leverage
# ----------------------------------------------------------------
class DynamicLeverageBacktester(UnifiedBacktester):
    """
    UnifiedBacktester를 상속받아 Dynamic Leverage를 지원하도록 수정
    """
    def __init__(self, config):
        # Advanced Analysis 비활성화 (KeyError 방지)
        config['use_advanced_analysis'] = False
        # 기본 기능 비활성화 (HFT 전략 순수 테스트)
        config['use_factor_ranking'] = False
        config['use_vol_targeting'] = False
        config['use_drawdown_defense'] = False
        super().__init__(config)

    def _calculate_target_positions(
        self,
        signals: dict,
        window_data: dict,
        current_time: datetime
    ) -> dict:
        """
        Override: 신호에 포함된 'leverage' 정보를 반영하여 포지션 크기 계산
        """
        if not signals:
            return {}
        
        target_positions = {}
        portfolio_value = self._get_portfolio_value(window_data, current_time)
        
        # 기본 배분 (1/N)
        base_allocation = portfolio_value / len(signals)
        
        for symbol, signal_info in signals.items():
            signal = signal_info.get("signal", 0)
            confidence = signal_info.get("confidence", 1.0)
            leverage = signal_info.get("leverage", 1.0) # Dynamic Leverage
            
            if symbol not in window_data:
                continue
            
            df = window_data[symbol]
            price = df['close'].iloc[-1] if 'close' in df.columns else df.iloc[-1, -1]
            
            if price <= 0:
                continue
            
            # 수량 계산: (기본배분 * 레버리지) / 가격
            target_notional = base_allocation * leverage
            quantity = target_notional / price
            
            target_positions[symbol] = quantity * signal
            
        if target_positions:
            # Debug
            print(f"  [DEBUG] Targets at {current_time}: {target_positions}")
            pass
        
        return target_positions

    def _execute_orders(
        self,
        target_positions: dict,
        window_data: dict,
        current_time: datetime
    ):
        """
        Override: Margin Trading 허용 (Cash가 부족해도 매수 가능)
        """
        for symbol, target_qty in target_positions.items():
            current_qty = self.positions.get(symbol, 0)
            diff_qty = target_qty - current_qty
            
            if abs(diff_qty) < 0.0001:
                continue
            
            # 가격
            df = window_data[symbol]
            price = df['close'].iloc[-1] if 'close' in df.columns else df.iloc[-1, -1]
            
            # 거래 비용 계산
            side = "buy" if diff_qty > 0 else "sell"
            cost_info = self.cost_model.calculate_total_cost(
                "crypto", symbol, side, abs(diff_qty), price
            ) if self.use_transaction_cost else {'total': 0}
            
            # 체결
            trade_value = abs(diff_qty) * price
            total_cost = trade_value + cost_info['total']
            
            if side == "buy":
                # Margin Trading: Cash 체크 제거
                self.cash -= total_cost
                self.positions[symbol] = target_qty
                
                print(f"  [DEBUG] Executing BUY {symbol} {abs(diff_qty)}")
                self.trades.append({
                    'time': current_time,
                    'symbol': symbol,
                    'side': side,
                    'quantity': abs(diff_qty),
                    'price': price,
                    'cost': cost_info['total']
                })
            else:
                self.cash += trade_value - cost_info['total']
                self.positions[symbol] = target_qty
                
                print(f"  [DEBUG] Executing SELL {symbol} {abs(diff_qty)}")
                self.trades.append({
                    'time': current_time,
                    'symbol': symbol,
                    'side': side,
                    'quantity': abs(diff_qty),
                    'price': price,
                    'cost': cost_info['total']
                })

    def _calculate_results(self) -> dict:
        """
        Override: AnalysisIntegrator 호환성 해결 (total_value 컬럼 추가)
        """
        results = super()._calculate_results()
        
        if 'equity_curve' in results:
            df = results['equity_curve']
            if 'equity' in df.columns and 'total_value' not in df.columns:
                df['total_value'] = df['equity']
                results['equity_curve'] = df
                
        return results

# ----------------------------------------------------------------
# 2. HFT Strategy with Dynamic Leverage
# ----------------------------------------------------------------
class HFTDynamicStrategy:
    def __init__(self, use_dynamic_leverage=True):
        self.use_dynamic_leverage = use_dynamic_leverage
        self.leverage_optimizer = UnifiedLeverageOptimizer({
            'base_leverage': 1.5,
            'max_leverage': 3.0,
            'min_leverage': 0.5,
            'target_volatility': 0.15 # HFT는 변동성 허용폭을 조금 더 둠
        })
        
        # 종목별 성과 추적을 위한 간이 메모리
        self.perf_memory = {} 

    def calculate(self, window_data, current_time=None, positions=None):
        signals = {}
        
        for symbol, df in window_data.items():
            if len(df) < 50:
                continue
                
            # 15분봉 데이터 가정
            closes = df['close']
            returns = closes.pct_change()
            
            # 1. 시그널 생성 (RSI + Momentum)
            # RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Momentum (Short term)
            mom = closes.pct_change(5).iloc[-1] # 5봉 전 대비
            
            signal = 0
            if current_rsi < 40 and mom > 0: # 과매도 + 반등
                signal = 1
            elif current_rsi > 60 and mom < 0: # 과매수 + 하락
                signal = -1
                
            if signal == 0:
                continue
                
            # 2. Dynamic Leverage 계산
            leverage = 1.5 # Default Static
            
            if self.use_dynamic_leverage:
                # 간이 성과 지표 계산 (최근 50봉 기준)
                recent_returns = returns.iloc[-50:]
                win_rate = (recent_returns > 0).mean() if len(recent_returns) > 0 else 0.5
                volatility = recent_returns.std() * np.sqrt(252*24*4) # 연율화 (15분봉)
                sharpe = (recent_returns.mean() / recent_returns.std() * np.sqrt(252*24*4)) if recent_returns.std() > 0 else 0
                
                # Optimizer 호출
                leverage = self.leverage_optimizer.get_optimal_leverage(
                    portfolio_return=0.0, # 개별 종목 관점이라 0 처리
                    portfolio_volatility=0.0,
                    current_drawdown=0.0, # 개별 종목 DD는 복잡하므로 생략
                    strategy_confidence=0.8, # 기본 신뢰도
                    market_regime='neutral', # 레짐 엔진 연동 생략
                    win_rate=win_rate,
                    sharpe_ratio=sharpe,
                    market_volatility=volatility
                )
                
            signals[symbol] = {
                "signal": signal,
                "confidence": 1.0,
                "leverage": leverage
            }
            
        return signals

# ----------------------------------------------------------------
# 3. Simulation Runner
# ----------------------------------------------------------------
global_data_dict = {}

def run_simulation():
    print(f"\n{'='*70}")
    print(f"⚡️ High Frequency Trading (HFT) Dynamic Leverage Simulation")
    print(f"{'='*70}")
    
    # 1. 데이터 수집
    print("📥 데이터 수집 중 (15m, 14일)...")
    symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'XRP/USDT:USDT'
    ]
    
    provider = CryptoDataProvider({
        'exchange': 'bybit', 'type': 'linear', 'testnet': True
    })
    provider.connect()
    
    for sym in symbols:
        try:
            # 14일치 15분봉
            df = provider.fetch_ohlcv(sym, timeframe='15m', limit=14*24*4)
            if df is not None and not df.empty:
                global_data_dict[sym] = df
        except Exception as e:
            print(f"  ⚠️ {sym} 실패: {e}")
            
    if not global_data_dict:
        print("❌ 데이터 없음")
        return

    # 2. 시나리오 실행
    scenarios = [
        ("Static Leverage (1.5x)", False),
        ("Dynamic Leverage (0.5x ~ 3.0x)", True)
    ]
    
    results_summary = []
    
    for name, use_dynamic in scenarios:
        print(f"\n▶️ Running: {name}...")
        
        strategy = HFTDynamicStrategy(use_dynamic_leverage=use_dynamic)
        
        # Config
        config = {
            "initial_capital": 100000,
            "commission_rate": 0.0005, # HFT라 수수료 낮게 가정 (VIP 등급 등)
            "position_limit_guard": {
                "enabled": True,
                "max_leverage": 5.0 # 시뮬레이션 상 제한 풀기
            }
        }
        
        backtester = DynamicLeverageBacktester(config)
        res = backtester.run(global_data_dict, strategy.calculate)
        
        results_summary.append({
            "Name": name,
            "Return": res.get('total_return', 0) * 100,
            "Sharpe": res.get('sharpe_ratio', 0),
            "MDD": res.get('max_drawdown', 0) * 100,
            "Trades": res.get('total_trades', 0)
        })

    # 3. 결과 비교
    print(f"\n{'='*80}")
    print(f"📊 HFT Simulation Results")
    print(f"{'='*80}")
    print(f"{'Name':<30} | {'Return':>8} | {'Sharpe':>6} | {'MDD':>8} | {'Trades':>6}")
    print("-" * 80)
    
    for row in results_summary:
        print(f"{row['Name']:<30} | {row['Return']:8.2f}% | {row['Sharpe']:6.2f} | {row['MDD']:8.2f}% | {row['Trades']:6d}")
    print("-" * 80)

if __name__ == "__main__":
    run_simulation()
