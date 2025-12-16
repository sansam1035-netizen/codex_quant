#!/usr/bin/env python3
# simulate_capital_utilization.py
# ================================================================
# 자본 효율성 개선 시뮬레이션 (레버리지 & 포지션 사이징)
# ================================================================

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import copy

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtest.unified_backtester import UnifiedBacktester
from engines.crypto_data_provider import CryptoDataProvider

def simple_momentum_strategy(window_data, current_time=None, positions=None):
    """간단한 모멘텀 전략 (15분봉 기준)"""
    signals = {}
    
    for symbol, df in window_data.items():
        if len(df) < 50:
            continue
            
        # 15분봉 기준 지표
        closes = df['close'] if 'close' in df.columns else df.iloc[:, -1]
        
        # 단기 모멘텀 (20봉 = 5시간)
        returns = closes.pct_change(20).iloc[-1]
        
        # 변동성
        vol = closes.pct_change().std()
        if np.isnan(vol) or vol == 0:
            vol = 0.01
            
        signal = 0
        # 진입 장벽 완화 (1% 변동)
        if returns > 0.01:
            signal = 1
        elif returns < -0.01:
            signal = -1
            
        # 신뢰도
        confidence = min(1.0, 0.01 / vol)
        
        if signal != 0:
            signals[symbol] = {
                "signal": signal,
                "confidence": confidence
            }
            
    return signals

def run_simulation(config_override, name):
    print(f"\n{'='*50}")
    print(f"🧪 시뮬레이션: {name}")
    print(f"{'='*50}")
    
    # 기본 설정
    base_config = {
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "use_transaction_cost": True,
        "use_factor_ranking": False,
        "use_vol_targeting": True,
        "use_drawdown_defense": True,
        "portfolio": {
            "max_weight_per_symbol": 0.1,  # 기본 10%
            "max_leverage": 1.5            # 기본 1.5배
        },
        "position_limit_guard": {
            "enabled": True,
            "max_single_position_pct": 0.1,
            "max_total_exposure_pct": 0.9,
            "max_leverage": 1.5
        }
    }
    
    # 설정 덮어쓰기
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
        
    config = update_nested_dict(copy.deepcopy(base_config), config_override)
    
    # 백테스터 실행
    backtester = UnifiedBacktester(config)
    
    # 데이터 (전역 변수 사용)
    results = backtester.run(
        data_dict=global_data_dict,
        strategy_func=simple_momentum_strategy
    )
    
    return results

# 전역 데이터 저장소
global_data_dict = {}

def main():
    # 1. 데이터 수집 (15분봉, 최근 30일)
    print("📥 데이터 수집 중 (15m, 30일)...")
    symbols = [
        'ADA/USDT:USDT', 'AAVE/USDT:USDT', 'OP/USDT:USDT', 'AXS/USDT:USDT', 
        'BCH/USDT:USDT', 'APE/USDT:USDT', 'CRV/USDT:USDT', 'MANA/USDT:USDT', 
        'ASTR/USDT:USDT', 'MINA/USDT:USDT', 'XTZ/USDT:USDT', 'UNI/USDT:USDT', 
        'THETA/USDT:USDT', 'EGLD/USDT:USDT', 'ETC/USDT:USDT', 'LDO/USDT:USDT', 
        'INJ/USDT:USDT', 'FLOW/USDT:USDT', 'TIA/USDT:USDT', 'APT/USDT:USDT'
    ]
    
    provider = CryptoDataProvider({
        'exchange': 'bybit', 'type': 'linear', 'testnet': True
    })
    provider.connect()
    
    for sym in symbols:
        try:
            df = provider.fetch_ohlcv(sym, timeframe='15m', limit=30*24*4)
            if df is not None and not df.empty:
                global_data_dict[sym] = df
        except Exception as e:
            print(f"  ⚠️ {sym} 실패: {e}")
            
    if not global_data_dict:
        print("❌ 데이터 없음")
        return

    # 2. 시나리오 정의
    scenarios = [
        {
            "name": "Baseline (Current)",
            "config": {} # 기본값 사용
        },
        {
            "name": "Option A: Position Size Up (20%)",
            "config": {
                "portfolio": {"max_weight_per_symbol": 0.2},
                "position_limit_guard": {"max_single_position_pct": 0.2}
            }
        },
        {
            "name": "Option B: Leverage Up (3.0x)",
            "config": {
                "portfolio": {"max_leverage": 3.0},
                "position_limit_guard": {"max_leverage": 3.0, "max_total_exposure_pct": 2.5}
            }
        },
        {
            "name": "Option C: Aggressive (Size 20% + Lev 3x)",
            "config": {
                "portfolio": {"max_weight_per_symbol": 0.2, "max_leverage": 3.0},
                "position_limit_guard": {"max_single_position_pct": 0.2, "max_leverage": 3.0, "max_total_exposure_pct": 2.5}
            }
        }
    ]
    
    # 3. 시뮬레이션 실행 및 비교
    summary = []
    
    for scenario in scenarios:
        res = run_simulation(scenario['config'], scenario['name'])
        
        summary.append({
            "Name": scenario['name'],
            "Return": res.get('total_return', 0) * 100,
            "Sharpe": res.get('sharpe_ratio', 0),
            "MDD": res.get('max_drawdown', 0) * 100,
            "Final Equity": res.get('final_equity', 0)
        })
        
    # 4. 결과 출력
    print(f"\n{'='*80}")
    print(f"📊 자본 효율성 시뮬레이션 결과 비교")
    print(f"{'='*80}")
    print(f"{'Name':<35} | {'Return':>8} | {'Sharpe':>6} | {'MDD':>8} | {'Equity':>12}")
    print("-" * 80)
    
    for row in summary:
        print(f"{row['Name']:<35} | {row['Return']:8.2f}% | {row['Sharpe']:6.2f} | {row['MDD']:8.2f}% | ${row['Final Equity']:,.0f}")
        
    print("-" * 80)
    
    # 추천
    best_sharpe = max(summary, key=lambda x: x['Sharpe'])
    best_return = max(summary, key=lambda x: x['Return'])
    
    print(f"\n💡 분석 결과:")
    print(f"   - 최고 수익률: {best_return['Name']} ({best_return['Return']:.2f}%)")
    print(f"   - 최고 효율(Sharpe): {best_sharpe['Name']} ({best_sharpe['Sharpe']:.2f})")
    
    if best_return['MDD'] < -30:
        print(f"   ⚠️ 주의: 최고 수익률 옵션의 낙폭({best_return['MDD']:.2f}%)이 큽니다.")
        print(f"   👉 추천: {best_sharpe['Name']} (안정성과 수익성의 균형)")
    else:
        print(f"   👉 추천: {best_return['Name']} (감당 가능한 리스크)")

if __name__ == "__main__":
    main()
