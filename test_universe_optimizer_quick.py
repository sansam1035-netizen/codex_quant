#!/usr/bin/env python3
# test_universe_optimizer_quick.py
# ================================================================
# 빠른 테스트 (3개 종목 x 2개 전략)
# ================================================================

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from engines.adaptive_universe_optimizer import AdaptiveUniverseOptimizer


def main():
    print("\n" + "="*70)
    print("🧪 빠른 테스트: 종목 선정 + 전략 최적화")
    print("="*70)
    print("테스트 설정:")
    print("  • 종목: 3개 (BTC, ETH, BNB)")
    print("  • 전략: 2개 (momentum, mean_reversion)")
    print("  • 기간: 7일 (빠른 테스트)")
    print("="*70 + "\n")
    
    # 테스트용 설정
    optimizer = AdaptiveUniverseOptimizer({
        'evaluation_metric': 'sharpe',
        'lookback_days': 7,  # 매우 짧은 기간
        'top_n_symbols': 3,
        'min_sharpe': -1.0,  # 매우 낮은 기준
        'max_drawdown': 0.99,  # 거의 모든 종목 허용
        'min_trades': 1,  # 최소 1회 거래만
        'candidate_symbols': [
            'BTC/USDT:USDT',
            'ETH/USDT:USDT',
            'BNB/USDT:USDT'
        ],
        'strategies': [
            'momentum',
            'mean_reversion'
        ]
    })
    
    # 평가 실행 (순차)
    print("📊 평가 시작...\n")
    df_results = optimizer.run_full_evaluation(parallel=False)
    
    if df_results.empty:
        print("\n❌ 평가 결과가 없습니다.")
        return
    
    print(f"\n✅ 평가 완료: {len(df_results)}개 조합")
    print("\n" + "="*70)
    print("📊 전체 결과:")
    print("="*70)
    print(df_results.to_string(index=False))
    
    # 최적 포트폴리오 선정
    print("\n")
    portfolio = optimizer.select_optimal_portfolio(df_results)
    
    if portfolio:
        print("\n" + "="*70)
        print("✅ 테스트 성공!")
        print("="*70)
        print(f"\n선정 종목: {portfolio['selected_symbols']}")
        print(f"종목-전략: {portfolio['symbol_strategies']}")
        print(f"예상 Sharpe: {portfolio['expected_metrics']['avg_sharpe']:.2f}")
        print("\n💡 실제 실행:")
        print("   $ python3 run_universe_optimization.py --days 60 --top-n 8")


if __name__ == '__main__':
    main()

