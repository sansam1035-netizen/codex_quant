#!/usr/bin/env python3
"""
빠른 테스트: 기존 모델이 정말 아무것도 안하는지 확인
"""
import os
import sys
import numpy as np
import torch
import ccxt
import pandas as pd
from tqdm import tqdm

# 기존 코드 임포트
sys.path.insert(0, os.path.dirname(__file__))
from strategies.Final_Transformer_Edition import (
    CONFIG, SimpleCryptoData, AdvancedFeatureEngine, 
    C51Agent, TradingEnv_V2, DEVICE
)

def test_model():
    print("="*70)
    print("🔍 Testing Existing Model - Action Distribution Check")
    print("="*70)
    
    # 데이터 로드 (한 개만 빠르게)
    dp = SimpleCryptoData()
    fe = AdvancedFeatureEngine()
    
    print("\n📥 Fetching recent data (30 days)...")
    df = dp.fetch_data('BTC/USDT', CONFIG['timeframe'], 30)
    df = fe.add_features(df)
    print(f"✅ Loaded: {df.shape}")
    
    # 에이전트 로드
    input_dim = (df.shape[1] - 1) + 1
    agent = C51Agent(input_dim)
    
    if os.path.exists("best_transformer_brain.pth"):
        agent.load("best_transformer_brain.pth")
        print("✅ Model loaded\n")
    else:
        print("⚠️  No model found. Using random agent.\n")
    
    # 환경 생성
    env = TradingEnv_V2(df)
    env.set_difficulty(1.0, 1.0)  # 실전 난이도
    state = env.reset()
    
    # 액션 카운터
    action_counts = {0: 0, 1: 0, 2: 0}
    action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}
    
    print("🔄 Running simulation...")
    for _ in tqdm(range(len(df)-CONFIG['seq_len']-1)):
        action = agent.act(state, training=False)
        action_counts[action] += 1
        next_state, _, done, balance = env.step(action)
        state = next_state
        if done:
            break
    
    # 결과 출력
    total_actions = sum(action_counts.values())
    print("\n" + "="*70)
    print("📊 Action Distribution:")
    print("="*70)
    
    for action_id, count in action_counts.items():
        pct = count / total_actions * 100
        bar_length = int(pct / 2)  # 50% = 25 chars
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{action_names[action_id]:>6}: [{bar}] {pct:5.1f}% ({count:>4} times)")
    
    print("="*70)
    print(f"💰 Final Balance: ${balance:.2f}")
    print(f"📈 Profit: {(balance-10000)/10000*100:+.2f}%")
    print("="*70)
    
    # 진단
    print("\n🔍 Diagnosis:")
    if action_counts[0] > total_actions * 0.9:
        print("⚠️  ZOMBIE DETECTED! Model is doing almost nothing (>90% HOLD)")
        print("   Recommendation: Increase hold_penalty and action_reward")
    elif action_counts[0] > total_actions * 0.7:
        print("⚠️  Too passive (>70% HOLD)")
        print("   Recommendation: Adjust reward structure")
    else:
        print("✅ Model is actively trading")
    
    # 롱/숏 밸런스
    if action_counts[1] > 0 or action_counts[2] > 0:
        long_short_ratio = action_counts[1] / (action_counts[2] + 1)
        print(f"\n📊 Long/Short Ratio: {long_short_ratio:.2f}")
        if long_short_ratio > 10:
            print("   ⚠️  Only going LONG (might be biased)")
        elif long_short_ratio < 0.1:
            print("   ⚠️  Only going SHORT (might be biased)")

if __name__ == "__main__":
    test_model()


