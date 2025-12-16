#!/usr/bin/env python3
"""
횡보장 공격적 스캘핑 전략 백테스트
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.aggressive_sideways_scalper import AggressiveSidewaysScalper

def fetch_recent_data(symbol='BTC/USDT', days=7):
    """최근 데이터 가져오기"""
    exchange = ccxt.binance({'enableRateLimit': True})
    
    since = exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
    ohlcv = []
    
    while len(ohlcv) < days * 24 * 60:  # 1분봉
        try:
            data = exchange.fetch_ohlcv(symbol, '1m', since=since, limit=1000)
            if not data:
                break
            ohlcv.extend(data)
            since = data[-1][0] + 1
            print(f"Downloaded {len(ohlcv)} candles...")
        except Exception as e:
            print(f"Error: {e}")
            break
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def backtest_scalper(df, initial_balance=10000):
    """스캘핑 전략 백테스트"""
    scalper = AggressiveSidewaysScalper()
    
    balance = initial_balance
    position = 0
    entry_price = None
    entry_time = None
    
    trades = []
    equity_curve = []
    
    print(f"\n{'='*60}")
    print(f"🔥 Aggressive Sideways Scalper Backtest")
    print(f"{'='*60}")
    print(f"Initial Balance: ${balance:,.0f}")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Candles: {len(df)}")
    print(f"{'='*60}\n")
    
    for i in range(100, len(df)):  # 최소 100봉 필요
        current_df = df.iloc[:i+1]
        current = current_df.iloc[-1]
        
        # 신호 생성
        signal_result = scalper.get_signal(
            current_df, 
            current_position=position,
            entry_price=entry_price
        )
        
        signal = signal_result['signal']
        reason = signal_result['reason']
        confidence = signal_result['confidence']
        
        # 거래 실행
        if signal == 1 and position == 0:  # LONG 진입
            position = 1
            entry_price = current['close']
            entry_time = current.name
            print(f"[{current.name}] 🟢 LONG  @ ${entry_price:.2f} | {reason}")
            
        elif signal == -1 and position == 0:  # SHORT 진입
            position = -1
            entry_price = current['close']
            entry_time = current.name
            print(f"[{current.name}] 🔴 SHORT @ ${entry_price:.2f} | {reason}")
            
        elif signal == 2 and position > 0:  # LONG 청산
            exit_price = current['close']
            profit_pct = (exit_price - entry_price) / entry_price
            profit_usd = balance * profit_pct * 0.95  # 수수료 0.05%
            balance += profit_usd
            
            trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'profit_pct': profit_pct,
                'profit_usd': profit_usd,
                'duration': (current.name - entry_time).total_seconds() / 60
            })
            
            print(f"[{current.name}] ✅ CLOSE LONG @ ${exit_price:.2f} | {reason} | "
                  f"P/L: {profit_pct:+.2%} (${profit_usd:+.2f}) | Balance: ${balance:.2f}")
            
            position = 0
            entry_price = None
            
        elif signal == -2 and position < 0:  # SHORT 청산
            exit_price = current['close']
            profit_pct = (entry_price - exit_price) / entry_price
            profit_usd = balance * profit_pct * 0.95
            balance += profit_usd
            
            trades.append({
                'entry': entry_price,
                'exit': exit_price,
                'profit_pct': profit_pct,
                'profit_usd': profit_usd,
                'duration': (current.name - entry_time).total_seconds() / 60
            })
            
            print(f"[{current.name}] ✅ CLOSE SHORT @ ${exit_price:.2f} | {reason} | "
                  f"P/L: {profit_pct:+.2%} (${profit_usd:+.2f}) | Balance: ${balance:.2f}")
            
            position = 0
            entry_price = None
        
        equity_curve.append(balance)
    
    # 결과 분석
    if trades:
        df_trades = pd.DataFrame(trades)
        wins = len(df_trades[df_trades['profit_pct'] > 0])
        losses = len(df_trades[df_trades['profit_pct'] <= 0])
        win_rate = wins / len(trades) * 100
        avg_win = df_trades[df_trades['profit_pct'] > 0]['profit_pct'].mean() * 100
        avg_loss = df_trades[df_trades['profit_pct'] <= 0]['profit_pct'].mean() * 100
        avg_duration = df_trades['duration'].mean()
        
        total_return = (balance - initial_balance) / initial_balance * 100
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Trades:     {len(trades)}")
        print(f"Wins:             {wins} ({win_rate:.1f}%)")
        print(f"Losses:           {losses}")
        print(f"Avg Win:          +{avg_win:.2f}%")
        print(f"Avg Loss:         {avg_loss:.2f}%")
        print(f"Avg Duration:     {avg_duration:.1f} minutes")
        print(f"")
        print(f"Initial Balance:  ${initial_balance:,.2f}")
        print(f"Final Balance:    ${balance:,.2f}")
        print(f"Total Return:     {total_return:+.2f}%")
        print(f"")
        print(f"Expected Daily:   ~{len(trades)/7:.0f} trades/day")
        print(f"Expected Daily %: {total_return/7:+.2f}% per day")
        print(f"{'='*60}\n")
    else:
        print("\n⚠️  No trades executed")

if __name__ == "__main__":
    # 최근 7일 데이터로 테스트
    print("📥 Downloading recent data...")
    df = fetch_recent_data('BTC/USDT', days=7)
    
    # 백테스트 실행
    backtest_scalper(df, initial_balance=10000)


