import ccxt
import time
import numpy as np
import threading
from datetime import datetime, timedelta

# =========================================================
# PART 1. 🦅 The Eyes: 미세 호가창 정밀 분석기
# =========================================================
class OrderBookScanner:
    """
    호가창(Order Book)의 불균형(Imbalance)과 거대한 벽(Wall)을 탐지
    """
    def __init__(self):
        pass

    def analyze_book(self, bids, asks):
        """
        [핵심 로직]
        1. OBI(Order Book Imbalance) 계산
        2. '벽(Wall)' 탐지: 평균 물량 대비 N배 이상 튀는 물량이 있는지 확인
        """
        # 상위 10개 호가만 분석 (단타용)
        top_bids = bids[:10]
        top_asks = asks[:10]
        
        # 1. OBI 계산 (매수세 vs 매도세 힘싸움)
        bid_vol = sum([x[1] for x in top_bids])
        ask_vol = sum([x[1] for x in top_asks])
        
        if bid_vol + ask_vol == 0:
            obi = 0
        else:
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol)

        # 2. 벽(Wall) 탐지 로직
        # "평균적인 호가 잔량보다 3배 이상 쌓여있으면 벽으로 간주"
        avg_ask_vol = ask_vol / len(top_asks)
        avg_bid_vol = bid_vol / len(top_bids)
        
        # 매도벽(Resistance) 체크: 바로 위(1~3호가)에 벽이 있나?
        resistance_detected = False
        limit_price = 0
        
        for price, vol in top_asks[:3]: # 가까운 3개 호가만 검사
            if vol > avg_ask_vol * 3.0: # 평균보다 3배 큰 물량 발견
                resistance_detected = True
                limit_price = price
                break # 벽 발견
        
        # 매수벽(Support) 체크
        support_detected = False
        
        for price, vol in top_bids[:3]:
            if vol > avg_bid_vol * 3.0:
                support_detected = True
                break

        # 3. 체결 강도 계산 (Trade Execution Intensity)
        # 최상단 호가에서의 잔량 비율로 체결 강도 측정
        if top_bids and top_asks:
            best_bid_vol = top_bids[0][1] if top_bids else 0
            best_ask_vol = top_asks[0][1] if top_asks else 0
            total_best_vol = best_bid_vol + best_ask_vol
            avg_best_vol = total_best_vol / 2.0
            
            # 체결 강도: 최상단 호가의 평균 잔량을 전체 평균과 비교
            execution_intensity = avg_best_vol / max(avg_bid_vol, 0.0001) if avg_bid_vol > 0 else 0.0
        else:
            execution_intensity = 0.0

        return {
            "obi": obi,                      # +1(매수우위) ~ -1(매도우위)
            "resistance": resistance_detected, # 매도벽 유무 (True/False)
            "support": support_detected,       # 매수벽 유무 (True/False)
            "resistance_price": limit_price,    # 매도벽 가격 (이거 아래로 주문 넣어야 함)
            "execution_intensity": execution_intensity  # 체결 강도 (높을수록 체결 가능성 높음)
        }

# =========================================================
# PART 2. 🧠 The Controller: LSMC & Execution 통합 제어
# =========================================================
class AlphaBotMain:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key, 
            'secret': secret,
            'options': {'defaultType': 'future'}
        })
        self.scanner = OrderBookScanner()
        
        # 봇 상태 관리
        self.lsmc_signal = None      # "LONG", "SHORT", "EXIT"
        self.signal_time = None      # 신호 발생 시각
        self.signal_ttl = 30         # 신호 유효기간 (초) - 30초 지나면 폐기
        
        self.in_position = False
        self.position_direction = None  # "LONG" or "SHORT" (포지션 방향 추적)
        self.position_size = 0

    # (가상의 LSMC 엔진 - 실제로는 GPU 코드 연동)
    def fetch_lsmc_signal(self):
        # 여기서는 테스트를 위해 랜덤 신호 생성
        # 실제로는 사용자님의 LSMC 코드가 여기서 return을 해줘야 함
        import random
        rand = random.random()
        if rand > 0.95: return "LONG"
        if 0.05 < rand <= 0.10: return "SHORT"  # SHORT 신호 추가
        if rand < 0.05: return "EXIT"
        return "WAIT"

    def run(self):
        print("🤖 Dual-Layer Bot Started...")
        
        while True:
            try:
                # 1. 데이터 수집 (1초 단위)
                # fetch_order_book은 API call이므로 너무 자주 하면 밴 당함 (0.5~1초 간격 권장)
                orderbook = self.exchange.fetch_order_book('BTC/USDT', limit=20)
                bids = orderbook['bids']
                asks = orderbook['asks']
                current_price = (bids[0][0] + asks[0][0]) / 2
                
                # 2. 호가창 분석 (The Eyes)
                market_status = self.scanner.analyze_book(bids, asks)
                obi = market_status['obi']
                has_resistance = market_status['resistance'] # 매도벽 (저항선)
                has_support = market_status['support']  # 매수벽 (지지선)

                # 3. 전략 신호 업데이트 (The Brain)
                # 매번 LSMC를 돌리는 게 아니라, 신호가 없을 때만 새로 받아옴
                if self.lsmc_signal is None:
                    new_signal = self.fetch_lsmc_signal()
                    if new_signal != "WAIT":
                        self.lsmc_signal = new_signal
                        self.signal_time = datetime.now()
                        print(f"\n💡 [LSMC Signal] {new_signal} Detected! Waiting for Execution Opportunity...")

                # 4. 신호 유효기간(TTL) 체크
                if self.lsmc_signal:
                    elapsed = (datetime.now() - self.signal_time).total_seconds()
                    if elapsed > self.signal_ttl:
                        print(f"⌛ Signal Expired. (Too much delay). Resetting...")
                        self.lsmc_signal = None
                        continue

                # 5. [최종 판단] 통합 의사결정 (Integration)
                
                # --- [상황 A: 롱 진입 시도] ---
                if self.lsmc_signal == "LONG" and not self.in_position:
                    # 조건: 매도벽이 없고(False), 매수세가 받쳐줄 때(OBI > -0.1)
                    if not has_resistance and obi > -0.1:
                        print(f"⚡ Execution Condition Met! (OBI: {obi:.2f}, No Resistance)")
                        print(f"🚀 BUY LONG MARKET @ {current_price}")
                        # self.exchange.create_market_buy_order('BTC/USDT', qty, None, None, {'positionSide': 'LONG'})
                        self.in_position = True
                        self.position_direction = "LONG"
                        self.lsmc_signal = None # 신호 소모 완료
                    else:
                        # 아직 진입 안 함 (대기)
                        print(f"✋ Pending LONG... (Resistance: {has_resistance}, OBI: {obi:.2f})", end='\r')

                # --- [상황 A-2: 숏 진입 시도] ---
                elif self.lsmc_signal == "SHORT" and not self.in_position:
                    # 조건: 매수벽이 없고(False), 매도세가 우위일 때(OBI < 0.1)
                    if not has_support and obi < 0.1:
                        print(f"⚡ Execution Condition Met! (OBI: {obi:.2f}, No Support)")
                        print(f"📉 SELL SHORT MARKET @ {current_price}")
                        # self.exchange.create_market_sell_order('BTC/USDT', qty, None, None, {'positionSide': 'SHORT'})
                        self.in_position = True
                        self.position_direction = "SHORT"
                        self.lsmc_signal = None # 신호 소모 완료
                    else:
                        # 아직 진입 안 함 (대기)
                        print(f"✋ Pending SHORT... (Support: {has_support}, OBI: {obi:.2f})", end='\r')

                # --- [상황 B: 청산 시도] ---
                elif self.lsmc_signal == "EXIT" and self.in_position:
                    if self.position_direction == "LONG":
                        # 롱 청산: 매수세가 약할 때
                        if obi < 0.3: 
                            print(f"📉 Closing LONG Position @ {current_price}")
                            # self.exchange.create_market_sell_order('BTC/USDT', qty, None, None, {'positionSide': 'LONG'})
                            self.in_position = False
                            self.position_direction = None
                            self.lsmc_signal = None
                        else:
                            print(f"✋ Trying to Exit LONG... but Buyers are strong. (OBI: {obi:.2f})", end='\r')
                    elif self.position_direction == "SHORT":
                        # 숏 청산: 매도세가 약할 때 (매수세가 강할 때)
                        if obi > -0.3:
                            print(f"🚀 Closing SHORT Position @ {current_price}")
                            # self.exchange.create_market_buy_order('BTC/USDT', qty, None, None, {'positionSide': 'SHORT'})
                            self.in_position = False
                            self.position_direction = None
                            self.lsmc_signal = None
                        else:
                            print(f"✋ Trying to Exit SHORT... but Sellers are strong. (OBI: {obi:.2f})", end='\r')

                time.sleep(1) # 1초 루프

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    bot = AlphaBotMain("API_KEY", "SECRET")
    bot.run()