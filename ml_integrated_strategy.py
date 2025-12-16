"""
ML Integrated Strategy
=======================
실제 ML 모델 예측을 활용한 트레이딩 전략

Features:
- LightGBM 기반 가격 예측
- 기술적 지표 + ML 예측 결합
- 신뢰도 기반 포지션 크기 조정
- 온라인 학습 지원
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install: pip install lightgbm")

from sklearn.preprocessing import StandardScaler


class MLIntegratedStrategy:
    """
    ML 통합 전략
    
    1. 기술적 지표 계산
    2. ML 모델로 가격 방향 예측
    3. 예측 신뢰도 계산
    4. 기술적 지표와 ML 예측 결합
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}  # {symbol: model}
        self.scalers = {}  # {symbol: scaler}
        self.trained_symbols = set()
        
        # ML 설정
        self.use_ml = self.config.get("use_ml", True) and LGBM_AVAILABLE
        self.ml_weight = self.config.get("ml_weight", 0.6)  # ML 예측 가중치
        self.technical_weight = self.config.get("technical_weight", 0.4)  # 기술적 지표 가중치
        
        # 학습 설정
        self.min_train_samples = self.config.get("min_train_samples", 500)
        self.retrain_interval = self.config.get("retrain_interval", 100)  # 100바마다 재학습
        self.last_train_time = {}
        
        print(f"✅ ML Integrated Strategy 초기화")
        print(f"  ML 사용: {self.use_ml}")
        print(f"  ML 가중치: {self.ml_weight:.1%}")
        print(f"  기술적 지표 가중치: {self.technical_weight:.1%}")
    
    def generate_signals(
        self,
        data_dict: Dict[str, pd.DataFrame],
        current_time: datetime,
        positions: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        신호 생성
        
        Returns:
            {symbol: {'signal': -1/0/1, 'confidence': 0-1, 'ml_pred': float, 'technical_score': float}}
        """
        signals = {}
        positions = positions or {}
        
        for symbol, df in data_dict.items():
            if len(df) < 100:
                continue
            
            # 현재까지의 데이터만 사용
            current_df = df[df.index <= current_time].copy()
            
            if len(current_df) < 100:
                continue
            
            # 1. 기술적 지표 계산
            technical_signal, technical_confidence = self._calculate_technical_signal(current_df)
            
            # 2. ML 예측
            ml_signal = 0
            ml_confidence = 0.5
            ml_pred = 0.0
            
            if self.use_ml:
                ml_signal, ml_confidence, ml_pred = self._get_ml_prediction(symbol, current_df)
            
            # 3. 신호 결합
            combined_signal = (
                technical_signal * self.technical_weight +
                ml_signal * self.ml_weight
            )
            
            # 4. 최종 신호 및 신뢰도
            if combined_signal > 0.3:
                final_signal = 1  # 매수
            elif combined_signal < -0.3:
                final_signal = -1  # 매도
            else:
                final_signal = 0  # 관망
            
            # 신뢰도 계산 (기술적 지표와 ML 예측이 일치하면 높음)
            if technical_signal * ml_signal > 0:
                # 같은 방향이면 신뢰도 높음
                final_confidence = (technical_confidence * self.technical_weight + 
                                   ml_confidence * self.ml_weight)
                final_confidence = min(final_confidence * 1.2, 1.0)  # 보너스
            else:
                # 다른 방향이면 신뢰도 낮음
                final_confidence = (technical_confidence * self.technical_weight + 
                                   ml_confidence * self.ml_weight) * 0.8
            
            signals[symbol] = {
                'signal': final_signal,
                'confidence': final_confidence,
                'ml_pred': ml_pred,
                'ml_signal': ml_signal,
                'ml_confidence': ml_confidence,
                'technical_signal': technical_signal,
                'technical_confidence': technical_confidence,
                'combined_score': combined_signal
            }
        
        return signals
    
    def _calculate_technical_signal(
        self,
        df: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        기술적 지표 기반 신호 계산
        
        Returns:
            signal (-1 to 1), confidence (0 to 1)
        """
        # 기술적 지표 계산
        close = df['close'].values
        
        # 1. 이동평균
        sma_20 = pd.Series(close).rolling(20).mean().values
        sma_50 = pd.Series(close).rolling(50).mean().values
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        
        # 2. MACD
        macd = ema_12 - ema_26
        signal_line = pd.Series(macd).ewm(span=9).mean().values
        
        # 3. RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.values
        
        # 4. Bollinger Bands
        bb_middle = pd.Series(close).rolling(20).mean().values
        bb_std = pd.Series(close).rolling(20).std().values
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        # 5. ATR (변동성)
        high = df['high'].values
        low = df['low'].values
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(14).mean().values
        
        # 현재 값
        current_price = close[-1]
        current_sma20 = sma_20[-1]
        current_sma50 = sma_50[-1]
        current_macd = macd[-1]
        current_signal = signal_line[-1]
        current_rsi = rsi[-1]
        current_bb_upper = bb_upper[-1]
        current_bb_lower = bb_lower[-1]
        current_atr = atr[-1]
        
        # 신호 점수 계산
        score = 0
        confidence_factors = []
        
        # 1. 이동평균 (30%)
        if current_price > current_sma20 > current_sma50:
            score += 0.3
            confidence_factors.append(0.8)
        elif current_price < current_sma20 < current_sma50:
            score -= 0.3
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # 2. MACD (25%)
        if current_macd > current_signal and current_macd > 0:
            score += 0.25
            confidence_factors.append(0.7)
        elif current_macd < current_signal and current_macd < 0:
            score -= 0.25
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # 3. RSI (20%)
        if current_rsi < 30:
            score += 0.2  # 과매도
            confidence_factors.append(0.6)
        elif current_rsi > 70:
            score -= 0.2  # 과매수
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.5)
        
        # 4. Bollinger Bands (15%)
        if current_price < current_bb_lower:
            score += 0.15  # 하단 돌파
            confidence_factors.append(0.7)
        elif current_price > current_bb_upper:
            score -= 0.15  # 상단 돌파
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # 5. 추세 강도 (10%)
        if len(sma_20) > 5:
            trend = (sma_20[-1] - sma_20[-5]) / sma_20[-5]
            if trend > 0.02:
                score += 0.1
            elif trend < -0.02:
                score -= 0.1
        
        # 신뢰도 계산 (지표들의 일치도)
        confidence = np.mean(confidence_factors)
        
        # 변동성 조정 (변동성 높으면 신뢰도 낮춤)
        if current_atr > 0:
            volatility = current_atr / current_price
            if volatility > 0.05:  # 5% 이상 변동성
                confidence *= 0.8
        
        return score, confidence
    
    def _get_ml_prediction(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        ML 모델 예측
        
        Returns:
            signal (-1 to 1), confidence (0 to 1), raw_prediction
        """
        if not LGBM_AVAILABLE:
            return 0, 0.5, 0.0
        
        # 모델 학습 또는 로드
        if symbol not in self.trained_symbols:
            self._train_model(symbol, df)
        
        # 재학습 체크
        if symbol in self.last_train_time:
            bars_since_train = len(df) - self.last_train_time[symbol]
            if bars_since_train >= self.retrain_interval:
                self._train_model(symbol, df)
        
        # 모델이 없으면 기본값 반환
        if symbol not in self.models:
            return 0, 0.5, 0.0
        
        # 특징 추출
        features = self._extract_features(df)
        
        if features is None or len(features) == 0:
            return 0, 0.5, 0.0
        
        # 예측
        try:
            model = self.models[symbol]
            scaler = self.scalers.get(symbol)
            
            # 마지막 행만 예측
            X = features.iloc[[-1]].values
            
            if scaler is not None:
                X = scaler.transform(X)
            
            # 예측 (회귀: 다음 가격 변화율)
            pred = model.predict(X)[0]
            
            # 신호 변환
            if pred > 0.01:  # 1% 이상 상승 예측
                signal = 1.0
                confidence = min(abs(pred) * 10, 1.0)  # 예측 크기에 비례
            elif pred < -0.01:  # 1% 이상 하락 예측
                signal = -1.0
                confidence = min(abs(pred) * 10, 1.0)
            else:
                signal = 0.0
                confidence = 0.5
            
            return signal, confidence, pred
        
        except Exception as e:
            print(f"    ⚠️ ML 예측 오류 ({symbol}): {e}")
            return 0, 0.5, 0.0
    
    def _train_model(self, symbol: str, df: pd.DataFrame):
        """ML 모델 학습"""
        if len(df) < self.min_train_samples:
            return
        
        try:
            # 특징 추출
            features = self._extract_features(df)
            
            if features is None or len(features) < 100:
                return
            
            # 타겟 생성 (다음 N바 후 수익률)
            future_returns = df['close'].pct_change(5).shift(-5)  # 5바 후 수익률
            
            # 결측치 제거
            valid_idx = ~(features.isna().any(axis=1) | future_returns.isna())
            X = features[valid_idx].values
            y = future_returns[valid_idx].values
            
            if len(X) < 100:
                return
            
            # 스케일링
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 학습 (최근 80%만 사용)
            train_size = int(len(X_scaled) * 0.8)
            X_train = X_scaled[:train_size]
            y_train = y[:train_size]
            
            # LightGBM 모델
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train, y_train)
            
            # 저장
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            self.trained_symbols.add(symbol)
            self.last_train_time[symbol] = len(df)
            
            print(f"    🤖 ML 모델 학습 완료: {symbol} ({len(X_train)} 샘플)")
        
        except Exception as e:
            print(f"    ⚠️ ML 학습 오류 ({symbol}): {e}")
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """특징 추출"""
        try:
            features = pd.DataFrame(index=df.index)
            
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # 1. 가격 기반 특징
            features['returns_1'] = close.pct_change(1)
            features['returns_5'] = close.pct_change(5)
            features['returns_10'] = close.pct_change(10)
            
            # 2. 이동평균
            features['sma_5'] = close.rolling(5).mean() / close - 1
            features['sma_20'] = close.rolling(20).mean() / close - 1
            features['sma_50'] = close.rolling(50).mean() / close - 1
            
            # 3. 변동성
            features['volatility_10'] = close.pct_change().rolling(10).std()
            features['volatility_20'] = close.pct_change().rolling(20).std()
            
            # 4. RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # 5. MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # 6. Bollinger Bands
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            features['bb_position'] = (close - bb_middle) / (2 * bb_std)
            
            # 7. 거래량
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            
            # 8. High-Low 범위
            features['hl_ratio'] = (high - low) / close
            
            return features
        
        except Exception as e:
            print(f"    ⚠️ 특징 추출 오류: {e}")
            return None


# 전략 함수 (백테스터 호환)
def ml_integrated_strategy_func(
    data_dict: Dict[str, pd.DataFrame],
    current_time: datetime,
    positions: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    ML 통합 전략 함수
    
    백테스터에서 직접 호출 가능
    """
    # 전역 전략 인스턴스 (재사용)
    if not hasattr(ml_integrated_strategy_func, 'strategy'):
        ml_integrated_strategy_func.strategy = MLIntegratedStrategy({
            'use_ml': True,
            'ml_weight': 0.6,
            'technical_weight': 0.4,
            'min_train_samples': 500,
            'retrain_interval': 100
        })
    
    return ml_integrated_strategy_func.strategy.generate_signals(
        data_dict,
        current_time,
        positions
    )

