#!/bin/bash
# GPU 학습 스크립트 (Bash Version)
# 180일 데이터, 300 에피소드 학습

echo "======================================================================"
echo "🚀 Transformer AI Training - GPU Accelerated"
echo "======================================================================"
echo "⏰ Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📊 Configuration:"
echo "   - Episodes: 300"
echo "   - Data Period: 180 days"
echo "   - Symbols: BTC, ETH, SOL, XRP, BNB"
echo "   - Model: Transformer + C51 DRL"
echo "======================================================================"
echo ""

# GPU 체크
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "⚠️  nvidia-smi not found. Make sure CUDA is properly installed."
    echo ""
fi

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🔥 Starting Training..."
echo "======================================================================"
echo ""

# 학습 시작 시간 기록
START=$(date +%s)

# Python 스크립트 실행
python3 strategies/Final_Transformer_Edition.py

# 학습 종료 시간 및 소요 시간 계산
END=$(date +%s)
DURATION=$((END - START))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "======================================================================"
echo "✅ Training Completed!"
echo "======================================================================"
printf "⏱️  Total Duration: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "🧠 Model saved to: best_transformer_brain.pth"
echo "⏰ End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"


