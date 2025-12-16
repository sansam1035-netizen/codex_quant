#!/usr/bin/env python3
"""
GPU 학습 스크립트 for Final Transformer Edition
- 180일 데이터
- 300 에피소드
- CUDA GPU 가속
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("="*70)
    print("🚀 Transformer AI Training - GPU Accelerated")
    print("="*70)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration:")
    print(f"   - Episodes: 300")
    print(f"   - Data Period: 180 days")
    print(f"   - Symbols: BTC, ETH, SOL, XRP, BNB")
    print(f"   - Model: Transformer + C51 DRL")
    print("="*70)
    
    # GPU 체크
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\n⚠️  No CUDA GPU detected. Training will use CPU (slower)")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
    except ImportError:
        print("\n⚠️  PyTorch not installed. Please install requirements first.")
        return
    
    print("\n" + "="*70)
    print("🔥 Starting Training...")
    print("="*70 + "\n")
    
    # 스크립트 경로
    script_path = os.path.join(os.path.dirname(__file__), "strategies", "Final_Transformer_Edition.py")
    
    # 학습 시작
    start_time = datetime.now()
    
    try:
        # Python 스크립트 실행
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__)
        )
        
        # 학습 완료
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        if result.returncode == 0:
            print("✅ Training Completed Successfully!")
        else:
            print("⚠️  Training ended with warnings/errors")
        print("="*70)
        print(f"⏱️  Total Duration: {duration}")
        print(f"🧠 Model saved to: best_transformer_brain.pth")
        print(f"⏰ End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("💾 Current progress should be saved in best_transformer_brain.pth")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return

if __name__ == "__main__":
    main()


