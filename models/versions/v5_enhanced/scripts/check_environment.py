#!/usr/bin/env python3
"""
Quick environment check script for V5 training
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not found")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found")
        return False
    
    return True

def check_data():
    """Check if data files are available"""
    print("\n📊 Checking data availability...")
    
    data_path = "/Users/bogalaxminarayana/myGit/ai_stock_predictions/data/raw/daily/RELIANCE_NSE_daily_20150911_to_20250910.csv"
    
    if os.path.exists(data_path):
        print(f"✅ Daily data found: {data_path}")
        
        # Check data size
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"   Data shape: {df.shape}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return True
        except Exception as e:
            print(f"❌ Error reading data: {e}")
            return False
    else:
        print(f"❌ Data file not found: {data_path}")
        return False

def check_model_import():
    """Check if V5 model can be imported"""
    print("\n🤖 Checking V5 model import...")
    
    try:
        from models.versions.v5_enhanced.model_v5 import EnhancedLSTMV5, create_enhanced_model_v5
        print("✅ V5 model import successful")
        
        # Test model creation
        config = {
            'input_size': 30,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 4,
            'dropout': 0.1,
            'attention_heads': 4,
            'prediction_horizon': 1
        }
        
        model = create_enhanced_model_v5(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model created successfully with {total_params:,} parameters")
        return True
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def main():
    """Main check function"""
    print("🚀 V5 Enhanced Model - Environment Check")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    data_ok = check_data()
    model_ok = check_model_import()
    
    print("\n" + "=" * 50)
    if deps_ok and data_ok and model_ok:
        print("🎉 All checks passed! Ready to train V5 model.")
        print("\n🚀 To start training, run:")
        print("   python train_v5.py")
    else:
        print("❌ Some checks failed. Please fix the issues before training.")
        
        if not deps_ok:
            print("   - Install missing dependencies")
        if not data_ok:
            print("   - Check data file availability")
        if not model_ok:
            print("   - Check model import issues")
    
    print("=" * 50)

if __name__ == "__main__":
    main()