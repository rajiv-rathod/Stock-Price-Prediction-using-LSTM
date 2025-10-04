"""
Quick test to verify Bollinger Bands fix
"""
import pandas as pd
import numpy as np
from advanced_model import AdvancedStockPredictor

# Create test data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
test_data = pd.DataFrame({
    'Open': np.random.uniform(100, 110, 100),
    'High': np.random.uniform(110, 120, 100),
    'Low': np.random.uniform(90, 100, 100),
    'Close': np.random.uniform(95, 115, 100),
    'Volume': np.random.randint(1000000, 5000000, 100)
}, index=dates)

print("Creating AdvancedStockPredictor...")
predictor = AdvancedStockPredictor(ticker='TEST', lookback=60, use_technical_indicators=True)

print("Testing add_technical_indicators...")
try:
    result = predictor.add_technical_indicators(test_data)
    print("✅ SUCCESS! Technical indicators added without errors")
    print(f"\nDataFrame shape: {result.shape}")
    print(f"\nColumns added: {len(result.columns)} total columns")
    print(f"\nSample columns: {list(result.columns[:10])}")
    
    # Check BB_Position specifically
    if 'BB_Position' in result.columns:
        print(f"\n✅ BB_Position column exists")
        print(f"   Type: {type(result['BB_Position'])}")
        print(f"   Sample values: {result['BB_Position'].tail(5).values}")
    else:
        print("\n❌ BB_Position column not found")
        
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
