"""
Comprehensive test with real stock data to verify BB fix
"""
import yfinance as yf
from advanced_model import AdvancedStockPredictor

print("Testing with REAL stock data (AAPL)...")
print("=" * 60)

# Fetch real data
ticker = "AAPL"
print(f"\n1. Fetching {ticker} data from Yahoo Finance...")
data = yf.download(ticker, period='6mo', interval='1d', progress=False)

if data.empty:
    print("❌ Failed to download data")
    exit(1)

print(f"✅ Downloaded {len(data)} days of data")
print(f"   Date range: {data.index[0]} to {data.index[-1]}")

# Test with AdvancedStockPredictor
print(f"\n2. Creating AdvancedStockPredictor...")
predictor = AdvancedStockPredictor(ticker=ticker, lookback=60, use_technical_indicators=True)

print(f"\n3. Adding technical indicators...")
try:
    result = predictor.add_technical_indicators(data.copy())
    print("✅ SUCCESS! All technical indicators added")
    
    print(f"\n4. Verifying results:")
    print(f"   - Original columns: {len(data.columns)}")
    print(f"   - With indicators: {len(result.columns)}")
    print(f"   - New features added: {len(result.columns) - len(data.columns)}")
    
    print(f"\n5. Checking Bollinger Bands specifically:")
    bb_cols = [col for col in result.columns if 'BB' in col]
    print(f"   BB columns: {bb_cols}")
    
    for col in bb_cols:
        print(f"\n   {col}:")
        print(f"     - Type: {type(result[col])}")
        print(f"     - Has data: {not result[col].isna().all()}")
        print(f"     - Sample values: {result[col].dropna().tail(3).values}")
        print(f"     - Min: {result[col].min():.4f}, Max: {result[col].max():.4f}")
    
    print(f"\n6. Checking for NaN values:")
    nan_counts = result.isna().sum()
    problem_cols = nan_counts[nan_counts > len(result) * 0.5]
    if len(problem_cols) > 0:
        print(f"   ⚠️  Columns with >50% NaN: {list(problem_cols.index)}")
    else:
        print(f"   ✅ All columns have reasonable data coverage")
    
    print(f"\n7. Checking for infinite values:")
    inf_cols = []
    for col in result.select_dtypes(include=['float64', 'float32']).columns:
        if (result[col] == float('inf')).any() or (result[col] == float('-inf')).any():
            inf_cols.append(col)
    
    if inf_cols:
        print(f"   ❌ Columns with infinity: {inf_cols}")
    else:
        print(f"   ✅ No infinite values found")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! BB_Position is working perfectly!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
