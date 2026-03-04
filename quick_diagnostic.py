"""
Quick diagnostic to check if lags/deltas are properly varying in the forecast file.
"""
import pandas as pd
import numpy as np

print("="*80)
print("QUICK DIAGNOSTIC: Checking for the 'all values the same' bug")
print("="*80)

try:
    df = pd.read_csv('results/forecast_25_year_march2050.csv', index_col=0, parse_dates=[0])
    forecast = df[df.index >= '2025-04-01'].copy()
    
    # Get lag columns
    lag_cols = [c for c in forecast.columns if '_lag_' in c]
    
    if not lag_cols:
        print("\n❌ No lag columns found in forecast file")
    else:
        print(f"\nFound {len(lag_cols)} lag columns")
        print(f"\nChecking first 3 lag columns for value variation:")
        
        for col in lag_cols[:3]:
            values = forecast[col].dropna()
            if len(values) == 0:
                print(f"\n  {col}: ALL NaN")
            else:
                unique_vals = len(values.unique())
                if unique_vals == 1:
                    print(f"\n  {col}: ❌ BUG - all {len(values)} values are {values.iloc[0]:.6f}")
                else:
                    print(f"\n  {col}: ✓ OK - {unique_vals} unique values")
                    print(f"     Range: {values.min():.6f} to {values.max():.6f}")
                    print(f"     First 5: {list(values.head().values)}")
        
except FileNotFoundError:
    print("\n❌ Forecast file not found. Run: python scripts/audit_updated.py")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*80)
