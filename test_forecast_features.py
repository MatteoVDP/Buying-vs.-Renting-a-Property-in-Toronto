"""Quick test to verify lags/deltas/RAs are being saved in forecast"""
import pandas as pd
import numpy as np

# Load the most recent forecast
df = pd.read_csv('results/forecast_25_year_march2050.csv', index_col=0, parse_dates=[0])

# Filter to forecast period
forecast_df = df[df.index >= '2025-04-01'].copy()

print("="*80)
print("FORECAST FEATURE VERIFICATION")
print("="*80)

# Check derived feature columns
lag_cols = [c for c in forecast_df.columns if '_lag_' in c]
delta_cols = [c for c in forecast_df.columns if '_delta_' in c]
ra_cols = [c for c in forecast_df.columns if '_RA_' in c]

print(f"\nDerived feature columns found:")
print(f"  Lags: {len(lag_cols)}")
print(f"  Deltas: {len(delta_cols)}")
print(f"  Rolling Averages: {len(ra_cols)}")

# Check month 24 (should have most features populated)
if len(forecast_df) >= 24:
    row_24 = forecast_df.iloc[23]
    date_24 = row_24.name
    
    populated_lags = sum(1 for c in lag_cols if pd.notna(row_24[c]))
    populated_deltas = sum(1 for c in delta_cols if pd.notna(row_24[c]))
    populated_ras = sum(1 for c in ra_cols if pd.notna(row_24[c]))
    
    print(f"\nAt month 24 ({date_24.strftime('%Y-%m-%d')}):")
    print(f"  Lags populated: {populated_lags}/{len(lag_cols)} ({100*populated_lags/len(lag_cols) if lag_cols else 0:.1f}%)")
    print(f"  Deltas populated: {populated_deltas}/{len(delta_cols)} ({100*populated_deltas/len(delta_cols) if delta_cols else 0:.1f}%)")
    print(f"  RAs populated: {populated_ras}/{len(ra_cols)} ({100*populated_ras/len(ra_cols) if ra_cols else 0:.1f}%)")
    
    if populated_lags == 0 and len(lag_cols) > 0:
        print("\n⚠️  WARNING: No lags are populated!")
        print("    This means features are NOT being extracted from current_hist")
    elif populated_lags > 0:
        print(f"\n✓ SUCCESS: Lags are being calculated and saved!")
        # Show some example values
        sample_lags = [c for c in lag_cols[:5] if pd.notna(row_24[c])]
        if sample_lags:
            print(f"\n  Example lag values at month 24:")
            for col in sample_lags:
                print(f"    {col}: {row_24[col]:.6f}")

# Check affordability
if 'Affordability_Ratio' in forecast_df.columns:
    afford_populated = forecast_df['Affordability_Ratio'].notna().sum()
    print(f"\nAffordability_Ratio:")
    print(f"  Populated: {afford_populated}/{len(forecast_df)} ({100*afford_populated/len(forecast_df):.1f}%)")
    if afford_populated > 0:
        print(f"  First: {forecast_df['Affordability_Ratio'].dropna().iloc[0]:.6f}")
        print(f"  Last: {forecast_df['Affordability_Ratio'].dropna().iloc[-1]:.6f}")

print("\n" + "="*80)
