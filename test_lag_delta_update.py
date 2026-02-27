"""
Test script to verify that lags, deltas, and rolling averages
are being properly calculated and persisted during forecasting.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from market_simulator import MarketSimulator

print("="*80)
print("TESTING LAG/DELTA/RA CALCULATION IN FORECAST LOOP")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()
print(f"   Loaded {len(df)} rows")

# Initialize simulator
print("\n2. Initializing simulator...")
sim = MarketSimulator(df, seed=42)

# Fit models
print("\n3. Fitting models (this may take a minute)...")
sim.fit()
print("   Models fitted")

# Run a SHORT forecast (just 30 months) to test
print("\n4. Running SHORT forecast (30 months) to test feature generation...")
steps = 30
future_index = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

# Generate exogenous variables
sim_exog = sim.simulate_exogenous(steps=steps)

# Initialize vars
base_hist = sim.df.copy()
current_hist = base_hist.copy()
current_log_price = np.log(sim.start_market_price)

print(f"\n5. Processing {steps} forecast months...")
for t in range(steps):
    current_date = future_index[t]
    sim_row = sim_exog.iloc[[t]]
   
    # Ensure all columns exist
    for col in current_hist.columns:
        if col not in sim_row.columns:
            sim_row[col] = np.nan
    
    # Append
    current_hist = pd.concat([current_hist, sim_row], axis=0)
    current_hist = current_hist.ffill()
    
    # THIS IS THE KEY PART: Update lags/deltas/RAs
    start_idx = max(0, len(current_hist) - 50)
    tail = current_hist.iloc[start_idx:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    # Write back to current_hist
    for col in tail.columns:
        current_hist.loc[tail.index, col] = tail[col]
    
    # Make prediction
    try:
        X_row = current_hist.iloc[[-1]][sim.feature_columns]
    except KeyError:
        X_row = current_hist.iloc[[-1]].copy()
        for col in sim.feature_columns:
            if col not in X_row.columns:
                X_row[col] = 0
    
    X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
    pred_log_return = float(sim.xgb_model.predict(X_row)[0])
    current_log_price += pred_log_return
    current_hist.at[current_date, sim.price_col] = pred_log_return
   
    if t % 10 == 0:
        print(f"   Month {t+1}/{steps}")
        
print("\n6. Analyzing results...")

# Extract forecast portion
forecast_hist = current_hist.loc[future_index]

# Check which columns are present and populated
lag_cols = [c for c in forecast_hist.columns if '_lag_' in c]
delta_cols = [c for c in forecast_hist.columns if '_delta_' in c]
ra_cols = [c for c in forecast_hist.columns if '_RA_' in c]

print(f"\n   Columns found:")
print(f"     Lags: {len(lag_cols)}")
print(f"     Deltas: {len(delta_cols)}")
print(f"     Rolling Averages: {len(ra_cols)}")

# Check month 24 (0-indexed = 23)
if len(forecast_hist) >= 24:
    row_24 = forecast_hist.iloc[23]
    
    populated_lags = sum(1 for c in lag_cols if pd.notna(row_24[c]))
    populated_deltas = sum(1 for c in delta_cols if pd.notna(row_24[c]))
    populated_ras = sum(1 for c in ra_cols if pd.notna(row_24[c]))
    
    print(f"\n   At month 24 ({row_24.name.strftime('%Y-%m-%d')}):")
    print(f"     Lags populated: {populated_lags}/{len(lag_cols)} ({100*populated_lags/len(lag_cols) if lag_cols else 0:.1f}%)")
    print(f"     Deltas populated: {populated_deltas}/{len(delta_cols)} ({100*populated_deltas/len(delta_cols) if delta_cols else 0:.1f}%)")
    print(f"     RAs populated: {populated_ras}/{len(ra_cols)} ({100*populated_ras/len(ra_cols) if ra_cols else 0:.1f}%)")
    
    if populated_lags > 0:
        print(f"\n   ✓ SUCCESS! Lags are being calculated and persisted")
        print(f"\n   Sample lag values at month 24:")
        for col in [c for c in lag_cols[:5] if pd.notna(row_24[c])]:
            print(f"     {col}: {row_24[col]:.6f}")
    else:
        print(f"\n   ❌ FAILURE: Lags are NOT being populated")
        print(f"   DEBUG: Checking if columns even exist in tail after _update_lags_and_deltas...")
        
        # Debug: manually call the function and check
        test_tail = current_hist.iloc[-24:].copy()
        test_tail_updated = sim._update_lags_and_deltas(test_tail)
        test_lag_cols = [c for c in test_tail_updated.columns if '_lag_' in c]
        print(f"     Lag columns created by _update_lags_and_deltas: {len(test_lag_cols)}")
        if test_lag_cols:
            print(f"     Example: {test_lag_cols[0]} = {test_tail_updated[test_lag_cols[0]].iloc[-1]}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
