"""
Test that lags/deltas/RAs are calculated correctly after the fix.
This simulates the forecast loop to verify derived features update properly.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from market_simulator import MarketSimulator

print("="*80)
print("TESTING LAG/DELTA/RA FIX")
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
print("\n3. Fitting models...")
sim.fit()
print("   Models fitted")

# Simulate SHORT forecast
print("\n4. Running 12-month forecast to test lag updates...")
steps = 12
future_index = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

# Generate exogenous variables
sim_exog = sim.simulate_exogenous(steps=steps)

# Initialize
base_hist = sim.df.copy()
current_hist = base_hist.copy()
current_log_price = np.log(sim.start_market_price)

# Storage for verification
lag_values_by_month = {}
base_var = 'Income_Growth_YoY'

print(f"\n5. Processing {steps} months and tracking {base_var}...")
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
    
    # Calculate lags (with the fix - only write current row)
    start_idx = max(0, len(current_hist) - 50)
    tail = current_hist.iloc[start_idx:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    # Write ONLY current row's derived features
    current_row_features = tail.iloc[-1]
    for col in tail.columns:
        if any(tag in col for tag in ['_lag_', '_delta_', '_RA_']):
            # Ensure column exists in current_hist
            if col not in current_hist.columns:
                current_hist[col] = np.nan
            current_hist.at[current_date, col] = current_row_features[col]
    
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
    
    # Store values for verification
    month_data = {
        'base': current_hist.loc[current_date, base_var],
        'lag_1': current_hist.loc[current_date, f'{base_var}_lag_1'] if f'{base_var}_lag_1' in current_hist.columns else None,
        'lag_6': current_hist.loc[current_date, f'{base_var}_lag_6'] if f'{base_var}_lag_6' in current_hist.columns else None,
    }
    lag_values_by_month[t] = month_data
    
    if t % 3 == 0:
        print(f"   Month {t+1}: {base_var}={month_data['base']:.6f}, lag_1={month_data['lag_1'] if month_data['lag_1'] else 'N/A'}, lag_6={month_data['lag_6'] if month_data['lag_6'] else 'N/A'}")

print("\n6. Verifying lag values update correctly...")
print(f"\n   Checking if lag_1 matches previous month's base value:")
mismatches = 0
matches = 0
for t in range(1, steps):
    if lag_values_by_month[t]['lag_1'] is not None:
        current_lag1 = lag_values_by_month[t]['lag_1']
        prev_base = lag_values_by_month[t-1]['base']
        
        if abs(current_lag1 - prev_base) < 0.0001:
            matches += 1
            status = "✓"
        else:
            mismatches += 1
            status = "✗"
            print(f"   Month {t}: lag_1={current_lag1:.6f}, prev_base={prev_base:.6f} {status}")

print(f"\n   Results: {matches} matches, {mismatches} mismatches")

if mismatches == 0 and matches > 0:
    print("\n   ✅ SUCCESS! Lags are updating correctly month-to-month")
elif matches == 0:
    print("\n   ❌ FAILURE: No lag_1 column found or all values are None")
else:
    print(f"\n   ⚠️  PARTIAL: Some lags are updating but {mismatches} mismatches found")

# Check if lag values are all the same (the original bug)
print(f"\n   Checking if all lag_6 values are the same (original bug):")
lag6_values = [lag_values_by_month[t]['lag_6'] for t in range(steps) if lag_values_by_month[t]['lag_6'] is not None]
if lag6_values:
    unique_lag6 = len(set(lag6_values))
    print(f"   Unique lag_6 values: {unique_lag6}/{len(lag6_values)}")
    if unique_lag6 == 1:
        print(f"   ❌ BUG STILL PRESENT: All lag_6 values are {lag6_values[0]:.6f}")
    elif unique_lag6 > 1:
        print(f"   ✅ FIXED: lag_6 values vary (min={min(lag6_values):.6f}, max={max(lag6_values):.6f})")
else:
    print(f"   ⚠️  No lag_6 values found to check")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
