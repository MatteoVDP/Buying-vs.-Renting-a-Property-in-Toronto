"""
Test that month_sin and month_cos are properly calculated during forecast.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from market_simulator import MarketSimulator

print("="*80)
print("TESTING month_sin AND month_cos CALCULATION")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()
print(f"   Loaded {len(df)} rows")

# Check last few months of historical data
print("\n2. Historical data (last 12 months):")
hist_sample = df[['month_sin', 'month_cos']].tail(12)
print(hist_sample)

# Initialize simulator
print("\n3. Initializing simulator...")
sim = MarketSimulator(df, seed=42)
sim.fit()

# Generate a short forecast
print("\n4. Generating 12-month forecast...")
steps = 12
future_index = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
sim_exog = sim.simulate_exogenous(steps=steps)

# Build forecast manually to test _update_lags_and_deltas
base_hist = sim.df.copy()
current_hist = base_hist.copy()

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
    
    # Calculate lags/deltas/RAs and month_sin/month_cos
    start_idx = max(0, len(current_hist) - 50)
    tail = current_hist.iloc[start_idx:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    # Write back current row's derived features
    current_row_features = tail.iloc[-1]
    for col in tail.columns:
        if any(tag in col for tag in ['_lag_', '_delta_', '_RA_', 'month_']):
            if col not in current_hist.columns:
                current_hist[col] = np.nan
            current_hist.at[current_date, col] = current_row_features[col]

# Extract forecast period
forecast_hist = current_hist.loc[future_index]

print("\n5. Forecast month_sin and month_cos:")
forecast_seasonal = forecast_hist[['month_sin', 'month_cos']].copy()
print(forecast_seasonal)

# Verify the values are correct
print("\n6. Verifying forecast values:")
correct = True
for idx, (date, row) in enumerate(forecast_seasonal.iterrows()):
    month = date.month
    expected_sin = np.sin(2 * np.pi * month / 12)
    expected_cos = np.cos(2 * np.pi * month / 12)
    
    sin_match = abs(row['month_sin'] - expected_sin) < 1e-10
    cos_match = abs(row['month_cos'] - expected_cos) < 1e-10
    
    if not (sin_match and cos_match):
        print(f"   ❌ Month {month}: sin={row['month_sin']:.6f} (expected {expected_sin:.6f}), cos={row['month_cos']:.6f} (expected {expected_cos:.6f})")
        correct = False

if correct:
    print("   ✅ All forecast month_sin and month_cos values are correct!")

# Check the repeating pattern
print("\n7. Checking 12-month repeating pattern:")
print("\n   Forecast months and their seasonal values:")
for idx, (date, row) in enumerate(forecast_seasonal.iterrows()):
    month = date.month
    print(f"   {date.strftime('%Y-%m-%d')} (Month {month:2d}): sin={row['month_sin']:7.4f}, cos={row['month_cos']:7.4f}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
