#!/usr/bin/env python3
"""
Diagnostic to see what's causing the step function pattern
"""
import sys
sys.path.insert(0, '/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')

import pandas as pd
import numpy as np
from market_simulator import MarketSimulator

print("="*80)
print("STEP FUNCTION DIAGNOSTIC")
print("="*80)

# Load data
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()

# Initialize and fit
print("\n1. Fitting models...")
sim = MarketSimulator(df, seed=42)
sim.fit(df)

# Generate ONE exogenous forecast
print("\n2. Generating exogenous variables (36 months)...")
sim_world = sim.simulate_exogenous(steps=36)

# Check variability in Tier 1
print("\n3. Checking Tier 1 variability:")
for var in sim.tier1_vars[:3]:  # Check first 3
    if var in sim_world.columns:
        values = sim_world[var].values
        print(f"   {var}:")
        print(f"      Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        print(f"      Range: [{values.min():.6f}, {values.max():.6f}]")
        print(f"      First 5: {values[:5]}")

# Check variability in Tier 2
print("\n4. Checking Tier 2 variability:")
for var in sim.tier2_vars[:3]:  # Check first 3
    if var in sim_world.columns:
        values = sim_world[var].values
        print(f"   {var}:")
        print(f"      Mean: {values.mean():.6f}, Std: {values.std():.6f}")
        print(f"      Range: [{values.min():.6f}, {values.max():.6f}]")
        print(f"      First 5: {values[:5]}")

# Now predict prices manually
print("\n5. Predicting log returns (first 36 months)...")
base_hist = sim.df.copy()
current_hist = base_hist.copy()

log_returns = []
for t in range(36):
    sim_row = sim_world.iloc[[t]]
    current_hist = pd.concat([current_hist, sim_row])
    
    tail = current_hist.iloc[-24:].copy()
    tail = sim._update_lags_and_deltas(tail)
    
    X_row = tail.iloc[[-1]][sim.feature_columns]
    X_row = X_row.fillna(0).replace([np.inf, -np.inf], 0)
    
    pred_log_return = float(sim.xgb_model.predict(X_row)[0])
    log_returns.append(pred_log_return)
    
    # Write back for next month
    current_hist.at[sim_world.index[t], sim.price_col] = pred_log_return

log_returns = np.array(log_returns)

print(f"\n6. Log return statistics:")
print(f"   Mean: {log_returns.mean():.8f}")
print(f"   Std:  {log_returns.std():.8f}")
print(f"   Range: [{log_returns.min():.8f}, {log_returns.max():.8f}]")
print(f"   Unique values: {len(np.unique(log_returns))}")

print(f"\n7. First 12 log returns:")
for i, ret in enumerate(log_returns[:12]):
    price_change = (np.exp(ret) - 1) * 100
    print(f"   Month {i+1:2d}: {ret:+.8f} ({price_change:+.4f}%)")

# Check for repeating patterns
print(f"\n8. Checking for repetition:")
if len(log_returns) >= 24:
    first_half = log_returns[:12]
    second_half = log_returns[12:24]
    correlation = np.corrcoef(first_half, second_half)[0, 1]
    print(f"   Correlation between first 12 and second 12 months: {correlation:.4f}")
    if correlation > 0.9:
        print(f"   ⚠️  HIGH CORRELATION - Pattern is repeating!")

# Check if values are identical
consecutive_same = sum(1 for i in range(len(log_returns)-1) if abs(log_returns[i] - log_returns[i+1]) < 1e-10)
print(f"   Consecutive identical predictions: {consecutive_same}/{len(log_returns)-1}")

if consecutive_same > len(log_returns) / 2:
    print(f"   ⚠️  STEP FUNCTION DETECTED - Model predicting same value repeatedly!")

print("\n" + "="*80)
