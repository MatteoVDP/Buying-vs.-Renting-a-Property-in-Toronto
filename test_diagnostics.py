#!/usr/bin/env python3
"""
Quick diagnostic test to see what's failing in the market simulator
"""
import sys
sys.path.insert(0, '/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')

import pandas as pd
from market_simulator import MarketSimulator

print("="*80)
print("MARKET SIMULATOR DIAGNOSTIC TEST")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()

print(f"   Data shape: {df.shape}")
print(f"   Columns available: {len(df.columns)}")
print(f"   Date range: {df.index.min()} to {df.index.max()}")

# Initialize
print("\n2. Initializing simulator...")
sim = MarketSimulator(df, seed=42)

print(f"   Tier 1 vars defined: {len(sim.tier1_vars)}")
print(f"   Tier 2 vars defined: {len(sim.tier2_vars)}")
print(f"   Tier 3 vars defined: {len(sim.tier3_vars)}")

# Check which variables are actually in the data
print("\n3. Checking variable availability in data...")
tier1_available = [v for v in sim.tier1_vars if v in df.columns]
tier1_missing = [v for v in sim.tier1_vars if v not in df.columns]
print(f"   Tier 1: {len(tier1_available)}/{len(sim.tier1_vars)} available")
if tier1_missing:
    print(f"   Missing: {tier1_missing}")

tier2_available = [v for v in sim.tier2_vars if v in df.columns]
tier2_missing = [v for v in sim.tier2_vars if v not in df.columns]
print(f"   Tier 2: {len(tier2_available)}/{len(sim.tier2_vars)} available")
if tier2_missing:
    print(f"   Missing: {tier2_missing}")

tier3_available = [v for v in sim.tier3_vars if v in df.columns]
tier3_missing = [v for v in sim.tier3_vars if v not in df.columns]
print(f"   Tier 3: {len(tier3_available)}/{len(sim.tier3_vars)} available")
if tier3_missing:
    print(f"   Missing: {tier3_missing}")

# Fit models with diagnostic output
print("\n4. Fitting models (with diagnostics)...")
print("="*80)
sim.fit(df)
print("="*80)

# Check how many models were actually fitted
tier1_fitted = sum(1 for v in sim.tier1_vars if sim.arima_models.get(v) is not None)
tier2_fitted = sum(1 for v in sim.tier2_vars if sim.sarimax_models.get(v) is not None)
tier3_fitted = sum(1 for v in sim.tier3_vars if sim.sarimax_models.get(v) is not None)

print(f"\n5. Model fitting summary:")
print(f"   Tier 1 (ARIMA):  {tier1_fitted}/{len(sim.tier1_vars)} fitted")
print(f"   Tier 2 (SARIMAX): {tier2_fitted}/{len(sim.tier2_vars)} fitted")
print(f"   Tier 3 (SARIMAX): {tier3_fitted}/{len(sim.tier3_vars)} fitted")
print(f"   XGBoost: {'✓ Fitted' if sim.xgb_model else '✗ Not fitted'}")
print(f"   XGBoost features: {len(sim.feature_columns) if sim.feature_columns else 0}")

# Try a short forecast to see what fails
print("\n6. Testing short exogenous simulation (12 months)...")
try:
    sim_world = sim.simulate_exogenous(steps=12)
    print(f"   ✓ Simulation successful: {sim_world.shape}")
    print(f"   Columns generated: {len(sim_world.columns)}")
except Exception as e:
    print(f"   ✗ Simulation failed: {e}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
