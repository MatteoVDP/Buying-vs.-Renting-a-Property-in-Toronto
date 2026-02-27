#!/usr/bin/env python3
"""Simplest possible test to verify the refactored MarketSimulator works."""
import sys
sys.path.insert(0, '/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/scripts')

import pandas as pd
import numpy as np
from market_simulator import MarketSimulator

# Load data
print("Loading processed_data.csv...")
df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")

# Initialize
print("Initializing MarketSimulator...")
sim = MarketSimulator(df)
print("✓ Initialized\n")

# Fit
print("Fitting models...")
sim.fit(df)
print("✓ Models fitted\n")

# Quick forecast
print("Running 5 iterations x 12 months forecast...")
try:
    prices = sim.forecast_price(iterations=5, steps=12)
    print(f"✓ Forecast complete: {prices.shape}")
    print(f"  Sample prices (first 5 rows, first 2 iterations):\n{prices.iloc[:5, :2]}\n")
except Exception as e:
    print(f"✗ Error: {e}")
    raise

print("✓✓✓ All tests passed!")
