#!/usr/bin/env python3
"""
Test the fixed Macro Gravity Override.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from market_simulator import MarketSimulator

print("="*80)
print("TESTING FIXED MACRO GRAVITY OVERRIDE")
print("="*80)

# Setup
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS').ffill()

sim = MarketSimulator(df=df, seed=42)
sim.fit()

print(f"\nHistorical Max Affordability: {sim.historical_max_affordability:.4f}")
print(f"Soft landing zone (95%):      {sim.historical_max_affordability * 0.95:.4f}")

# Run a quick 30-month test
paths = sim.forecast_price(iterations=1, steps=30)
prices = paths.iloc[:, 0]

# Check affordability during forecast
print("\nForecast (30 months):")
print(f"  Starting price (from log return): {prices.iloc[0]:.2f}")
print(f"  Final price:                     {prices.iloc[-1]:.2f}")
print(f"  Growth factor:                   {prices.iloc[-1] / prices.iloc[0]:.4f}x")

# Check if growth is reasonable (should NOT be exploding)
expected_annual_growth = 0.05  # ~5% annually is reasonable
months = 30
expected_growth = (1 + expected_annual_growth) ** (months / 12)
actual_growth = prices.iloc[-1] / prices.iloc[0]

print(f"\n  Expected growth (5% annually): {expected_growth:.4f}x")
print(f"  Actual growth:                 {actual_growth:.4f}x")

if actual_growth > expected_growth * 2:
    print(f"  ❌ STILL EXPLODING: {actual_growth:.2f}x is way too high!")
elif actual_growth > expected_growth * 1.5:
    print(f"  ⚠️  STILL TOO HIGH: {actual_growth:.2f}x, should be closer to {expected_growth:.2f}x")
else:
    print(f"  ✅ REASONABLE: Growth is within expected bounds")

print("\n" + "="*80)
