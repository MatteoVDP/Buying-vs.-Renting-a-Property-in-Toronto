#!/usr/bin/env python3
"""
Test the Macro Gravity Override with income constraint.
Tests both the price penalty AND the income suppression when affordability is too high.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from market_simulator import MarketSimulator

# Load data
data_path = Path(__file__).parent / "data" / "processed_data.csv"
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print("=" * 80)
print("TESTING MACRO GRAVITY OVERRIDE WITH INCOME SUPPRESSION")
print("=" * 80)

# Split: use first 80% for training
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"\nTrain data: {train_df['Date'].min()} to {train_df['Date'].max()}")
print(f"Test data: {test_df['Date'].min()} to {test_df['Date'].max()}")

# Check historical max affordability
max_aff = train_df['Affordability_Ratio'].max()
print(f"\nHistorical max affordability in training data: {max_aff:.4f}")

# Initialize and fit
sim = MarketSimulator(
    data_df=df.copy(),
    forecast_steps=12,  # 12 months to test
    random_seed=42
)

print("\nFitting market simulator...")
sim.fit(train_df, test_df)

print(f"After fit, historical_max_affordability = {sim.historical_max_affordability:.4f}")

# Now forecast with gravity override
print("\nForecasting 12 months (should see gravity override constrain growth)...")
forecast_df = sim.forecast_price(steps=12)

# Check results
print("\nForecast results (last 5 months):")
print(forecast_df[['Date', 'Price_Forecast', 'Affordability_Forecast']].tail(5).to_string(index=False))

# Analyze the trend
prices = forecast_df['Price_Forecast'].values
affordability = forecast_df['Affordability_Forecast'].values

print(f"\nPrice trend:")
print(f"  First month: ${prices[0]:,.0f}")
print(f"  Last month: ${prices[-1]:,.0f}")
print(f"  Change: {(prices[-1]/prices[0] - 1)*100:+.2f}%")

print(f"\nAffordability trend (should stabilize below {max_aff:.2f}):")
print(f"  First month: {affordability[0]:.4f}")
print(f"  Max: {affordability.max():.4f}")
print(f"  Last month: {affordability[-1]:.4f}")

# Check if gravity override is working
if affordability.max() > max_aff * 1.05:
    print(f"\n⚠️  WARNING: Affordability ratio grew to {affordability.max():.4f}, above historical max {max_aff:.4f}")
    print("   Gravity override may not be effective!")
elif affordability[-1] > max_aff * 0.90:
    print(f"\n⚠️  CAUTION: Affordability at {affordability[-1]:.4f}, approaching historical max")
else:
    print(f"\n✓ SUCCESS: Affordability constrained to {affordability[-1]:.4f}, below historical max {max_aff:.4f}")
    print("  Gravity override with income suppression is working!")

print("\n" + "=" * 80)
