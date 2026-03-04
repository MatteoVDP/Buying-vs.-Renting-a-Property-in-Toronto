#!/usr/bin/env python3
"""
Diagnostic script to check what data is being generated in audit_updated
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator

# Load data
print("Loading data...")
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS')
df = df.ffill()

print(f"Historical data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Initialize simulator
print("\nInitializing simulator...")
sim = MarketSimulator(df, seed=42)

# Fit
print("Fitting models...")
sim.fit()

# Run forecast
print("\nRunning forecast...")
last_date = df.index.max()
future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=300, freq='MS')
start_market_price = 1090326.0
sim.start_market_price = float(start_market_price)

price_paths = sim.forecast_price(iterations=1, steps=300)

print(f"\n{'='*80}")
print("PRICE_PATHS DIAGNOSTIC")
print(f"{'='*80}")
print(f"Type: {type(price_paths)}")
print(f"Shape: {price_paths.shape}")
print(f"Columns: {price_paths.columns.tolist()}")
print(f"Index (first 5): {price_paths.index[:5].tolist()}")
print(f"Index (last 5): {price_paths.index[-5:].tolist()}")

# Check the values
print(f"\nFirst 5 values of iter_0 column (log returns):")
print(price_paths.iloc[:5, 0].values)

print(f"\nStatistics of log returns:")
print(f"  Min: {price_paths.iloc[:, 0].min():.6f}")
print(f"  Max: {price_paths.iloc[:, 0].max():.6f}")
print(f"  Mean: {price_paths.iloc[:, 0].mean():.6f}")
print(f"  Count NaN: {price_paths.iloc[:, 0].isna().sum()}")

# Try to generate forecast prices as in the script
forecast_prices = np.exp(price_paths.iloc[:, 0].values) * start_market_price

print(f"\n{'='*80}")
print("FORECAST PRICES DIAGNOSTIC")
print(f"{'='*80}")
print(f"Type: {type(forecast_prices)}")
print(f"Shape: {forecast_prices.shape}")
print(f"First 5 prices: {forecast_prices[:5]}")
print(f"Last 5 prices: {forecast_prices[-5:]}")
print(f"Min price: ${forecast_prices.min():,.2f}")
print(f"Max price: ${forecast_prices.max():,.2f}")
print(f"Mean price: ${forecast_prices.mean():,.2f}")
print(f"Count NaN: {np.isnan(forecast_prices).sum()}")
print(f"Count Inf: {np.isinf(forecast_prices).sum()}")

# Check if values are reasonable
if forecast_prices.max() > 1e9:
    print("\n⚠️  WARNING: Prices exceeding $1 billion (explosion detected)")
elif forecast_prices.min() < 100000:
    print("\n⚠️  WARNING: Prices below $100k (collapse detected)")
else:
    print("\n✓ Prices appear reasonable")

print(f"\n{'='*80}")
print("FUTURE INDEX DIAGNOSTIC")
print(f"{'='*80}")
print(f"Type: {type(future_index)}")
print(f"Length: {len(future_index)}")
print(f"First 5 dates: {future_index[:5].tolist()}")
print(f"Last 5 dates: {future_index[-5:].tolist()}")
print(f"Same length as forecast_prices? {len(future_index) == len(forecast_prices)}")

print(f"\n{'='*80}")
print("EXTENDED FORECAST DIAGNOSTIC")
print(f"{'='*80}")
extended = sim.get_extended_forecast(price_paths)
print(f"Type: {type(extended)}")
print(f"Shape: {extended.shape}")
print(f"Columns: {extended.columns.tolist()}")
print(f"Has Price_mean? {'Price_mean' in extended.columns}")
if 'Price_mean' in extended.columns:
    print(f"Price_mean first 5: {extended['Price_mean'].head(5).values}")
    print(f"Price_mean last 5: {extended['Price_mean'].tail(5).values}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Forecast prices are ready for plotting: {len(forecast_prices) > 0 and np.isfinite(forecast_prices).all()}")
print(f"Future index is ready: {len(future_index) > 0}")
print(f"Initial price (start): ${forecast_prices[0]:,.2f}")
print(f"Final price (2050): ${forecast_prices[-1]:,.2f}")
print(f"Total 25-year change: {100*(forecast_prices[-1]/forecast_prices[0] - 1):+.2f}%")
