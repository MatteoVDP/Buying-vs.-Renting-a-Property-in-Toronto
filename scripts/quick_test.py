import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator

# Load and prepare data
df = pd.read_csv('data/processed_data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('MS')
df = df.ffill()
print(f"Data Frequency: {df.index.freq}")

# Quick test with 5 iterations
sim = MarketSimulator(df)
print('Fitting models...')
sim.fit()
print('Models fitted. Running 5-iteration forecast test...')
paths = sim.forecast_price(iterations=5, steps=300)
print(f'Forecast complete. Price paths shape: {paths.shape}')

# Test extended forecast
print('Creating extended forecast...')
extended = sim.get_extended_forecast(paths)
print(f'Extended forecast shape: {extended.shape}')
print(f'Extended forecast columns: {list(extended.columns)[-5:]}')

# Final prices
final_date = paths.index[-1]
print(f'Final date: {final_date}')
print(f'Final price percentiles:')
print(paths.loc[final_date].quantile([0.05, 0.5, 0.95]))

print('\nTest successful!')
