import pandas as pd
import numpy as np
import traceback
import sys, os
# ensure repository root is on sys.path so `scripts` package can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator


def main():
    try:
        df = pd.read_csv('data/processed_data.csv')
        
        # --- ENSURE PROPER MONTHLY FREQUENCY ---
        # 1. Ensure the Date column is actually datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Set the Date as the Index
        df.set_index('date', inplace=True)
        
        # 3. FORCE the frequency to be 'Month Start' (MS)
        # This fills any missing months with NaNs (which we then fill) to ensure a perfect heartbeat
        df = df.asfreq('MS')
        
        # 4. Fill any gaps created by forcing the frequency
        # (Forward fill: use last known value for empty months)
        df = df.ffill()
        
        # 5. Verify it worked (Should print 'MS')
        print(f"Data Frequency is now set to: {df.index.freq}")
        # --- END FREQUENCY ALIGNMENT ---
        
        sim = MarketSimulator(df)
        print('Fitting models (this may take a while)...')
        sim.fit()
        print('Models fitted. Running Monte Carlo forecast...')

        print('Running Monte Carlo: 100 iterations x 300 months')
        paths = sim.forecast_price(iterations=100, steps=300)
        out_path = 'data/simulated_price_paths.csv'
        paths.to_csv(out_path)
        final_date = paths.index[-1]
        pct = paths.loc[final_date].quantile([0.05, 0.5, 0.95])
        print(f'Final simulated date: {final_date}')
        print('2050 price percentiles (5%,50%,95%):')
        print(pct)
        print(f'Saved simulated paths to {out_path}')
        
        # Also output extended dataframe with exogenous variables and mean/percentile prices
        extended = sim.get_extended_forecast(paths)
        extended_path = 'data/extended_forecast_2050.csv'
        extended.to_csv(extended_path)
        print(f'Saved extended forecast (exogenous + prices) to {extended_path}')
        
        # Display sample of extended forecast (first, middle, and last rows)
        print('\n--- EXTENDED FORECAST SAMPLE (exogenous variables + price predictions) ---')
        print(f'Total rows: {len(extended)}, Total columns: {len(extended.columns)}')
        print('\nFirst 5 rows (May 2025 - Sept 2025):')
        print(extended.iloc[:5][['GDP_Growth_YoY', 'variable_mortgage_rate', 'housing_starts_per_cap', 'Price_median', 'Price_5pct', 'Price_95pct']])
        print('\nMiddle rows (approx. Oct 2037):')
        mid_idx = len(extended) // 2
        print(extended.iloc[mid_idx:mid_idx+3][['GDP_Growth_YoY', 'variable_mortgage_rate', 'housing_starts_per_cap', 'Price_median', 'Price_5pct', 'Price_95pct']])
        print('\nFinal rows (Jan-Mar 2050):')
        print(extended.iloc[-3:][['GDP_Growth_YoY', 'variable_mortgage_rate', 'housing_starts_per_cap', 'Price_median', 'Price_5pct', 'Price_95pct']])
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()
