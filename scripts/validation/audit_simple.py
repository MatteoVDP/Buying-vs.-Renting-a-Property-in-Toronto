#!/usr/bin/env python3
"""
Simplified 25-Year Market Forecast with Clean Visualization
Runs the market simulator to generate a complete price forecast with gravity override.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator
from scripts.visualization_helper import create_forecast_visualization


def run_25_year_forecast():
    """Run single 25-year (300-month) forecast with visualization."""
    
    print("="*80)
    print("25-YEAR HOUSING MARKET FORECAST: MARCH 2025 - MARCH 2050")
    print("="*80)
    
    # === STEP 1: LOAD DATA ===
    print("\n1. Loading historical market data...")
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    df = df.ffill()
    
    print(f"   Historical data: {len(df)} months")
    print(f"   Date range: {df.index.min().strftime('%B %Y')} to {df.index.max().strftime('%B %Y')}")
    
    last_date = df.index.max()
    
    # === STEP 2: INITIALIZE SIMULATOR ===
    print("\n2. Initializing MarketSimulator...")
    sim = MarketSimulator(df, seed=42, start_market_price=1090326.0)
    
    # === STEP 3: FIT MODELS ===
    print("\n3. Fitting 4-tier forecasting model...")
    print("   (ARIMA for Tier 1, SARIMAX for Tiers 2-3, XGBoost for prices)")
    sim.fit(df)
    print("   ✓ Model fitting complete")
    
    # === STEP 4: GENERATE FORECAST ===
    print("\n4. Generating 25-year Monte Carlo forecast...")
    steps = 300
    start_market_price = 1090326.0
    
    # Run 1 iteration for the audit
    price_paths = sim.forecast_price(iterations=1, steps=steps)
    
    print(f"   ✓ Generated {steps} months of forecasts")
    
    # === STEP 5: PROCESS RESULTS ===
    print("\n5. Processing forecast results...")
    
    # Generate future dates
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')
    
    # forecast_price() already returns simulated market prices
    forecast_prices = price_paths.iloc[:, 0].values.astype(float)
    
    print(f"   Starting price: ${forecast_prices[0]:,.2f}")
    print(f"   Ending price (March 2050): ${forecast_prices[-1]:,.2f}")
    print(f"   Min price: ${forecast_prices.min():,.2f}")
    print(f"   Max price: ${forecast_prices.max():,.2f}")
    
    # Get extended forecast for affordability data
    extended = sim.get_extended_forecast(price_paths)
    affordability_data = extended.get('Affordability_Ratio', None)
    
    # === STEP 6: CREATE VISUALIZATION ===
    print("\n6. Creating visualization...")
    output_fig = 'results/forecast_25_year_march2050.png'
    os.makedirs('results', exist_ok=True)
    
    create_forecast_visualization(
        forecast_prices=forecast_prices,
        future_index=future_index,
        start_market_price=start_market_price,
        affordability_data=affordability_data,
        sim=sim,
        output_path=output_fig
    )
    
    # === STEP 7: SAVE FORECAST DATA ===
    print("\n7. Saving forecast data to CSV...")
    
    # Build output dataframe
    output_df = pd.DataFrame(index=future_index)
    output_df['Market_Price'] = forecast_prices
    output_df['Log_Price'] = np.log(forecast_prices)
    output_df['Price_Change_MoM_%'] = pd.Series(forecast_prices, index=future_index).pct_change() * 100
    
    # Add exogenous variables if available
    if extended is not None:
        for col in extended.columns:
            if col not in output_df.columns:
                try:
                    output_df[col] = extended[col].values
                except:
                    pass
    
    output_csv = 'results/forecast_25_year_march2050.csv'
    output_df.to_csv(output_csv)
    print(f"   ✓ Saved {len(output_df)} rows × {len(output_df.columns)} columns to {output_csv}")
    
    # === STEP 8: PRINT SUMMARY STATISTICS ===
    print("\n8. Forecast Summary Statistics")
    print("   " + "="*76)
    
    total_change_pct = ((forecast_prices[-1] / forecast_prices[0]) - 1) * 100
    avg_annual_change = ((forecast_prices[-1] / forecast_prices[0]) ** (1/25) - 1) * 100
    
    print(f"\n   PRICE TRAJECTORY:")
    print(f"   Starting Price (Mar 2025):        ${forecast_prices[0]:,.2f}")
    print(f"   Ending Price (Mar 2050):         ${forecast_prices[-1]:,.2f}")
    print(f"   Total 25-Year Change:            {total_change_pct:+.2f}%")
    print(f"   Average Annual Growth:           {avg_annual_change:+.2f}%")
    print(f"   Lowest Price in Forecast:        ${forecast_prices.min():,.2f}")
    print(f"   Highest Price in Forecast:       ${forecast_prices.max():,.2f}")
    
    # Monthly stats
    monthly_returns = (forecast_prices[1:] / forecast_prices[:-1] - 1) * 100
    print(f"\n   MONTHLY RETURNS:")
    print(f"   Mean Monthly Return:             {np.mean(monthly_returns):+.3f}%")
    print(f"   Std Dev Monthly Return:          {np.std(monthly_returns):.3f}%")
    print(f"   Min Monthly Return:              {np.min(monthly_returns):+.3f}%")
    print(f"   Max Monthly Return:              {np.max(monthly_returns):+.3f}%")
    
    # Affordability stats
    if affordability_data is not None:
        aff_valid = affordability_data.dropna()
        if len(aff_valid) > 0:
            print(f"\n   AFFORDABILITY RATIO:")
            print(f"   Starting Affordability:          {aff_valid.iloc[0]:.4f}")
            print(f"   Ending Affordability:            {aff_valid.iloc[-1]:.4f}")
            print(f"   Peak Affordability in Forecast:  {aff_valid.max():.4f}")
            print(f"   Historical Max Threshold:        {sim.historical_max_affordability:.4f}")
            print(f"   Threshold Exceeded?              {'YES ⚠️' if aff_valid.max() > sim.historical_max_affordability else 'NO ✓'}")
    
    print("\n" + "="*80)
    print("FORECAST COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  • Visualization: {output_fig}")
    print(f"  • Data Export: {output_csv}")
    print(f"\n✓ Successfully completed with Macro Gravity Override active")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_25_year_forecast()
