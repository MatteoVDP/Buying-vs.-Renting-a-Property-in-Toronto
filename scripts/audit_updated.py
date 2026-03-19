"""
25-Year Market Forecast: March 2025 to March 2050
================================================
Runs the market simulator to generate a complete price forecast
with all exogenous variables filled for the 25-year period.
Outputs: CSV with complete data + visualization of price path.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.market_simulator import MarketSimulator


def run_25_year_forecast():
    """Run single 25-year (300-month) forecast and save results."""
    
    print("="*80)
    print("25-YEAR MARKET FORECAST: MARCH 2025 - MARCH 2050")
    print("="*80)
    
    # 1. LOAD DATA
    print("\n1. Loading historical market data...")
    df = pd.read_csv('data/processed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.asfreq('MS')
    df = df.ffill()
    
    last_date = df.index.max()
    print(f"   Historical data: {len(df)} rows")
    print(f"   Date range: {df.index.min().strftime('%B %Y')} to {last_date.strftime('%B %Y')}")
    
    # 2. INITIALIZE SIMULATOR
    print("\n2. Initializing MarketSimulator with seed=42...")
    sim = MarketSimulator(df, seed=42)
    
    # 3. FIT MODELS
    print("\n3. Fitting ARIMA/SARIMAX/XGBoost models on historical data...")
    import sys
    sys.stdout.flush()
    sim.fit()
    sys.stdout.flush()
    print("   ✓ Models fitted successfully")
    sys.stdout.flush()
    
    # 4. GENERATE 25-YEAR FORECAST (300 months)
    print("\n4. Generating 25-year (300-month) forecast...")
    steps = 300
    print(f"   Running 1 Monte Carlo iteration x {steps} months...")

    # Use March 2025 market price as the base for log return accumulation
    start_market_price = 1090326.0
    sim.start_market_price = float(start_market_price)

    # Generate future dates
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

    # 5. RUN FORECAST USING SIMULATOR'S BUILT-IN METHOD (single iteration for audit)
    print("\n5. Running single 25-year forecast using MarketSimulator.forecast_price()...")
    print(f"   This uses the full 4-tier model (ARIMA/SARIMAX/XGBoost + sentiment)...")
    
    # Run 1 iteration with 300 months
    forecast_result = sim.forecast_price(iterations=3, steps=steps)
    
    # Extract price paths and full feature history
    price_paths = forecast_result['price_paths']
    full_history = forecast_result['full_history']
    
    # Get extended forecast with all variables and exogenous data
    print(f"   ✓ Forecast complete: {len(price_paths)} months")
    extended = sim.get_extended_forecast(forecast_result)
    
    # 6. BUILD FORECAST DATAFRAME
    print("\n6. Building forecast dataframe with all features...")
    
    # Build forecast dataframe from extended forecast (already has all exogenous + engineered features)
    forecast_df = extended.copy()
    
    # Extract prices from the full history (they were calculated during simulation)
    forecast_start_idx = len(sim.df)
    forecast_prices_from_history = full_history.iloc[forecast_start_idx:]['_Log_Price_Internal'].apply(np.exp).values.astype(float)
    forecast_df['Market_Price'] = forecast_prices_from_history
    
    # Calculate log price
    forecast_df['Log_Price'] = np.log(forecast_df['Market_Price'])
    
    # Calculate price change statistics
    forecast_df['Price_Change_MoM_%'] = forecast_df['Market_Price'].pct_change() * 100
    forecast_df['Price_Change_YoY_%'] = forecast_df['Market_Price'].pct_change(12) * 100
    

    
    # Count derived feature columns
    lag_cols = [c for c in forecast_df.columns if '_lag_' in c]
    delta_cols = [c for c in forecast_df.columns if '_delta_' in c]
    ra_cols = [c for c in forecast_df.columns if '_RA_' in c]
    print(f"   ✓ Forecast complete: {len(forecast_df)} months, {len(forecast_df.columns)} columns")
    print(f"   Derived features: {len(lag_cols)} lags, {len(delta_cols)} deltas, {len(ra_cols)} RAs")
    
    # 7. COMBINE HISTORICAL + FORECAST
    print("\n7. Combining historical and forecast data...")
    
    # Add missing columns to historical data (they won't exist yet)
    for col in ['Market_Price', 'Log_Price', 'Price_Change_MoM_%', 'Price_Change_YoY_%']:
        if col not in df.columns:
            df[col] = np.nan
    
    # Compute historical prices (if not available)
    if df['Market_Price'].isna().all():
        # Use Log_Return_MoM to reconstruct prices (if available)
        # Otherwise mark as historical
        df['Market_Price'] = np.nan
        df['Log_Price'] = np.nan
    
    # Mark data source
    df['Data_Source'] = 'Historical'
    forecast_df['Data_Source'] = 'Forecast'
    
    # Concatenate
    combined_df = pd.concat([df, forecast_df])
    combined_df = combined_df.sort_index()
    
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Total columns: {len(combined_df.columns)}")
    print(f"   Date range: {combined_df.index.min().strftime('%B %Y')} to {combined_df.index.max().strftime('%B %Y')}")
    
    # 8. SAVE TO CSV
    output_csv = 'results/forecast_25_year_march2050.csv'
    print(f"\n7. Saving complete forecast to {output_csv}...")
    combined_df.to_csv(output_csv)
    print(f"   ✓ Saved {len(combined_df)} rows × {len(combined_df.columns)} columns")
    
    # 9. CREATE VISUALIZATION
    print("\n9. Creating price path visualization...")
    
    # Extract the prices from the forecast dataframe
    forecast_prices = forecast_df['Market_Price'].values.astype(float)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # --- SUBPLOT 1: PRICE PATH ---
    ax1 = axes[0]
    
    # Historical prices (from combined_df)
    hist_mask = combined_df['Data_Source'] == 'Historical'
    if hist_mask.any():
        hist_data = combined_df[hist_mask].copy()
        # Get Market_Price from CSV if it exists, otherwise mark as unavailable
        if 'Market_Price' in hist_data.columns and hist_data['Market_Price'].notna().any():
            ax1.plot(hist_data.index, hist_data['Market_Price'], 
                    color='navy', linewidth=2.5, label='Historical Prices', alpha=0.9, marker='o', markersize=2)
        else:
            ax1.text(0.05, 0.95, '(Historical prices not available in dataset)', 
                    transform=ax1.transAxes, fontsize=9, style='italic', alpha=0.6, va='top')
    
    # Forecast prices (from simulation)
    ax1.plot(future_index, forecast_prices, 
            color='crimson', linewidth=2.5, label='Forecast (25-year horizon)', 
            linestyle='-', alpha=0.85, marker='o', markersize=1)
    
    # Add vertical line at forecast start
    ax1.axvline(x=last_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.text(last_date, ax1.get_ylim()[1] * 0.95, ' Forecast Start', 
            fontsize=9, color='gray', va='top')
    
    # Formatting
    ax1.set_title('Toronto Housing Market: Historical + 25-Year Forecast (March 2025 - March 2050)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Market Price ($)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
    
    # Format x-axis dates
    ax1.xaxis.set_major_locator(mdates.YearLocator(3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # --- SUBPLOT 2: MONTHLY RETURNS ---
    ax2 = axes[1]
    
    if len(forecast_prices) > 1:
        monthly_returns = forecast_prices[1:] / forecast_prices[:-1] - 1
        ax2.plot(future_index[1:], monthly_returns, 
                color='crimson', linewidth=1.5, alpha=0.7, marker='o', markersize=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax2.set_title('Monthly Price Growth Rate', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Growth Rate', fontsize=11, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.2f}%'))
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Format x-axis dates
        ax2.xaxis.set_major_locator(mdates.YearLocator(3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for visualization', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    output_fig = 'results/forecast_25_year_march2050.png'
    print(f"   Saving visualization to {output_fig}...")
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved")
    
    # 10. SUMMARY STATISTICS
    print("\n10. Forecast Summary Statistics:")
    print("   " + "="*76)
    
    if len(forecast_prices) > 0:
        initial_price = forecast_prices[0]
        final_price = forecast_prices[-1]
        min_price = np.min(forecast_prices)
        max_price = np.max(forecast_prices)
        mean_price = np.mean(forecast_prices)
        
        total_change_pct = ((final_price / initial_price) - 1) * 100
        total_change_avg_annual = ((final_price / initial_price) ** (1/25) - 1) * 100
        
        # Calculate monthly returns
        monthly_returns = forecast_prices[1:] / forecast_prices[:-1] - 1
        
        print(f"   Initial Price (Start of Forecast): ${initial_price:,.2f}")
        print(f"   Final Price (March 2050):          ${final_price:,.2f}")
        print(f"   Lowest Price:                      ${min_price:,.2f}")
        print(f"   Highest Price:                     ${max_price:,.2f}")
        print(f"   Mean Price:                        ${mean_price:,.2f}")
        print(f"\n   Total 25-Year Change:              {total_change_pct:+.2f}%")
        print(f"   Average Annual Growth Rate:        {total_change_avg_annual:+.2f}%")
        print(f"\n   Monthly Returns - Mean:            {np.mean(monthly_returns)*100:+.3f}%")
        print(f"   Monthly Returns - Std Dev:         {np.std(monthly_returns)*100:.3f}%")
        print(f"   Monthly Returns - Min:             {np.min(monthly_returns)*100:+.3f}%")
        print(f"   Monthly Returns - Max:             {np.max(monthly_returns)*100:+.3f}%")
    
    # Feature Engineering Diagnostics
    print("\n   " + "-"*76)
    print("   DERIVED FEATURES DIAGNOSTICS:")
    print("   " + "-"*76)
    
    # Check for lag/delta/RA columns
    lag_cols = [c for c in combined_df.columns if '_lag_' in c]
    delta_cols = [c for c in combined_df.columns if '_delta_' in c]
    ra_cols = [c for c in combined_df.columns if '_RA_' in c]
    
    print(f"   Lag columns defined: {len(lag_cols)}")
    print(f"   Delta columns defined: {len(delta_cols)}")
    print(f"   Rolling Average columns defined: {len(ra_cols)}")
    
    # Sample a few rows from forecast to check if lags/deltas are populated
    forecast_with_features = combined_df[combined_df['Data_Source'] == 'Forecast'].copy()
    if len(forecast_with_features) > 24:
        sample_row = forecast_with_features.iloc[24]  # Check row after 24 months
        populated_lags = sum(1 for c in lag_cols if pd.notna(sample_row.get(c)))
        populated_deltas = sum(1 for c in delta_cols if pd.notna(sample_row.get(c)))
        populated_ras = sum(1 for c in ra_cols if pd.notna(sample_row.get(c)))
        
        print(f"\n   At Month 24 of forecast:")
        print(f"     - Lags populated: {populated_lags}/{len(lag_cols)}")
        print(f"     - Deltas populated: {populated_deltas}/{len(delta_cols)}")
        print(f"     - Rolling Averages populated: {populated_ras}/{len(ra_cols)}")
    
    print("   " + "="*76)
    
    # 11. COMPLETION
    print("\n" + "="*80)
    print("FORECAST GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  1. Complete CSV: {output_csv}")
    print(f"     - {len(combined_df)} rows (historical + forecast)")
    print(f"     - All exogenous variables filled for forecast period")
    print(f"     - Price projections from March 2025 to March 2050")
    print(f"\n  2. Visualization: {output_fig}")
    print(f"     - Monthly price path with confidence visual")
    print(f"     - Monthly percent changes")
    print(f"\nRecommended next steps:")
    print(f"  - Review CSV in Excel for detailed month-by-month data")
    print(f"  - Open PNG to visualize the forecast trajectory")
    print(f"  - Compare with multiple iterations if desired")
    

if __name__ == '__main__':
    run_25_year_forecast()
