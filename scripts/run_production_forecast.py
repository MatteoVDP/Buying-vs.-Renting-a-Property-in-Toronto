"""
Production Forecasting Script: Large-Scale Monte Carlo Simulation
==================================================================
Executes 1,000 independent Monte Carlo iterations to forecast housing prices
over the next 25 years (300 months). Generates a fan chart, aggregates
statistical metrics, and exports results for financial risk analysis.

Author: Quantitative Development Team
Date: 2026
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Add script directory to path to import MarketSimulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from market_simulator import MarketSimulator


def load_historical_data(data_path: str) -> pd.DataFrame:
    """Load the full historical dataset."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading historical data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def initialize_simulator(df: pd.DataFrame, seed: int = 42) -> MarketSimulator:
    """Initialize the MarketSimulator with seed."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing MarketSimulator (seed={seed})...")
    # Use the last known market price from March 2025 as reference
    simulator = MarketSimulator(df, seed=seed, start_market_price=1090326.0)
    return simulator


def train_master_model(simulator: MarketSimulator, df: pd.DataFrame) -> None:
    """Train the master model on the entire historical dataset."""
    train_start = datetime.now()
    print(f"[{train_start.strftime('%Y-%m-%d %H:%M:%S')}] === TRAINING PHASE STARTED ===")
    print(f"[{train_start.strftime('%Y-%m-%d %H:%M:%S')}] Training on full dataset: {df.shape[0]} months")
    
    simulator.fit(df)
    
    train_end = datetime.now()
    train_duration = (train_end - train_start).total_seconds()
    print(f"[{train_end.strftime('%Y-%m-%d %H:%M:%S')}] === TRAINING PHASE COMPLETED ===")
    print(f"[{train_end.strftime('%Y-%m-%d %H:%M:%S')}] Training duration: {train_duration:.2f} seconds\n")


def run_monte_carlo_simulation(simulator: MarketSimulator, steps: int = 300, iterations: int = 25) -> pd.DataFrame:
    """Execute the large-scale Monte Carlo simulation."""
    sim_start = datetime.now()
    print(f"[{sim_start.strftime('%Y-%m-%d %H:%M:%S')}] === MONTE CARLO SIMULATION STARTED ===")
    print(f"[{sim_start.strftime('%Y-%m-%d %H:%M:%S')}] Running {iterations} iterations over {steps} months (25 years)...")
    
    # Run the multiverse simulation
    price_paths = simulator.forecast_price(steps=steps, iterations=iterations)
    
    sim_end = datetime.now()
    sim_duration = (sim_end - sim_start).total_seconds()
    print(f"[{sim_end.strftime('%Y-%m-%d %H:%M:%S')}] === MONTE CARLO SIMULATION COMPLETED ===")
    print(f"[{sim_end.strftime('%Y-%m-%d %H:%M:%S')}] Simulation duration: {sim_duration:.2f} seconds")
    print(f"[{sim_end.strftime('%Y-%m-%d %H:%M:%S')}] Output shape: {price_paths.shape[0]} months × {price_paths.shape[1]} simulations\n")
    
    return price_paths


def aggregate_statistics(price_paths: pd.DataFrame) -> pd.DataFrame:
    """Calculate statistical aggregation across all 1,000 Monte Carlo iterations."""
    agg_start = datetime.now()
    print(f"[{agg_start.strftime('%Y-%m-%d %H:%M:%S')}] Calculating percentiles and summary statistics...")
    
    summary_stats_df = pd.DataFrame(index=price_paths.index)
    
    # Calculate row-by-row statistics across all 1,000 iterations
    summary_stats_df['Bear_10th'] = price_paths.quantile(0.10, axis=1)
    summary_stats_df['Base_50th'] = price_paths.quantile(0.50, axis=1)
    summary_stats_df['Bull_90th'] = price_paths.quantile(0.90, axis=1)
    summary_stats_df['Absolute_Max'] = price_paths.max(axis=1)
    summary_stats_df['Absolute_Min'] = price_paths.min(axis=1)
    
    agg_end = datetime.now()
    agg_duration = (agg_end - agg_start).total_seconds()
    print(f"[{agg_end.strftime('%Y-%m-%d %H:%M:%S')}] Aggregation complete in {agg_duration:.2f} seconds\n")
    
    return summary_stats_df


def create_fan_chart(price_paths: pd.DataFrame, summary_stats_df: pd.DataFrame, 
                     df_historical: pd.DataFrame, output_path: str) -> None:
    """Create and save the fan chart visualization with historical data from 1968 and forecast to 2050."""
    viz_start = datetime.now()
    print(f"[{viz_start.strftime('%Y-%m-%d %H:%M:%S')}] === VISUALIZATION PHASE STARTED ===")
    print(f"[{viz_start.strftime('%Y-%m-%d %H:%M:%S')}] Creating fan chart with {price_paths.shape[1]} simulation paths...")
    
    # Extract historical prices and dates
    historical_dates = pd.to_datetime(df_historical['date'])
    
    # Use the market_price_target_average column (lag_1 is the most recent price)
    # Calculate actual prices from log prices using exponential transformation
    historical_prices = np.exp(df_historical['Log_Price_lag_1'].astype(float))
    
    # Get the last historical date for reference
    last_historical_date = historical_dates.iloc[-1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Historical data spans from {historical_dates.iloc[0].strftime('%Y-%m-%d')} to {last_historical_date.strftime('%Y-%m-%d')}")
    
    # Create date index for forecast period (300 months = 25 years, target 2050)
    # From 2025-03 to 2050-03 is 25 years or 300 months
    forecast_dates = [last_historical_date + relativedelta(months=i) for i in range(1, len(summary_stats_df) + 1)]
    forecast_dates = pd.to_datetime(forecast_dates)
    
    # Create combined timeline
    combined_dates = pd.concat([pd.Series(historical_dates), pd.Series(forecast_dates)])
    combined_dates.reset_index(drop=True, inplace=True)
    
    # Extend price_paths and summary_stats_df indices to match combined dates
    price_paths_with_dates = price_paths.copy()
    price_paths_with_dates.index = forecast_dates
    
    summary_stats_df_with_dates = summary_stats_df.copy()
    summary_stats_df_with_dates.index = forecast_dates
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot historical data as a baseline (solid dark blue line)
    ax.plot(historical_dates, historical_prices, 
            color='darkblue', linewidth=2.5, label='Historical (1968-2025)', zorder=10)
    
    # Plot all simulation paths with high transparency
    for col in price_paths_with_dates.columns:
        ax.plot(forecast_dates, price_paths_with_dates[col], 
                color='lightgray', alpha=0.02, linewidth=0.5)
    
    # Overlay percentile lines for forecast with bold, distinct colors
    
    # Bear case (10th percentile) - Red
    ax.plot(forecast_dates, summary_stats_df_with_dates['Bear_10th'], 
            color='red', linewidth=2.5, label='Bear (10th %ile)', linestyle='-', alpha=0.8, zorder=9)
    
    # Base case (50th percentile / median) - Black
    ax.plot(forecast_dates, summary_stats_df_with_dates['Base_50th'], 
            color='black', linewidth=2.5, label='Base (50th %ile)', linestyle='-', alpha=0.9, zorder=9)
    
    # Bull case (90th percentile) - Green
    ax.plot(forecast_dates, summary_stats_df_with_dates['Bull_90th'], 
            color='green', linewidth=2.5, label='Bull (90th %ile)', linestyle='-', alpha=0.8, zorder=9)
    
    # Add a vertical line to mark the transition between historical and forecast
    ax.axvline(x=last_historical_date, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=5)
    ax.text(last_historical_date, ax.get_ylim()[1] * 0.95, 'Forecast Start', 
            rotation=0, fontsize=10, color='gray', alpha=0.7, ha='right')
    
    # Chart formatting
    ax.set_title('Toronto Housing Market: Historical Data (1968-2025) & 25-Year Monte Carlo Forecast to 2050', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Market Price ($CAD)', fontsize=12, fontweight='bold')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fan chart saved to {output_path}")
    plt.close()
    
    viz_end = datetime.now()
    viz_duration = (viz_end - viz_start).total_seconds()
    print(f"[{viz_end.strftime('%Y-%m-%d %H:%M:%S')}] === VISUALIZATION PHASE COMPLETED ===")
    print(f"[{viz_end.strftime('%Y-%m-%d %H:%M:%S')}] Visualization duration: {viz_duration:.2f} seconds\n")


def export_results(price_paths: pd.DataFrame, summary_stats_df: pd.DataFrame,
                   paths_output: str, stats_output: str) -> None:
    """Export raw simulations and summary statistics to CSV files."""
    export_start = datetime.now()
    print(f"[{export_start.strftime('%Y-%m-%d %H:%M:%S')}] === DATA EXPORT PHASE STARTED ===")
    
    # Export raw simulation paths
    print(f"[{export_start.strftime('%Y-%m-%d %H:%M:%S')}] Exporting {price_paths.shape[1]} raw simulation paths...")
    price_paths.to_csv(paths_output, index=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Raw simulations saved to {paths_output}")
    
    # Export aggregated statistics
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Exporting summary statistics...")
    summary_stats_df.to_csv(stats_output, index=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Summary statistics saved to {stats_output}")
    
    export_end = datetime.now()
    export_duration = (export_end - export_start).total_seconds()
    print(f"[{export_end.strftime('%Y-%m-%d %H:%M:%S')}] === DATA EXPORT PHASE COMPLETED ===")
    print(f"[{export_end.strftime('%Y-%m-%d %H:%M:%S')}] Export duration: {export_duration:.2f} seconds\n")


def main():
    """Main execution pipeline."""
    overall_start = datetime.now()
    print("=" * 80)
    print("PRODUCTION FORECASTING: TORONTO HOUSING MARKET (2026-2050)")
    print("=" * 80)
    print(f"[{overall_start.strftime('%Y-%m-%d %H:%M:%S')}] Job started\n")
    
    # Configuration
    data_path = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv"
    results_dir = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results"
    
    paths_output = os.path.join(results_dir, "final_simulations.csv")
    stats_output = os.path.join(results_dir, "final_summary_stats.csv")
    plot_output = os.path.join(results_dir, "final_simulations_plot.png")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # === PIPELINE EXECUTION ===
    
    # Stage 1: Load Data
    df = load_historical_data(data_path)
    
    # Stage 2: Initialize Simulator
    simulator = initialize_simulator(df, seed=42)
    
    # Stage 3: Train Master Model
    train_master_model(simulator, df)
    
    # Stage 4: Run Monte Carlo Simulation
    price_paths = run_monte_carlo_simulation(simulator, steps=300, iterations=3)
    
    # Stage 5: Aggregate Statistics
    summary_stats_df = aggregate_statistics(price_paths)
    
    # Stage 6: Create Fan Chart (with historical data)
    create_fan_chart(price_paths, summary_stats_df, df, plot_output)
    
    # Stage 7: Export Results
    export_results(price_paths, summary_stats_df, paths_output, stats_output)
    
    # === COMPLETION ===
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    print("=" * 80)
    print("PRODUCTION FORECASTING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"[{overall_end.strftime('%Y-%m-%d %H:%M:%S')}] Total runtime: {overall_duration:.2f} seconds ({overall_duration/60:.2f} minutes)")
    print(f"\nDeliverables:")
    print(f"  1. Raw Simulations (1,000 paths):  {paths_output}")
    print(f"  2. Summary Statistics (percentiles): {stats_output}")
    print(f"  3. Fan Chart Visualization:         {plot_output}")
    print("\nRisk Matrix & Buy vs. Rent Analysis can now proceed with these outputs.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
