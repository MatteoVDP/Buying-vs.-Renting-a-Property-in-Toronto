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


TARGET_DATE = "2050-03-01"
TARGET_PRICE_THRESHOLD = 5_590_000.0


def _extract_price_paths(forecast_result) -> pd.DataFrame:
    """Normalize forecast output to a non-empty DataFrame of price paths."""
    price_paths = forecast_result.get('price_paths') if isinstance(forecast_result, dict) else forecast_result
    if not isinstance(price_paths, pd.DataFrame):
        raise TypeError(
            f"forecast_price() must return DataFrame or dict with DataFrame 'price_paths', got {type(price_paths)}"
        )
    if price_paths.empty:
        raise ValueError("forecast_price() produced an empty price_paths DataFrame")
    return price_paths


def _rebuild_prices_from_log_returns(log_returns: pd.Series, terminal_price: float) -> pd.Series:
    """Rebuild a level price path from monthly log returns and a terminal anchor price."""
    returns = pd.to_numeric(log_returns, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    if len(returns) == 0:
        return pd.Series(dtype=float)

    log_prices = np.empty(len(returns), dtype=float)
    log_prices[-1] = float(np.log(terminal_price))
    for i in range(len(returns) - 2, -1, -1):
        log_prices[i] = log_prices[i + 1] - float(returns[i + 1])

    return pd.Series(np.exp(log_prices), index=log_returns.index)


def load_historical_data(data_path: str) -> pd.DataFrame:
    """Load the full historical dataset."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading historical data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def initialize_simulator(df: pd.DataFrame, seed: int = 42, start_market_price: float = 1090326.0) -> MarketSimulator:
    """Initialize the MarketSimulator with seed."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing MarketSimulator (seed={seed})...")
    # Use the last known market price from March 2025 as reference
    simulator = MarketSimulator(df, seed=seed, start_market_price=start_market_price)
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
    forecast_result = simulator.forecast_price(steps=steps, iterations=iterations)
    price_paths = _extract_price_paths(forecast_result)
    
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


def build_terminal_performance_stats(
    price_paths: pd.DataFrame,
    target_date: str = TARGET_DATE,
    threshold_price: float = TARGET_PRICE_THRESHOLD,
) -> pd.DataFrame:
    """Create one-row terminal performance stats from the final forecast month."""
    if price_paths.empty:
        raise ValueError("Cannot build terminal performance stats from empty price_paths")

    terminal_prices = pd.to_numeric(price_paths.iloc[-1], errors='coerce').dropna()
    if terminal_prices.empty:
        raise ValueError("Terminal forecast row contains no valid numeric values")

    simulation_count = int(terminal_prices.shape[0])
    threshold_diff = terminal_prices - threshold_price

    bear_10th = float(terminal_prices.quantile(0.10))
    p25 = float(terminal_prices.quantile(0.25))
    median_50th = float(terminal_prices.quantile(0.50))
    p75 = float(terminal_prices.quantile(0.75))
    bull_90th = float(terminal_prices.quantile(0.90))
    mean_price = float(terminal_prices.mean())
    std_dev = float(terminal_prices.std(ddof=1)) if simulation_count > 1 else 0.0
    absolute_min = float(terminal_prices.min())
    absolute_max = float(terminal_prices.max())

    count_strictly_gt_threshold = int((terminal_prices > threshold_price).sum())
    count_at_or_above_threshold = int((terminal_prices >= threshold_price).sum())
    count_strictly_lt_threshold = int((terminal_prices < threshold_price).sum())
    count_at_or_below_threshold = int((terminal_prices <= threshold_price).sum())

    overshoot_values = threshold_diff[threshold_diff > 0]
    shortfall_values = -threshold_diff[threshold_diff <= 0]
    mean_overshoot_if_gt = float(overshoot_values.mean()) if not overshoot_values.empty else 0.0
    max_overshoot_if_gt = float(overshoot_values.max()) if not overshoot_values.empty else 0.0
    mean_shortfall_if_le = float(shortfall_values.mean()) if not shortfall_values.empty else 0.0
    max_shortfall_if_le = float(shortfall_values.max()) if not shortfall_values.empty else 0.0

    terminal_stats_df = pd.DataFrame(
        [
            {
                'Target_Date': target_date,
                'Forecast_Horizon_Months': int(price_paths.shape[0]),
                'Simulation_Count': simulation_count,
                'Bear_10th': bear_10th,
                'P25': p25,
                'Median_50th': median_50th,
                'P75': p75,
                'Bull_90th': bull_90th,
                'Mean': mean_price,
                'Std_Dev': std_dev,
                'IQR_P75_minus_P25': float(p75 - p25),
                'Absolute_Min': absolute_min,
                'Absolute_Max': absolute_max,
                'Threshold_Price': float(threshold_price),
                'Count_Strictly_GT_Threshold': count_strictly_gt_threshold,
                'Share_Strictly_GT_Threshold': float(count_strictly_gt_threshold / simulation_count),
                'Count_At_Or_Above_Threshold': count_at_or_above_threshold,
                'Share_At_Or_Above_Threshold': float(count_at_or_above_threshold / simulation_count),
                'Count_Strictly_LT_Threshold': count_strictly_lt_threshold,
                'Share_Strictly_LT_Threshold': float(count_strictly_lt_threshold / simulation_count),
                'Count_At_Or_Below_Threshold': count_at_or_below_threshold,
                'Share_At_Or_Below_Threshold': float(count_at_or_below_threshold / simulation_count),
                'Mean_Deviation_From_Threshold': float(mean_price - threshold_price),
                'Median_Deviation_From_Threshold': float(median_50th - threshold_price),
                'Bear_10th_Deviation_From_Threshold': float(bear_10th - threshold_price),
                'Bull_90th_Deviation_From_Threshold': float(bull_90th - threshold_price),
                'Absolute_Min_Deviation_From_Threshold': float(absolute_min - threshold_price),
                'Absolute_Max_Deviation_From_Threshold': float(absolute_max - threshold_price),
                'Mean_Absolute_Deviation_From_Threshold': float(np.abs(threshold_diff).mean()),
                'RMSE_From_Threshold': float(np.sqrt(np.mean(np.square(threshold_diff)))),
                'Mean_to_Threshold_Ratio': float(mean_price / threshold_price),
                'Median_to_Threshold_Ratio': float(median_50th / threshold_price),
                'Absolute_Min_to_Threshold_Ratio': float(absolute_min / threshold_price),
                'Absolute_Max_to_Threshold_Ratio': float(absolute_max / threshold_price),
                'Mean_Overshoot_If_GT_Threshold': mean_overshoot_if_gt,
                'Max_Overshoot_If_GT_Threshold': max_overshoot_if_gt,
                'Mean_Shortfall_If_LE_Threshold': mean_shortfall_if_le,
                'Max_Shortfall_If_LE_Threshold': max_shortfall_if_le,
                'Threshold_Z_Score_vs_Terminal_Distribution': (
                    float((threshold_price - mean_price) / std_dev) if std_dev > 0 else np.nan
                ),
            }
        ]
    )
    return terminal_stats_df


def _render_stats_table(ax, rows, title):
    """Render a compact two-column stats table on the provided axes."""
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    table = ax.table(
        cellText=[[label, value] for label, value in rows],
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor('#1f4e79')
            cell.set_text_props(color='white', weight='bold')
        elif row_idx % 2 == 0:
            cell.set_facecolor('#f4f7fb')


def export_terminal_performance_stats(
    price_paths: pd.DataFrame,
    output_path: str,
    target_date: str = TARGET_DATE,
    threshold_price: float = TARGET_PRICE_THRESHOLD,
) -> None:
    """Export terminal distribution stats to a standalone image dashboard."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Exporting terminal performance stats...")
    terminal_stats_df = build_terminal_performance_stats(
        price_paths=price_paths,
        target_date=target_date,
        threshold_price=threshold_price,
    )
    stats = terminal_stats_df.iloc[0]

    left_rows = [
        ('Target Date', str(stats['Target_Date'])),
        ('Forecast Horizon', f"{int(stats['Forecast_Horizon_Months'])} months"),
        ('Simulation Count', f"{int(stats['Simulation_Count'])}"),
        ('Rent Price', f"${float(stats['Threshold_Price']):,.0f}"),
        ('Absolute Min', f"${float(stats['Absolute_Min']):,.0f}"),
        ('Bear (10th)', f"${float(stats['Bear_10th']):,.0f}"),
        ('Median (50th)', f"${float(stats['Median_50th']):,.0f}"),
        ('Mean', f"${float(stats['Mean']):,.0f}"),
        ('Bull (90th)', f"${float(stats['Bull_90th']):,.0f}"),
        ('Absolute Max', f"${float(stats['Absolute_Max']):,.0f}"),
    ]

    right_rows = [
        ('Count > Rent', f"{int(stats['Count_Strictly_GT_Threshold'])}"),
        ('Share > Rent', f"{float(stats['Share_Strictly_GT_Threshold']) * 100:.2f}%"),
        ('Count <= Rent', f"{int(stats['Count_At_Or_Below_Threshold'])}"),
        ('Share <= Rent', f"{float(stats['Share_At_Or_Below_Threshold']) * 100:.2f}%"),
        ('Mean Deviation', f"${float(stats['Mean_Deviation_From_Threshold']):+,.0f}"),
        ('Median Deviation', f"${float(stats['Median_Deviation_From_Threshold']):+,.0f}"),
        ('Min Deviation', f"${float(stats['Absolute_Min_Deviation_From_Threshold']):+,.0f}"),
        ('Max Deviation', f"${float(stats['Absolute_Max_Deviation_From_Threshold']):+,.0f}"),
        ('Mean Abs Deviation', f"${float(stats['Mean_Absolute_Deviation_From_Threshold']):,.0f}"),
        ('RMSE vs Rent', f"${float(stats['RMSE_From_Threshold']):,.0f}"),
    ]

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.3, wspace=0.2)

    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    _render_stats_table(ax_left, left_rows, 'Terminal Distribution Snapshot')
    _render_stats_table(ax_right, right_rows, 'Threshold Analysis')

    ax_levels = fig.add_subplot(grid[1, :])
    labels = ['Absolute Min', 'Bear 10th', 'Median 50th', 'Mean', 'Bull 90th', 'Absolute Max']
    values = [
        float(stats['Absolute_Min']),
        float(stats['Bear_10th']),
        float(stats['Median_50th']),
        float(stats['Mean']),
        float(stats['Bull_90th']),
        float(stats['Absolute_Max']),
    ]
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#313695']
    ax_levels.barh(labels, values, color=colors, alpha=0.9)
    ax_levels.axvline(
        x=float(stats['Threshold_Price']),
        color='black',
        linestyle='--',
        linewidth=2,
        label=f"Threshold (${float(stats['Threshold_Price']):,.0f})",
    )
    ax_levels.set_title('March 2050 Price Levels vs Rent', fontsize=12, fontweight='bold')
    ax_levels.set_xlabel('Price (CAD)')
    ax_levels.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax_levels.grid(True, axis='x', alpha=0.25, linestyle='--')
    ax_levels.legend(loc='lower right', framealpha=0.95)

    fig.suptitle(
        'Terminal Forecast Performance Dashboard',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    count_gt_threshold = int(terminal_stats_df.loc[0, 'Count_Strictly_GT_Threshold'])
    total_simulations = int(terminal_stats_df.loc[0, 'Simulation_Count'])
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Terminal stats saved to {output_path}")
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"March 2050 forecasts strictly > ${threshold_price:,.0f}: {count_gt_threshold}/{total_simulations}\n"
    )


def create_fan_chart(price_paths: pd.DataFrame, summary_stats_df: pd.DataFrame,
                     df_historical: pd.DataFrame, output_path: str,
                     anchor_price: float = 1090326.0) -> None:
    """Create and save the fan chart visualization with historical data from 1968 and forecast to 2050."""
    viz_start = datetime.now()
    print(f"[{viz_start.strftime('%Y-%m-%d %H:%M:%S')}] === VISUALIZATION PHASE STARTED ===")
    print(f"[{viz_start.strftime('%Y-%m-%d %H:%M:%S')}] Creating fan chart with {price_paths.shape[1]} simulation paths...")
    
    # Extract historical prices and dates
    historical_dates = pd.to_datetime(df_historical['date'])

    if 'Log_Return_MoM' not in df_historical.columns:
        raise ValueError("Historical dataset must include 'Log_Return_MoM' to rebuild prices.")

    historical_prices = _rebuild_prices_from_log_returns(df_historical['Log_Return_MoM'], anchor_price)
    historical_prices = historical_prices.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if historical_prices.isna().all():
        raise ValueError("Historical price series could not be constructed (all values are NaN)")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Historical price source: Log_Return_MoM (reconstructed)")
    
    # Get the last historical date for reference
    last_historical_date = historical_dates.iloc[-1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Historical data spans from {historical_dates.iloc[0].strftime('%Y-%m-%d')} to {last_historical_date.strftime('%Y-%m-%d')}")
    
    # Create date index for forecast period (300 months = 25 years, target 2050)
    # From 2025-03 to 2050-03 is 25 years or 300 months
    forecast_dates = [last_historical_date + relativedelta(months=i) for i in range(1, len(summary_stats_df) + 1)]
    forecast_dates = pd.to_datetime(forecast_dates)
    
    # Extend price_paths and summary_stats_df indices to match combined dates
    price_paths_with_dates = price_paths.copy()
    price_paths_with_dates.index = forecast_dates
    
    summary_stats_df_with_dates = summary_stats_df.copy()
    summary_stats_df_with_dates.index = forecast_dates
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot historical data as a baseline (solid dark blue line)
    ax.plot(historical_dates, historical_prices, 
            color='darkblue', linewidth=2.5, label='Historical (1968-2025)', zorder=10)
    
    # Plot all Monte Carlo paths as visible light-gray traces (match validation output style)
    for col in price_paths_with_dates.columns:
        ax.plot(
            forecast_dates,
            price_paths_with_dates[col],
            color='gray',
            alpha=0.3,
            linewidth=0.8,
        )
    
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
    start_market_price = 1090326.0
    
    paths_output = os.path.join(results_dir, "final_simulations.csv")
    stats_output = os.path.join(results_dir, "final_summary_stats.csv")
    plot_output = os.path.join(results_dir, "final_simulations_plot.png")
    terminal_stats_output = os.path.join(results_dir, "final_terminal_performance_stats.png")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # === PIPELINE EXECUTION ===
    
    # Stage 1: Load Data
    df = load_historical_data(data_path)
    
    # Stage 2: Initialize Simulator
    simulator = initialize_simulator(df, seed=42, start_market_price=start_market_price)
    
    # Stage 3: Train Master Model
    train_master_model(simulator, df)
    
    # Stage 4: Run Monte Carlo Simulation
    price_paths = run_monte_carlo_simulation(simulator, steps=300, iterations=100)
    
    # Stage 5: Aggregate Statistics
    summary_stats_df = aggregate_statistics(price_paths)
    
    # Stage 6: Create Fan Chart (with historical data)
    create_fan_chart(price_paths, summary_stats_df, df, plot_output, anchor_price=start_market_price)
    
    # Stage 7: Export Results
    export_results(price_paths, summary_stats_df, paths_output, stats_output)

    # Stage 8: Export terminal performance stats (March 2050 distribution)
    export_terminal_performance_stats(
        price_paths=price_paths,
        output_path=terminal_stats_output,
        target_date=TARGET_DATE,
        threshold_price=TARGET_PRICE_THRESHOLD,
    )
    
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
    print(f"  4. Terminal Performance Dashboard:  {terminal_stats_output}")
    print("\nRisk Matrix & Buy vs. Rent Analysis can now proceed with these outputs.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
