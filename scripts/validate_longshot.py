"""
Baseline Sentiment Test: Validation of MarketSimulator Default Parameters
==========================================================================

This script establishes a baseline for the stochastic parameters of the MarketSimulator
by testing its current default sentiment configuration against a 15-year hold-out set
(March 2010 - March 2025).

Methodology:
1. Split data at March 1, 2010: train_df (all rows before) and test_df (from that date onwards)
2. Extract anchor market price from last row of train_df
3. Fit MarketSimulator on train_df using default sentiment parameters
4. Run 100 Monte Carlo iterations with forecast_price()
5. Calculate the median of 100 paths
6. Compute MAPE and RMSE vs. actual historical prices
7. Visualize: 100 simulated paths (light gray), median (bold blue), actual (bold red)
8. Log baseline metrics for subsequent grid search comparison

Author: Senior Quantitative Developer
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import sys
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from market_simulator import MarketSimulator


def load_and_split_data(data_path: str, split_date: str = '2000-03-01'):
    """
    Load processed data and split into train/test at specified date.
    
    Args:
        data_path: Path to processed_data.csv
        split_date: Date string (YYYY-MM-DD) to split on. Test includes this date onwards.
    
    Returns:
        train_df, test_df: DataFrames split at split_date, with date parsed as datetime
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Parse date column as datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to ensure proper ordering
    df = df.sort_values('date').reset_index(drop=True)
    
    split_dt = pd.to_datetime(split_date)
    
    # Split: train_df = before split, test_df = split and onwards
    train_df = df[df['date'] < split_dt].copy()
    test_df = df[df['date'] >= split_dt].copy()
    
    print(f"✓ Data loaded: {len(df)} total rows")
    print(f"✓ Train set: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"✓ Test set: {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    return train_df, test_df


def get_anchor_market_price(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Extract the anchor market price from the last row of training data.
    Uses Log_Price if available, otherwise attempts to reconstruct from market_price_target_average.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (unused, kept for consistency)
    
    Returns:
        float: Anchor market price (in level, not log)
    """
    # Try Log_Price first from last row of train_df
    if 'Log_Price' in train_df.columns:
        last_log_price = train_df['Log_Price'].iloc[-1]
        if pd.notna(last_log_price):
            anchor_price = float(np.exp(last_log_price))
            print(f"✓ Anchor market price: ${anchor_price:,.2f} (from Log_Price)")
            return anchor_price
    
    # Fallback: use market_price_target_average from last row of train_df
    market_price_col = 'market_price_target_average,_detached_single_family_homes'
    if market_price_col in train_df.columns:
        last_market_price = train_df[market_price_col].iloc[-1]
        if pd.notna(last_market_price):
            print(f"✓ Anchor market price: ${last_market_price:,.2f} (from market_price_target_average)")
            return float(last_market_price)
    
    # Final fallback: use correct default price
    default_price = 235535.0
    print(f"! Using default anchor price: ${default_price:,.2f}")
    return default_price


def run_simulations(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   anchor_price: float, n_iterations: int = 25) -> pd.DataFrame:
    """
    Initialize MarketSimulator, fit on train_df, and forecast over test period.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame (used only for determining forecast length)
        anchor_price: Starting market price for simulator
        n_iterations: Number of Monte Carlo iterations
    
    Returns:
        all_paths: DataFrame with simulated price paths (one column per iteration)
    """
    print(f"\nInitializing MarketSimulator with default sentiment parameters...")
    
    # Initialize with train_df and anchor price, using default sentiment parameters
    simulator = MarketSimulator(
        df=train_df,
        seed=42,
        start_market_price=anchor_price
        # Note: Sentiment parameters use class defaults:
        # - sentiment_shock_mean = 0.001 (+0.1% monthly optimism bias)
        # - sentiment_shock_std = 0.005 (volatility of sentiment)
        # - sentiment_mean_reversion = 0.8 (mean reversion speed)
    )
    
    print("Fitting models (Tier 1: ARIMA, Tier 2-3: SARIMAX, Tier 4: XGBoost)...")
    simulator.fit(train_df)
    
    # Forecast for the length of test_df with specified iterations
    forecast_steps = len(test_df)
    print(f"\nRunning {n_iterations} Monte Carlo iterations for {forecast_steps} months...")
    all_paths = simulator.forecast_price(iterations=n_iterations, steps=forecast_steps)
    
    print(f"✓ Generated {all_paths.shape[1]} simulated paths, {all_paths.shape[0]} steps each")
    
    return all_paths


def calculate_metrics(predicted_prices: np.ndarray, actual_prices: np.ndarray) -> tuple:
    """
    Calculate Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE).
    
    Args:
        predicted_prices: Predicted price sequence (numeric array)
        actual_prices: Actual historical price sequence (numeric array)
    
    Returns:
        (mape, rmse): Tuple of metric values
    """
    # Remove NaN values
    mask = ~(np.isnan(predicted_prices) | np.isnan(actual_prices))
    pred = predicted_prices[mask]
    actual = actual_prices[mask]
    
    # MAPE: Mean Absolute Percentage Error
    # MAPE = (1/n) * Σ |actual - predicted| / |actual|
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    
    # RMSE: Root Mean Squared Error
    # RMSE = sqrt((1/n) * Σ (actual - predicted)^2)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    
    return mape, rmse


def extract_actual_prices(test_df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct actual observed prices from log returns (following validate_rolling.py method).
    
    Reconstruction formula: Price(t) = Price(t-1) * exp(Log_Return_MoM(t))
    
    Args:
        test_df: Test DataFrame containing Log_Return_MoM and lag price columns
    
    Returns:
        np.ndarray: Array of actual prices reconstructed from log returns
    """
    log_returns = test_df['Log_Return_MoM'].fillna(0).values
    
    # Get starting price from the lag_1 column (previous month's price)
    price_col_lag1 = 'market_price_target_average,_detached_single_family_homes_lag_1'
    
    if price_col_lag1 not in test_df.columns:
        raise ValueError(f"Cannot find starting price column: {price_col_lag1}")
    
    # Initialize with starting price
    if pd.notna(test_df[price_col_lag1].iloc[0]):
        current_price = float(test_df[price_col_lag1].iloc[0])
    else:
        raise ValueError("Starting price (lag_1) is NaN in first test row")
    
    # Reconstruct prices by applying log returns cumulatively
    prices = []
    for ret in log_returns:
        current_price = current_price * np.exp(ret)
        prices.append(current_price)
    
    print(f"✓ Reconstructed {len(prices)} actual prices from Log_Return_MoM column")
    return np.array(prices)


def create_visualization(all_paths: pd.DataFrame, actual_prices: np.ndarray,
                        median_path: np.ndarray, test_df: pd.DataFrame,
                        output_path: str):
    """
    Create and save visualization of simulated vs. actual prices.
    
    Args:
        all_paths: DataFrame of all simulated price paths
        actual_prices: Array of actual historical prices
        median_path: Array of median across all iterations
        test_df: Test DataFrame (for dates)
        output_path: Path to save plot PNG
    """
    print(f"\nCreating visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get dates from test_df
    dates = test_df['date'].values[:len(actual_prices)]
    
    # Plot all simulated paths in light gray with high transparency
    for col in all_paths.columns:
        ax.plot(dates, all_paths[col].values[:len(actual_prices)], 
               color='gray', alpha=0.05, linewidth=0.8)
    
    # Plot median in bold blue
    ax.plot(dates, median_path, 
           color='blue', alpha=1.0, linewidth=2.5, label='Median Forecasted Path')
    
    # Plot actual prices in bold red
    ax.plot(dates, actual_prices, 
           color='red', alpha=1.0, linewidth=2.5, label='Actual Historical Prices', linestyle='--')
    
    # Format
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Median Price ($)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Sentiment Validation: 25 Monte Carlo Simulations vs. Actual Prices\nTrain: pre-Mar2010 | Test: Mar2010-Mar2025', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Format Y-axis as currency
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("BASELINE SENTIMENT VALIDATION TEST")
    print("=" * 80)
    
    # Configuration
    data_path = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv"
    split_date = "2000-03-01"
    n_iterations = 25
    output_plot = "/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/results/baseline_validation_plot.png"
    
    # Step 1: Load and split data
    train_df, test_df = load_and_split_data(data_path, split_date)
    
    # Step 2: Get anchor price (March 2010, from first row of test set)
    anchor_price = get_anchor_market_price(train_df, test_df)
    
    # Step 3: Run simulations
    all_paths = run_simulations(train_df, test_df, anchor_price, n_iterations)
    
    # Step 4: Calculate median path
    print(f"\nCalculating median path across {n_iterations} iterations...")
    median_path = all_paths.median(axis=1).values
    print(f"✓ Median path calculated")
    
    # Step 5: Extract actual prices
    print(f"Extracting actual historical prices...")
    actual_prices = extract_actual_prices(test_df)
    
    # Align lengths (simulated path should match test_df length)
    min_len = min(len(median_path), len(actual_prices))
    median_path = median_path[:min_len]
    actual_prices = actual_prices[:min_len]
    test_df_aligned = test_df.iloc[:min_len]
    
    print(f"✓ Aligned to {min_len} data points")
    
    # Step 6: Calculate metrics
    print(f"\nCalculating baseline metrics...")
    mape, rmse = calculate_metrics(median_path, actual_prices)
    
    # Step 7: Print baseline metrics
    print(f"\n" + "=" * 80)
    print(f"BASELINE METRICS (Default Sentiment Parameters)")
    print(f"=" * 80)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"Root Mean Squared Error (RMSE):        ${rmse:,.2f}")
    print(f"Dataset: {test_df_aligned['date'].iloc[0].date()} to {test_df_aligned['date'].iloc[-1].date()}")
    print(f"Iterations: {n_iterations}")
    print(f"=" * 80 + "\n")
    
    # Step 8: Create visualization
    create_visualization(all_paths.iloc[:min_len], actual_prices, median_path, 
                        test_df_aligned, output_plot)
    
    print("✓ Baseline validation test complete!")
    
    return {
        'mape': mape,
        'rmse': rmse,
        'n_iterations': n_iterations,
        'forecast_months': min_len,
        'anchor_price': anchor_price,
        'median_path': median_path,
        'actual_prices': actual_prices
    }


if __name__ == "__main__":
    results = main()
