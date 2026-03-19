"""Build a combined comparison chart for simulation bands and deterministic baselines."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rent_model


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMULATIONS_PATH = os.path.join(ROOT_DIR, "results", "final_simulations.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed_data.csv")
OUTPUT_PLOT_PATH = os.path.join(ROOT_DIR, "results", "all_model_comp_plot.png")

START_PRICE = 1090320.0
HISTORICAL_START = pd.Timestamp("1995-03-01")
HIST_LOG_RETURN_START = HISTORICAL_START


def _format_currency(value: float) -> str:
    """Format currency in a compact, legend-friendly style."""
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:.0f}"


def load_simulation_percentiles(simulations_path: str) -> pd.DataFrame:
    """Load simulation paths and derive bear/base/bull lines."""
    simulations = pd.read_csv(simulations_path, index_col=0, parse_dates=True)
    simulations = simulations.apply(pd.to_numeric, errors="coerce")
    simulations = simulations.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if simulations.empty:
        raise ValueError("No valid simulation data found in final_simulations.csv")

    summary = pd.DataFrame(index=simulations.index)
    summary["Bear_10th"] = simulations.quantile(0.10, axis=1)
    summary["Base_50th"] = simulations.quantile(0.50, axis=1)
    summary["Bull_90th"] = simulations.quantile(0.90, axis=1)
    return summary


def trace_portfolio_path(periods: int) -> pd.Series:
    """Replicate rent_model.py portfolio path for each month."""
    rent_increase = float(rent_model.rent_increase)
    inflation = float(rent_model.inflation)
    average_rent = float(rent_model.average_rent)
    mortgage_payment_monthly = float(rent_model.mortgage_payment_monthly)
    own_extra_costs = float(rent_model.own_extra_costs)
    rent_extra_costs = float(rent_model.rent_extra_costs)
    monthly_ror = float(rent_model.monthly_RoR)
    portfolio = float(rent_model.initial_down_payment)

    path_values: list[float] = []
    for month in range(periods):
        portfolio = portfolio * (1 + monthly_ror)

        if month % 12 == 11:
            average_rent = average_rent * (1 + rent_increase)

        rent_extra_costs = rent_extra_costs * (1 + inflation / 12)
        own_extra_costs = own_extra_costs * (1 + inflation / 12)

        total_rent_cost = average_rent + rent_extra_costs
        total_own_cost = mortgage_payment_monthly + own_extra_costs
        diff = total_own_cost - total_rent_cost

        portfolio = portfolio + diff
        path_values.append(portfolio)

    return pd.Series(path_values)


def build_avg_log_return_path(
    periods: int,
    start_price: float,
    processed_data_path: str,
) -> tuple[pd.Series, float]:
    """Build an exponential path using average monthly log return from HIST_LOG_RETURN_START."""
    history = pd.read_csv(processed_data_path, usecols=["date", "Log_Return_MoM"])
    history["date"] = pd.to_datetime(history["date"])

    history = history[history["date"] >= HIST_LOG_RETURN_START]
    avg_log_return = pd.to_numeric(history["Log_Return_MoM"], errors="coerce").dropna().mean()

    if pd.isna(avg_log_return):
        raise ValueError("Could not compute average log return from processed_data.csv")

    t = np.arange(periods, dtype=float)
    path = start_price * np.exp(avg_log_return * t)
    return pd.Series(path), float(avg_log_return)


def _rebuild_prices_from_log_returns(log_returns: pd.Series, terminal_price: float) -> pd.Series:
    """Rebuild level prices from monthly log returns using a terminal anchor."""
    returns = pd.to_numeric(log_returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(returns) == 0:
        return pd.Series(dtype=float)

    log_prices = np.empty(len(returns), dtype=float)
    log_prices[-1] = float(np.log(terminal_price))
    for i in range(len(returns) - 2, -1, -1):
        log_prices[i] = log_prices[i + 1] - float(returns[i + 1])

    return pd.Series(np.exp(log_prices))


def build_historical_series(
    processed_data_path: str,
    forecast_start: pd.Timestamp,
    anchor_price: float,
) -> pd.Series:
    """Build historical price path (pre-forecast) from Log_Return_MoM."""
    history = pd.read_csv(processed_data_path, usecols=["date", "Log_Return_MoM"])
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date")

    prices = _rebuild_prices_from_log_returns(history["Log_Return_MoM"], anchor_price)
    historical = pd.Series(prices.to_numpy(), index=history["date"])
    historical = historical.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    historical = historical[historical.index >= HISTORICAL_START]
    historical = historical[historical.index < forecast_start]
    if historical.empty:
        raise ValueError("Historical series is empty after filtering for pre-forecast dates")

    return historical


def create_plot(
    summary: pd.DataFrame,
    historical_prices: pd.Series,
    portfolio_path: pd.Series,
    avg_log_return_path: pd.Series,
    avg_log_return: float,
    output_plot_path: str,
) -> None:
    """Create and save a single chart with all requested lines."""
    fig, ax = plt.subplots(figsize=(11, 9))

    ax.plot(
        historical_prices.index,
        historical_prices.values,
        color="darkblue",
        linewidth=2.5,
        label=(
            f"Historical ({historical_prices.index.min().year}-{historical_prices.index.max().year}), "
            f"Last = {_format_currency(historical_prices.iloc[-1])}"
        ),
        zorder=10,
    )

    ax.plot(
        summary.index,
        summary["Bear_10th"],
        color="red",
        linewidth=2.5,
        label=f"Bear (10th %ile), FV = {_format_currency(summary['Bear_10th'].iloc[-1])}",
    )
    ax.plot(
        summary.index,
        summary["Base_50th"],
        color="black",
        linewidth=2.5,
        label=f"Base (50th %ile), FV = {_format_currency(summary['Base_50th'].iloc[-1])}",
    )
    ax.plot(
        summary.index,
        summary["Bull_90th"],
        color="green",
        linewidth=2.5,
        label=f"Bull (90th %ile), FV = {_format_currency(summary['Bull_90th'].iloc[-1])}",
    )

    ax.plot(
        summary.index,
        portfolio_path,
        color="darkorange",
        linewidth=2.5,
        label=f"Rent Model Portfolio Path, FV = {_format_currency(portfolio_path.iloc[-1])}",
    )

    ax.plot(
        summary.index,
        avg_log_return_path,
        color="navy",
        linewidth=2.5,
        linestyle="--",
        label=(
            f"Average Historical Return ({avg_log_return:.3%}/mo), "
            f"FV = {_format_currency(avg_log_return_path.iloc[-1])}"
        ),
    )

    forecast_start_marker = historical_prices.index.max()
    ax.axvline(x=forecast_start_marker, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        forecast_start_marker,
        ax.get_ylim()[1] * 0.95,
        "Forecast Start",
        fontsize=10,
        color="gray",
        alpha=0.8,
        ha="right",
    )

    ax.set_title(
        "Toronto Housing Comparison: Bear/Base/Bull vs Rent Model & Avg Historical Return",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value ($CAD)", fontsize=12, fontweight="bold")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value/1e6:.1f}M"))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    summary = load_simulation_percentiles(SIMULATIONS_PATH)
    periods = len(summary)
    forecast_start = summary.index.min()

    historical_prices = build_historical_series(
        processed_data_path=PROCESSED_DATA_PATH,
        forecast_start=forecast_start,
        anchor_price=START_PRICE,
    )

    portfolio_path = trace_portfolio_path(periods)
    portfolio_path.index = summary.index

    avg_log_return_path, avg_log_return = build_avg_log_return_path(
        periods=periods,
        start_price=START_PRICE,
        processed_data_path=PROCESSED_DATA_PATH,
    )
    avg_log_return_path.index = summary.index

    create_plot(
        summary=summary,
        historical_prices=historical_prices,
        portfolio_path=portfolio_path,
        avg_log_return_path=avg_log_return_path,
        avg_log_return=avg_log_return,
        output_plot_path=OUTPUT_PLOT_PATH,
    )

    print(f"Saved comparison chart to: {OUTPUT_PLOT_PATH}")
    print(f"Plotted periods: {periods} ({summary.index.min().date()} to {summary.index.max().date()})")
    print(
        f"Historical span: {historical_prices.index.min().date()} to {historical_prices.index.max().date()}"
    )
    print(f"Avg monthly log return since {HIST_LOG_RETURN_START.date()}: {avg_log_return:.6f}")


if __name__ == "__main__":
    main()

