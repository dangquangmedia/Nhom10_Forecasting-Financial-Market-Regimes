from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate the maximum drawdown from an equity curve series."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def run_regime_backtest(
    predictions_df: pd.DataFrame,
    final_dataset_df: pd.DataFrame,
    initial_capital: float = 10000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest comparing Regime-Switching Strategy vs Buy-and-Hold S&P 500.
    
    Strategy:
    - If yesterday's prediction for today was "Bull" or "Recovery" -> Invest 100% S&P 500.
    - Else (Bear, High Volatility, Sideways) -> Cash, earning the daily Fed Funds Rate.
    """
    # Clone and prepare dates
    preds = predictions_df.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    
    market = final_dataset_df[["date", "market_return_mean", "fed_funds_rate"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    
    # Merge prediction signals with market actual returns
    merged = pd.merge(preds, market, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    
    # Shift predictions by 1 day: yesterday's prediction determines today's allocation
    merged["prev_y_pred"] = merged["y_pred"].shift(1)
    
    # Daily risk-free rate from Fed Funds Rate (DFF is annualized percent, e.g. 5.33)
    merged["rf_daily"] = (merged["fed_funds_rate"] / 100.0) / 252.0
    
    # Strategy Return
    # If Yesterday's prediction was Bull or Recovery, get S&P 500 return today. Else, get Fed Funds daily return.
    is_risk_on = merged["prev_y_pred"].isin(["Bull", "Recovery"])
    merged["strat_return"] = np.where(
        is_risk_on,
        merged["market_return_mean"],
        merged["rf_daily"]
    )
    
    # Buy & Hold Return is simply the market return
    merged["bh_return"] = merged["market_return_mean"]
    
    # Drop the first row since it has no yesterday's prediction (prev_y_pred is NaN)
    merged = merged.dropna(subset=["prev_y_pred"]).reset_index(drop=True)
    
    # Calculate Equity Curves starting at initial_capital
    merged["strat_equity"] = initial_capital * (1.0 + merged["strat_return"]).cumprod()
    merged["bh_equity"] = initial_capital * (1.0 + merged["bh_return"]).cumprod()
    
    # Calculate Performance Metrics
    metrics_rows = []
    
    for name, return_col, equity_col in [
        ("Regime-Switching Strategy", "strat_return", "strat_equity"),
        ("Buy-and-Hold S&P 500", "bh_return", "bh_equity")
    ]:
        returns = merged[return_col]
        equity = merged[equity_col]
        
        cum_return = (equity.iloc[-1] / initial_capital) - 1.0
        
        # Annualization factor (assume 252 trading days per year)
        n_days = len(merged)
        years = n_days / 252.0
        
        ann_return = (equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        ann_vol = returns.std() * np.sqrt(252)
        
        # Average annualized risk-free rate in the test period
        avg_rf_ann = (merged["fed_funds_rate"].mean() / 100.0)
        
        # Sharpe Ratio
        sharpe = (ann_return - avg_rf_ann) / ann_vol if ann_vol > 0 else 0.0
        
        # Max Drawdown
        max_dd = calculate_max_drawdown(equity)
        
        metrics_rows.append({
            "Strategy": name,
            "Cumulative Return": cum_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Final Value": equity.iloc[-1]
        })
        
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Keep only important columns for UI charting in equity curve output
    equity_curves = merged[["date", "prev_y_pred", "market_return_mean", "rf_daily", "strat_equity", "bh_equity"]].copy()
    
    return equity_curves, metrics_df
