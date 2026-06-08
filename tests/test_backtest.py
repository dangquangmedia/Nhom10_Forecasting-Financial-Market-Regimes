import numpy as np
import pandas as pd
import pytest

from src.backtest import calculate_max_drawdown, run_regime_backtest


def test_calculate_max_drawdown_flat_equity():
    equity = pd.Series([10000.0, 10000.0, 10000.0])
    assert calculate_max_drawdown(equity) == 0.0


def test_calculate_max_drawdown_realistic_loss():
    # Peak at 12000, drop to 9000, max drawdown = (9000 - 12000) / 12000 = -0.25
    equity = pd.Series([10000.0, 12000.0, 9000.0, 11000.0])
    assert calculate_max_drawdown(equity) == -0.25


def test_run_regime_backtest_computes_expected_returns_and_equity():
    # Mock predictions
    dates = pd.date_range("2025-06-01", periods=5, freq="B")
    preds_data = {
        "date": dates.strftime("%Y-%m-%d"),
        "y_true": ["Bull", "Bear", "Bull", "Bear", "Bull"],
        # Prediction for t+1 made at t
        "y_pred": ["Bull", "Bear", "Bull", "Bear", "Bull"],
        "confidence": [0.8, 0.9, 0.7, 0.8, 0.9]
    }
    preds_df = pd.DataFrame(preds_data)
    
    # Mock final dataset
    dataset_data = {
        "date": dates,
        "market_return_mean": [0.01, -0.02, 0.03, -0.01, 0.02],
        # DFF is annualized percentage
        "fed_funds_rate": [5.04, 5.04, 5.04, 5.04, 5.04]
    }
    dataset_df = pd.DataFrame(dataset_data)
    
    # Run backtest
    equity_curves, metrics_df = run_regime_backtest(preds_df, dataset_df, initial_capital=1000.0)
    
    # Assert return shapes
    # 5 periods, drop first because of shift(1) -> 4 output rows
    assert len(equity_curves) == 4
    assert len(metrics_df) == 2
    
    # Check that it contains the expected columns
    assert "strat_equity" in equity_curves.columns
    assert "bh_equity" in equity_curves.columns
    
    # Verify alignment:
    # Row 0 of output corresponds to date index 1 (the second date: 2025-06-03 or equivalent)
    # The active prediction for today was yesterday's y_pred (index 0 which is "Bull")
    # Since yesterday's prediction was "Bull" (Risk-on), today's strat_return should equal today's bh_return (market_return_mean index 1 = -0.02)
    # Row 1 of output corresponds to date index 2. Yesterday's y_pred (index 1) was "Bear" (Risk-off).
    # Today's strat_return should be risk-free rate = (5.04 / 100) / 252 = 0.0002. Today's bh_return should be 0.03.
    
    # Daily RF rate check
    expected_rf_daily = (5.04 / 100.0) / 252.0
    
    # Check that strat_equity at step 1 (second date) has S&P return:
    # Vốn ban đầu 1000 * (1 - 0.02) = 980
    assert equity_curves["strat_equity"].iloc[0] == pytest.approx(1000.0 * (1.0 - 0.02))
    assert equity_curves["bh_equity"].iloc[0] == pytest.approx(1000.0 * (1.0 - 0.02))
    
    # Check that strat_equity at step 2 (third date) has RF return:
    # 980 * (1 + expected_rf_daily)
    assert equity_curves["strat_equity"].iloc[1] == pytest.approx(980.0 * (1.0 + expected_rf_daily))
    # Bh equity should have market return: 980 * (1 + 0.03) = 1009.4
    assert equity_curves["bh_equity"].iloc[1] == pytest.approx(980.0 * (1.0 + 0.03))
    
    # Assert metrics structure
    assert "Strategy" in metrics_df.columns
    assert "Cumulative Return" in metrics_df.columns
    assert "Sharpe Ratio" in metrics_df.columns
    assert "Max Drawdown" in metrics_df.columns
