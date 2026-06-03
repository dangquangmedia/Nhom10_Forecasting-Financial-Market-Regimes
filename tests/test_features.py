import numpy as np
import pandas as pd

from src.features import build_market_features, build_stock_features


def _sample_clean_market_inputs() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    rows = []
    tickers = [
        ("AAPL", "Information Technology", 100.0, 1000),
        ("MSFT", "Information Technology", 90.0, 900),
        ("JPM", "Financials", 80.0, 800),
    ]
    for day_index, date in enumerate(dates):
        for ticker, sector, base_price, base_volume in tickers:
            drift = day_index * 0.5
            sector_bias = 2.0 if sector == "Information Technology" else -0.1 * day_index
            close = base_price + drift + sector_bias
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.4,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": base_volume + day_index * 10,
                    "sector": sector,
                    "vix": 12.0 + day_index * 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_build_stock_features_adds_returns_indicators_and_drawdown():
    raw = _sample_clean_market_inputs()

    features = build_stock_features(raw)

    expected_columns = {
        "daily_return",
        "log_return",
        "volume_change",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "volatility_5",
        "volatility_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "drawdown",
    }
    assert expected_columns.issubset(features.columns)
    aapl = features[features["ticker"] == "AAPL"].sort_values("date")
    assert aapl["daily_return"].iloc[0] == 0.0
    assert np.isclose(
        aapl["daily_return"].iloc[1],
        aapl["close"].iloc[1] / aapl["close"].iloc[0] - 1.0,
    )
    assert aapl["drawdown"].le(0).all()


def test_build_market_features_aggregates_breadth_sector_dispersion_and_vix():
    stock_features = build_stock_features(_sample_clean_market_inputs())

    market = build_market_features(stock_features)

    expected_columns = {
        "date",
        "market_return_mean",
        "market_return_median",
        "market_volatility_mean",
        "market_rsi_mean",
        "market_macd_hist_mean",
        "market_volume_change_mean",
        "advance_ratio",
        "decline_ratio",
        "cross_sectional_volatility",
        "avg_drawdown",
        "best_sector_return",
        "worst_sector_return",
        "sector_dispersion",
        "positive_sector_count",
        "vix",
        "vix_change",
        "vix_ma_5",
        "vix_ma_20",
        "vix_zscore_60",
    }
    assert expected_columns.issubset(market.columns)
    assert market["date"].is_monotonic_increasing
    assert market["advance_ratio"].between(0, 1).all()
    assert market["decline_ratio"].between(0, 1).all()
    assert (market["sector_dispersion"] >= 0).all()
    assert market["positive_sector_count"].ge(0).all()
