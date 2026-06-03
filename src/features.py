from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLEAN_MARKET_INPUTS_PATH, MARKET_FEATURES_PATH, STOCK_FEATURES_PATH
from src.data_loader import read_csv


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = _safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def _add_stock_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["close"]
    volume = group["volume"]

    group["daily_return"] = close.pct_change().fillna(0.0)
    group["log_return"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    group["volume_change"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    group["sma_10"] = close.rolling(window=10, min_periods=1).mean()
    group["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    group["sma_50"] = close.rolling(window=50, min_periods=1).mean()
    group["ema_20"] = close.ewm(span=20, adjust=False).mean()

    group["volatility_5"] = group["daily_return"].rolling(window=5, min_periods=2).std().fillna(0.0)
    group["volatility_20"] = group["daily_return"].rolling(window=20, min_periods=2).std().fillna(0.0)
    group["rsi_14"] = _rsi(close, window=14).fillna(50.0)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    group["macd"] = ema_12 - ema_26
    group["macd_signal"] = group["macd"].ewm(span=9, adjust=False).mean()
    group["macd_hist"] = group["macd"] - group["macd_signal"]

    running_high = close.cummax()
    group["drawdown"] = close / running_high - 1.0
    return group


def build_stock_features(clean_market_inputs: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "high", "low", "close", "volume", "sector", "vix"}
    missing = sorted(required - set(clean_market_inputs.columns))
    if missing:
        raise ValueError(f"Clean market inputs are missing required columns: {missing}")

    frame = clean_market_inputs.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    features = pd.concat(
        [_add_stock_features(group) for _, group in frame.groupby("ticker", sort=False)],
        ignore_index=True,
    )
    return features.sort_values(["date", "ticker"]).reset_index(drop=True)


def _sector_features(stock_features: pd.DataFrame) -> pd.DataFrame:
    sector_returns = (
        stock_features.groupby(["date", "sector"])["daily_return"]
        .mean()
        .unstack("sector")
        .sort_index()
    )
    sector_returns = sector_returns.add_prefix("sector_return_").reset_index()

    sector_columns = [column for column in sector_returns.columns if column.startswith("sector_return_")]
    sector_returns["best_sector_return"] = sector_returns[sector_columns].max(axis=1)
    sector_returns["worst_sector_return"] = sector_returns[sector_columns].min(axis=1)
    sector_returns["sector_dispersion"] = (
        sector_returns["best_sector_return"] - sector_returns["worst_sector_return"]
    )
    sector_returns["positive_sector_count"] = (sector_returns[sector_columns] > 0).sum(axis=1)
    return sector_returns


def _vix_features(vix_by_date: pd.DataFrame) -> pd.DataFrame:
    frame = vix_by_date.sort_values("date").copy()
    frame["vix_change"] = frame["vix"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    frame["vix_ma_5"] = frame["vix"].rolling(window=5, min_periods=1).mean()
    frame["vix_ma_20"] = frame["vix"].rolling(window=20, min_periods=1).mean()
    vix_mean_60 = frame["vix"].rolling(window=60, min_periods=2).mean()
    vix_std_60 = frame["vix"].rolling(window=60, min_periods=2).std()
    frame["vix_zscore_60"] = _safe_divide(frame["vix"] - vix_mean_60, vix_std_60).fillna(0.0)
    return frame


def build_market_features(stock_features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "sector",
        "daily_return",
        "volume_change",
        "volatility_20",
        "rsi_14",
        "macd_hist",
        "drawdown",
        "vix",
    }
    missing = sorted(required - set(stock_features.columns))
    if missing:
        raise ValueError(f"Stock features are missing required columns: {missing}")

    market = (
        stock_features.groupby("date")
        .agg(
            market_return_mean=("daily_return", "mean"),
            market_return_median=("daily_return", "median"),
            market_volatility_mean=("volatility_20", "mean"),
            market_rsi_mean=("rsi_14", "mean"),
            market_macd_hist_mean=("macd_hist", "mean"),
            market_volume_change_mean=("volume_change", "mean"),
            advance_ratio=("daily_return", lambda values: float((values > 0).mean())),
            decline_ratio=("daily_return", lambda values: float((values < 0).mean())),
            cross_sectional_volatility=("daily_return", "std"),
            avg_drawdown=("drawdown", "mean"),
            vix=("vix", "mean"),
        )
        .reset_index()
    )
    market["cross_sectional_volatility"] = market["cross_sectional_volatility"].fillna(0.0)

    sector = _sector_features(stock_features)
    vix = _vix_features(market[["date", "vix"]])
    market = market.drop(columns=["vix"]).merge(vix, on="date", how="left")
    market = market.merge(sector, on="date", how="left")
    market = market.sort_values("date").reset_index(drop=True)
    return market


def build_feature_datasets(
    clean_inputs_path=CLEAN_MARKET_INPUTS_PATH,
    stock_output_path=STOCK_FEATURES_PATH,
    market_output_path=MARKET_FEATURES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_inputs = read_csv(clean_inputs_path)
    stock_features = build_stock_features(clean_inputs)
    market_features = build_market_features(stock_features)

    stock_output_path.parent.mkdir(parents=True, exist_ok=True)
    market_output_path.parent.mkdir(parents=True, exist_ok=True)
    stock_features.to_csv(stock_output_path, index=False)
    market_features.to_csv(market_output_path, index=False)
    return stock_features, market_features


if __name__ == "__main__":
    stocks, market = build_feature_datasets()
    print(f"Saved {len(stocks):,} stock feature rows to {STOCK_FEATURES_PATH}")
    print(f"Saved {len(market):,} market feature rows to {MARKET_FEATURES_PATH}")
