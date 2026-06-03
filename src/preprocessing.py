from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
MERGED_COLUMNS = OHLCV_COLUMNS + ["sector", "vix"]


def _normalize_column_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _rename_first_available(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    target: str,
) -> pd.DataFrame:
    normalized_candidates = {_normalize_column_name(candidate) for candidate in candidates}
    rename_map = {
        column: target
        for column in frame.columns
        if _normalize_column_name(column) in normalized_candidates
    }
    return frame.rename(columns=rename_map)


def clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean raw S&P 500 OHLCV rows into a predictable ticker-date table."""
    cleaned = frame.copy()
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = _rename_first_available(cleaned, ["symbol", "ticker"], "ticker")
    if "close" not in cleaned.columns:
        cleaned = _rename_first_available(cleaned, ["adj_close", "adjusted_close"], "close")

    missing = [column for column in OHLCV_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing required columns: {missing}")

    cleaned = cleaned[OHLCV_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip().str.upper()

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    cleaned = cleaned[
        (cleaned["open"] > 0)
        & (cleaned["high"] > 0)
        & (cleaned["low"] > 0)
        & (cleaned["close"] > 0)
        & (cleaned["volume"] >= 0)
    ]
    cleaned = cleaned.sort_values(["date", "ticker"]).reset_index(drop=True)
    return cleaned


def clean_sector_map(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean S&P 500 company metadata to ticker-sector mapping."""
    cleaned = frame.copy()
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = _rename_first_available(cleaned, ["symbol", "ticker"], "ticker")
    cleaned = _rename_first_available(cleaned, ["gics_sector", "sector"], "sector")

    missing = [column for column in ["ticker", "sector"] if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Sector data is missing required columns: {missing}")

    cleaned = cleaned[["ticker", "sector"]].copy()
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip().str.upper()
    cleaned["sector"] = cleaned["sector"].astype(str).str.strip()
    cleaned = cleaned.dropna(subset=["ticker", "sector"])
    cleaned = cleaned.drop_duplicates(subset=["ticker"], keep="first")
    return cleaned.reset_index(drop=True)


def clean_vix(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean FRED VIXCLS rows to daily date-vix table."""
    cleaned = frame.copy()
    cleaned.columns = [_normalize_column_name(column) for column in cleaned.columns]
    cleaned = _rename_first_available(cleaned, ["date", "observation_date"], "date")
    cleaned = _rename_first_available(cleaned, ["vixcls", "vix", "value"], "vix")

    missing = [column for column in ["date", "vix"] if column not in cleaned.columns]
    if missing:
        raise ValueError(f"VIX data is missing required columns: {missing}")

    cleaned = cleaned[["date", "vix"]].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["vix"] = pd.to_numeric(cleaned["vix"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "vix"])
    cleaned = cleaned.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return cleaned.reset_index(drop=True)


def merge_market_inputs(
    ohlcv: pd.DataFrame,
    sectors: pd.DataFrame,
    vix: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cleaned OHLCV, sector, and VIX inputs on ticker/date."""
    trading_dates = pd.DataFrame({"date": sorted(ohlcv["date"].drop_duplicates())})
    vix_on_trading_dates = pd.merge_asof(
        trading_dates.sort_values("date"),
        vix.sort_values("date"),
        on="date",
        direction="backward",
    )

    merged = ohlcv.merge(sectors, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")
    merged = merged.merge(vix_on_trading_dates, on="date", how="left")
    merged = merged.dropna(subset=["vix"])
    merged = merged[MERGED_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    return merged
