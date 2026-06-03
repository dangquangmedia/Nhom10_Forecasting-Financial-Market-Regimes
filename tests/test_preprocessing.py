import pandas as pd

from src.preprocessing import clean_ohlcv, clean_sector_map, clean_vix, merge_market_inputs


def test_clean_ohlcv_standardizes_columns_and_filters_invalid_rows():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03", "2020-01-02"],
            "Symbol": [" aapl ", "AAPL", "MSFT"],
            "Open": [100.0, 101.0, 50.0],
            "High": [110.0, 111.0, 55.0],
            "Low": [99.0, 100.0, 49.0],
            "Close": [105.0, -1.0, 52.0],
            "Volume": [1_000_000, 1_200_000, 800_000],
        }
    )

    cleaned = clean_ohlcv(raw)

    assert list(cleaned.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert cleaned["ticker"].tolist() == ["AAPL", "MSFT"]
    assert cleaned["close"].tolist() == [105.0, 52.0]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_clean_ohlcv_keeps_close_when_adjusted_close_also_exists():
    raw = pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "Date": ["2020-01-02"],
            "Open": [100.0],
            "High": [110.0],
            "Low": [99.0],
            "Close": [105.0],
            "Adj Close": [104.5],
            "Volume": [1_000_000],
        }
    )

    cleaned = clean_ohlcv(raw)

    assert cleaned.columns.tolist() == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert cleaned["close"].tolist() == [105.0]


def test_merge_market_inputs_adds_sector_and_forward_fills_vix_on_trading_dates():
    ohlcv = clean_ohlcv(
        pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
                "ticker": ["AAPL", "AAPL", "MSFT"],
                "open": [100.0, 101.0, 50.0],
                "high": [110.0, 111.0, 55.0],
                "low": [99.0, 100.0, 49.0],
                "close": [105.0, 106.0, 52.0],
                "volume": [1_000_000, 1_200_000, 800_000],
            }
        )
    )
    sectors = clean_sector_map(
        pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Sector": ["Information Technology", "Information Technology"],
            }
        )
    )
    vix = clean_vix(
        pd.DataFrame(
            {
                "DATE": ["2020-01-02", "2020-01-06"],
                "VIXCLS": [12.5, 14.0],
            }
        )
    )

    merged = merge_market_inputs(ohlcv, sectors, vix)

    assert merged["sector"].tolist() == [
        "Information Technology",
        "Information Technology",
        "Information Technology",
    ]
    assert merged["vix"].tolist() == [12.5, 12.5, 14.0]
    assert merged.columns.tolist() == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sector",
        "vix",
    ]
