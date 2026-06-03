from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_MARKET_INPUTS_PATH,
    RAW_OHLCV_PATH,
    RAW_SECTORS_PATH,
    RAW_VIX_PATH,
)
from src.preprocessing import clean_ohlcv, clean_sector_map, clean_vix, merge_market_inputs


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def build_clean_market_inputs(
    ohlcv_path: Path = RAW_OHLCV_PATH,
    sectors_path: Path = RAW_SECTORS_PATH,
    vix_path: Path = RAW_VIX_PATH,
    output_path: Path = CLEAN_MARKET_INPUTS_PATH,
) -> pd.DataFrame:
    ohlcv = clean_ohlcv(read_csv(ohlcv_path))
    sectors = clean_sector_map(read_csv(sectors_path))
    vix = clean_vix(read_csv(vix_path))
    merged = merge_market_inputs(ohlcv, sectors, vix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    result = build_clean_market_inputs()
    print(f"Saved {len(result):,} cleaned rows to {CLEAN_MARKET_INPUTS_PATH}")

