from __future__ import annotations

import datetime
import shutil
from pathlib import Path
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from src.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    RAW_OHLCV_PATH,
    RAW_SECTORS_PATH,
    RAW_VIX_PATH,
)
from src.macro_fetcher import fetch_macro_data


def clean_ticker_for_yf(ticker: str) -> str:
    """Convert ticker symbol format for Yahoo Finance (e.g., BRK.B -> BRK-B)."""
    return ticker.strip().replace(".", "-")


def fetch_all_live_data(
    start_date: str = "2021-01-01",
    end_date: str | None = None,
) -> None:
    """Fetch S&P 500 prices, VIX and macro features, and save to raw data directory."""
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Sao chép thông tin phân ngành (sp500_companies.csv)
    local_sp500_sectors = DATA_DIR / "SP500.csv"
    if local_sp500_sectors.exists():
        shutil.copy(local_sp500_sectors, RAW_SECTORS_PATH)
        print(f"[Cập nhật] Đã đồng bộ phân ngành từ {local_sp500_sectors} -> {RAW_SECTORS_PATH}")
    else:
        raise FileNotFoundError(f"Không tìm thấy file danh mục S&P 500 tại {local_sp500_sectors}")

    # Đọc danh sách các mã cổ phiếu gốc
    sectors_df = pd.read_csv(RAW_SECTORS_PATH)
    original_tickers = sectors_df["Symbol"].dropna().unique().tolist()
    
    # 2. Tải dữ liệu S&P 500 OHLCV qua yfinance
    yf_tickers = [clean_ticker_for_yf(t) for t in original_tickers]
    yf_to_orig = {clean_ticker_for_yf(t): t for t in original_tickers}
    
    print(f"[Cập nhật] Bắt đầu tải dữ liệu giá cho {len(yf_tickers)} mã cổ phiếu từ yfinance...")
    try:
        df = yf.download(yf_tickers, start=start_date, end=end_date, threads=True)
        if df.empty or not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Dữ liệu tải về trống hoặc không đúng cấu trúc MultiIndex.")
            
        print("[Cập nhật] Đang định dạng lại dữ liệu sang dạng cột dọc (Long format)...")
        # stack level=1 (Ticker) để biến đổi các mã thành hàng
        df_stacked = df.stack(level=1, future_stack=True).reset_index()
        
        # Đổi tên cột về dạng chữ thường và thay khoảng trắng bằng dấu gạch dưới
        df_stacked.columns = [col.lower().replace(" ", "_") for col in df_stacked.columns]
        
        # Ánh xạ ngược lại mã gốc của S&P 500 (BRK-B -> BRK.B)
        df_stacked["ticker"] = df_stacked["ticker"].map(yf_to_orig)
        
        # Lưu vào raw
        df_stacked.to_csv(RAW_OHLCV_PATH, index=False)
        print(f"[Thành công] Đã tải và lưu {len(df_stacked):,} hàng dữ liệu giá S&P 500 vào: {RAW_OHLCV_PATH}")
        
    except Exception as e:
        print(f"[Cảnh báo] Lỗi tải dữ liệu yfinance: {e}")
        fallback_path = DATA_DIR / "SP500_Historical_Data.csv"
        if fallback_path.exists():
            print(f"[Dự phòng] Đang sử dụng tệp tin lịch sử có sẵn: {fallback_path} -> {RAW_OHLCV_PATH}")
            shutil.copy(fallback_path, RAW_OHLCV_PATH)
        else:
            raise FileNotFoundError(f"Không thể tải dữ liệu yfinance và không tìm thấy file dự phòng tại {fallback_path}")

    # 3. Tải chỉ số VIX từ FRED
    print("[Cập nhật] Bắt đầu tải dữ liệu chỉ số VIX từ FRED...")
    try:
        vix_df = web.DataReader("VIXCLS", "fred", start_date, end_date)
        vix_df = vix_df.reset_index()
        vix_df.to_csv(RAW_VIX_PATH, index=False)
        print(f"[Thành công] Đã tải và lưu {len(vix_df):,} hàng chỉ số VIX vào: {RAW_VIX_PATH}")
    except Exception as e:
        print(f"[Cảnh báo] Lỗi tải dữ liệu VIX từ FRED: {e}")
        fallback_vix = DATA_DIR / "VIXCLS.csv"
        if fallback_vix.exists():
            print(f"[Dự phòng] Đang sử dụng tệp tin VIXCLS có sẵn: {fallback_vix} -> {RAW_VIX_PATH}")
            shutil.copy(fallback_vix, RAW_VIX_PATH)
        else:
            raise FileNotFoundError(f"Không thể tải dữ liệu VIX và không tìm thấy file dự phòng tại {fallback_vix}")

    # 4. Tải dữ liệu vĩ mô FRED
    print("[Cập nhật] Bắt đầu đồng bộ đặc trưng vĩ mô từ FRED...")
    fetch_macro_data(start_date=start_date, end_date=end_date)
    print("[Hoàn tất] Đồng bộ toàn bộ dữ liệu thô thành công.")


if __name__ == "__main__":
    fetch_all_live_data()
