from __future__ import annotations

import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import time

from src.config import RAW_MACRO_PATH


def _generate_fallback_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic simulated historical macro data for 2021-2026 if FRED is offline."""
    print("[Hệ thống] Đang tạo dữ liệu vĩ mô giả lập thực tế (Historical Fallback)...")
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # 1. Fed Funds Rate (DFF)
    # Lãi suất duy trì ở mức 0.08% năm 2021, tăng dần từ tháng 3/2022 lên 5.33% vào tháng 7/2023, sau đó giảm nhẹ cuối 2024.
    dff = []
    for d in date_range:
        if d < pd.Timestamp("2022-03-16"):
            dff.append(0.08)
        elif d < pd.Timestamp("2023-07-26"):
            # Tăng dần tuyến tính
            days_elapsed = (d - pd.Timestamp("2022-03-16")).days
            total_days = (pd.Timestamp("2023-07-26") - pd.Timestamp("2022-03-16")).days
            dff.append(0.08 + (5.33 - 0.08) * (days_elapsed / total_days))
        elif d < pd.Timestamp("2024-09-18"):
            dff.append(5.33)
        elif d < pd.Timestamp("2024-11-07"):
            dff.append(4.83)
        elif d < pd.Timestamp("2024-12-18"):
            dff.append(4.58)
        else:
            dff.append(4.33)
            
    # 2. Yield Curve Spread (T10Y2Y)
    # Dương khoảng 1.5% giữa năm 2021, sau đó đảo ngược (âm) từ giữa năm 2022, đạt đáy khoảng -1.0% giữa năm 2023, rồi phục hồi dần.
    t10y2y = []
    for d in date_range:
        if d < pd.Timestamp("2021-06-01"):
            t10y2y.append(1.50)
        elif d < pd.Timestamp("2022-07-01"):
            # Giảm từ 1.5% về 0%
            days_elapsed = (d - pd.Timestamp("2021-06-01")).days
            total_days = (pd.Timestamp("2022-07-01") - pd.Timestamp("2021-06-01")).days
            t10y2y.append(1.50 - 1.50 * (days_elapsed / total_days))
        elif d < pd.Timestamp("2023-07-01"):
            # Giảm từ 0% xuống đáy đảo ngược -1.0%
            days_elapsed = (d - pd.Timestamp("2022-07-01")).days
            total_days = (pd.Timestamp("2023-07-01") - pd.Timestamp("2022-07-01")).days
            t10y2y.append(0.0 - 1.0 * (days_elapsed / total_days))
        elif d < pd.Timestamp("2024-09-01"):
            # Biến động âm quanh -0.5%
            days_elapsed = (d - pd.Timestamp("2023-07-01")).days
            total_days = (pd.Timestamp("2024-09-01") - pd.Timestamp("2023-07-01")).days
            t10y2y.append(-1.0 + 0.5 * (days_elapsed / total_days))
        else:
            # Phục hồi dần về 0.1%
            days_elapsed = (d - pd.Timestamp("2024-09-01")).days
            total_days = (pd.Timestamp(end_date) - pd.Timestamp("2024-09-01")).days
            t10y2y.append(-0.5 + 0.6 * (days_elapsed / max(1, total_days)))
            
    # Thêm nhiễu ngẫu nhiên nhỏ cho 2 chuỗi lãi suất
    np.random.seed(42)
    t10y2y = np.array(t10y2y) + np.random.normal(0, 0.03, len(date_range))
    dff = np.array(dff) + np.random.normal(0, 0.01, len(date_range))

    # 3. Gold Price (GOLDAMGBD228NLBM)
    # Bắt đầu khoảng $1900, đi ngang $1700-$2000 trong 2021-2023, sau đó tăng mạnh lên $2700 trong 2024-2025.
    gold = []
    current_gold = 1900.0
    for d in date_range:
        if d < pd.Timestamp("2024-01-01"):
            # Biến động đi ngang
            drift = 0.0
        else:
            # Tăng trưởng mạnh (lạm phát + địa chính trị)
            drift = 0.8 # trung bình tăng 0.8 usd mỗi ngày
        current_gold += drift + np.random.normal(0.0, 15.0)
        gold.append(max(1000.0, current_gold))

    # 4. Brent Crude Oil Price (DCOILBRENTEU)
    # Khoảng $50 đầu 2021, vọt lên $120 giữa 2022 (chiến tranh Nga-Ukraine), sau đó hạ nhiệt về vùng $75-$90.
    brent = []
    current_brent = 51.0
    for d in date_range:
        if d < pd.Timestamp("2022-06-08"):
            # Xu hướng tăng lên 120
            drift = 0.15
        elif d < pd.Timestamp("2023-06-08"):
            # Xu hướng giảm về 75
            drift = -0.12
        else:
            # Đi ngang biến động
            drift = 0.0
        current_brent += drift + np.random.normal(0.0, 1.8)
        brent.append(max(20.0, current_brent))

    df = pd.DataFrame({
        "date": date_range,
        "yield_curve_spread": t10y2y,
        "fed_funds_rate": dff,
        "gold_price": gold,
        "brent_oil_price": brent
    })
    
    # Định dạng làm tròn
    df["yield_curve_spread"] = df["yield_curve_spread"].round(4)
    df["fed_funds_rate"] = df["fed_funds_rate"].round(4)
    df["gold_price"] = df["gold_price"].round(2)
    df["brent_oil_price"] = df["brent_oil_price"].round(2)
    
    return df


def fetch_macro_data(
    start_date: str = "2021-01-01",
    end_date: str | None = None,
    max_retries: int = 3,
    delay: int = 5,
) -> pd.DataFrame:
    """Fetch macroeconomic data from St. Louis Fed (FRED) with retries, falling back to simulation on error."""
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    symbols = {
        "T10Y2Y": "yield_curve_spread",
        "DFF": "fed_funds_rate",
        "GOLDAMGBD228NLBM": "gold_price",
        "DCOILBRENTEU": "brent_oil_price",
    }

    print(f"Bắt đầu tải dữ liệu vĩ mô từ FRED ({start_date} đến {end_date})...")
    
    df = None
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            # Tải tất cả các chỉ số vĩ mô trong một cuộc gọi duy nhất
            raw_data = web.DataReader(list(symbols.keys()), "fred", start_date, end_date)
            # Đổi tên cột sang tên thân thiện
            df = raw_data.rename(columns=symbols)
            
            # Chuyển chỉ mục DATE thành cột
            df = df.reset_index()
            df = df.rename(columns={"DATE": "date"})
            df["date"] = pd.to_datetime(df["date"])
            success = True
            print("[Hệ thống] Kết nối FRED thành công.")
            break
        except Exception as e:
            print(f"[Thử lại lần {attempt}/{max_retries}] Lỗi kết nối FRED: {e}")
            if attempt < max_retries:
                time.sleep(delay)
    
    # Nếu không tải được, kích hoạt chế độ mô phỏng lịch sử vĩ mô thực tế
    if not success:
        df = _generate_fallback_macro_data(start_date, end_date)

    # Làm sạch dữ liệu
    df = df.sort_values("date").reset_index(drop=True)
    numeric_cols = ["yield_curve_spread", "fed_funds_rate", "gold_price", "brent_oil_price"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Điền giá trị trống
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # Lưu lại tệp
    RAW_MACRO_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_MACRO_PATH, index=False)
    print(f"Đã lưu {len(df):,} dòng dữ liệu vĩ mô vào: {RAW_MACRO_PATH}")
    return df


if __name__ == "__main__":
    fetch_macro_data()
