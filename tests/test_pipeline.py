from __future__ import annotations

from unittest.mock import MagicMock
import pandas as pd
import pytest

import src.data_fetcher
import src.pipeline


def test_run_full_pipeline_calls_all_stages_sequentially(monkeypatch):
    """Verify that run_full_pipeline calls all six stages of the pipeline in order."""
    mock_fetch = MagicMock()
    mock_loader = MagicMock()
    mock_features = MagicMock()
    mock_labeling = MagicMock()
    mock_modeling = MagicMock()
    mock_eval = MagicMock()
    
    monkeypatch.setattr(src.pipeline, "fetch_all_live_data", mock_fetch)
    monkeypatch.setattr(src.pipeline, "build_clean_market_inputs", mock_loader)
    monkeypatch.setattr(src.pipeline, "build_feature_datasets", mock_features)
    monkeypatch.setattr(src.pipeline, "build_labeled_dataset", mock_labeling)
    monkeypatch.setattr(src.pipeline, "train_all_models", mock_modeling)
    monkeypatch.setattr(src.pipeline, "export_xgboost_evaluation_artifacts", mock_eval)
    
    # Chạy pipeline
    status = src.pipeline.run_full_pipeline()
    
    # Xác minh các cuộc gọi
    mock_fetch.assert_called_once()
    mock_loader.assert_called_once()
    mock_features.assert_called_once()
    mock_labeling.assert_called_once_with(method="hmm")
    mock_modeling.assert_called_once()
    mock_eval.assert_called_once()
    
    # Xác minh cấu trúc kết quả trả về
    assert status["data_fetcher"] == "Thành công: Đã cập nhật dữ liệu từ yfinance và FRED."
    assert status["data_loader"] == "Thành công: Đã làm sạch và hợp nhất dữ liệu."
    assert status["features"] == "Thành công: Đã tính toán đặc trưng."
    assert status["labeling"] == "Thành công: Đã gán nhãn cơ chế thị trường."
    assert status["modeling"] == "Thành công: Đã huấn luyện lại mô hình."
    assert status["evaluation"] == "Thành công: Đã xuất báo cáo và dữ liệu giải thích SHAP."


def test_fetch_all_live_data_saves_expected_files(tmp_path, monkeypatch):
    """Verify that fetch_all_live_data downloads, parses, and saves S&P 500, VIX, and macro data."""
    # Giả lập các đường dẫn thư mục thô trong config
    monkeypatch.setattr(src.data_fetcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(src.data_fetcher, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(src.data_fetcher, "RAW_OHLCV_PATH", tmp_path / "raw" / "sp500_ohlcv.csv")
    monkeypatch.setattr(src.data_fetcher, "RAW_SECTORS_PATH", tmp_path / "raw" / "sp500_companies.csv")
    monkeypatch.setattr(src.data_fetcher, "RAW_VIX_PATH", tmp_path / "raw" / "vixcls.csv")
    
    # Tạo tệp tin SP500.csv giả lập
    sp500_input = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "BRK.B"],
        "GICS Sector": ["IT", "IT", "Financials"]
    })
    sp500_input.to_csv(tmp_path / "SP500.csv", index=False)
    
    # Giả lập yfinance.download
    # yfinance trả về MultiIndex columns: Level 0 = Price Metric, Level 1 = Ticker
    mock_columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["AAPL", "MSFT", "BRK-B"]],
        names=["Price", "Ticker"]
    )
    mock_dates = pd.date_range("2026-05-01", periods=2)
    mock_data = pd.DataFrame(100.0, index=mock_dates, columns=mock_columns)
    
    mock_yf_download = MagicMock(return_value=mock_data)
    monkeypatch.setattr(src.data_fetcher.yf, "download", mock_yf_download)
    
    # Giả lập DataReader cho VIX từ FRED
    mock_vix_data = pd.DataFrame({"VIXCLS": [15.5, 16.2]}, index=mock_dates)
    mock_vix_data.index.name = "DATE"
    mock_dr = MagicMock(return_value=mock_vix_data)
    monkeypatch.setattr(src.data_fetcher.web, "DataReader", mock_dr)
    
    # Giả lập fetch_macro_data từ FRED để tránh gọi mạng thật
    mock_macro = MagicMock()
    monkeypatch.setattr(src.data_fetcher, "fetch_macro_data", mock_macro)
    
    # Gọi hàm tải dữ liệu tự động
    src.data_fetcher.fetch_all_live_data(start_date="2026-05-01", end_date="2026-05-02")
    
    # Xác minh các mock được gọi đúng tham số
    mock_yf_download.assert_called_once()
    mock_dr.assert_called_once_with("VIXCLS", "fred", "2026-05-01", "2026-05-02")
    mock_macro.assert_called_once_with(start_date="2026-05-01", end_date="2026-05-02")
    
    # Xác minh các tệp tin thô được lưu trữ thành công
    assert (tmp_path / "raw" / "sp500_companies.csv").exists()
    assert (tmp_path / "raw" / "sp500_ohlcv.csv").exists()
    assert (tmp_path / "raw" / "vixcls.csv").exists()
    
    # Xác minh định dạng của sp500_ohlcv.csv lưu trên ổ đĩa
    ohlcv_written = pd.read_csv(tmp_path / "raw" / "sp500_ohlcv.csv")
    assert set(ohlcv_written.columns) == {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    assert set(ohlcv_written["ticker"]) == {"AAPL", "MSFT", "BRK.B"} # Đã ánh xạ ngược lại từ BRK-B về BRK.B
