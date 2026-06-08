from __future__ import annotations

from src.data_fetcher import fetch_all_live_data
from src.data_loader import build_clean_market_inputs
from src.features import build_feature_datasets
from src.labeling import build_labeled_dataset
from src.modeling import train_all_models
from src.evaluation import export_xgboost_evaluation_artifacts


def run_full_pipeline() -> dict[str, str]:
    """Run the entire data fetching, processing, labeling, training, and evaluation pipeline."""
    status = {}
    
    # Giai đoạn 1: Tải dữ liệu trực tuyến
    print("\n=== [Pipeline] Giai đoạn 1/6: Tải dữ liệu mới nhất ===")
    fetch_all_live_data()
    status["data_fetcher"] = "Thành công: Đã cập nhật dữ liệu từ yfinance và FRED."
    
    # Giai đoạn 2: Tiền xử lý & Hợp nhất
    print("\n=== [Pipeline] Giai đoạn 2/6: Tiền xử lý và gộp dữ liệu ===")
    build_clean_market_inputs()
    status["data_loader"] = "Thành công: Đã làm sạch và hợp nhất dữ liệu."
    
    # Giai đoạn 3: Tính toán Đặc trưng
    print("\n=== [Pipeline] Giai đoạn 3/6: Tính toán đặc trưng vĩ mô & kỹ thuật ===")
    build_feature_datasets()
    status["features"] = "Thành công: Đã tính toán đặc trưng."
    
    # Giai đoạn 4: Gán nhãn bằng Rolling HMM
    print("\n=== [Pipeline] Giai đoạn 4/6: Gán nhãn cơ chế thị trường HMM ===")
    build_labeled_dataset(method="hmm")
    status["labeling"] = "Thành công: Đã gán nhãn cơ chế thị trường."
    
    # Giai đoạn 5: Huấn luyện mô hình
    print("\n=== [Pipeline] Giai đoạn 5/6: Huấn luyện lại các mô hình baseline & XGBoost ===")
    train_all_models()
    status["modeling"] = "Thành công: Đã huấn luyện lại mô hình."
    
    # Giai đoạn 6: Đánh giá, Backtest, SHAP
    print("\n=== [Pipeline] Giai đoạn 6/6: Đánh giá, chạy Backtest và tính toán SHAP ===")
    export_xgboost_evaluation_artifacts()
    status["evaluation"] = "Thành công: Đã xuất báo cáo và dữ liệu giải thích SHAP."
    
    print("\n=== [Pipeline] Hoàn tất toàn bộ quy trình cập nhật và huấn luyện lại! ===")
    return status


if __name__ == "__main__":
    run_full_pipeline()
