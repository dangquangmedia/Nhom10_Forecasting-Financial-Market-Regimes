# RegimeLens AI
# SV Thực hiện :Lê Văn Quang    MSV:25410011

RegimeLens AI là một dự án phân tích và dự báo regime thị trường tài chính cho chỉ số S&P 500. Project sử dụng dữ liệu OHLCV, thông tin ngành của công ty và chỉ số VIX từ FRED để xây dựng tập dữ liệu theo ngày phục vụ huấn luyện mô hình học máy.

Mô hình chính: XGBoost.

Các mô hình so sánh:

- Logistic Regression
- Random Forest

Dashboard và kết quả mô hình chỉ dùng cho mục đích nghiên cứu và giáo dục, không phải lời khuyên đầu tư.

## Tổng quan hệ thống

Project gồm 3 phần chính:

- Pipeline dữ liệu: thu thập, làm sạch, kết hợp và tạo đặc trưng từ dữ liệu thị trường.
- Huấn luyện mô hình: sinh nhãn regime, xây dựng tập dữ liệu cuối cùng và train các mô hình phân loại.
- Giao diện: dashboard cho xem kết quả dự báo, chỉ số đánh giá và giải thích mô hình.

## Yêu cầu môi trường

- Python 3.10+
- Node.js 18+ cho frontend React/Vite

## Cài đặt local

### 1) Tạo môi trường Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Chuẩn bị dữ liệu thô

Đặt các file CSV vào thư mục data/raw/:

```text
data/raw/sp500_ohlcv.csv
data/raw/sp500_companies.csv
data/raw/vixcls.csv
```

Các file này cần có các cột tối thiểu như sau:

```text
sp500_ohlcv.csv: date, ticker hoặc symbol, open, high, low, close, volume
sp500_companies.csv: symbol hoặc ticker, sector hoặc gics_sector
vixcls.csv: date hoặc observation_date, vixcls hoặc vix
```

## Chạy pipeline dữ liệu

```bash
python -m src.data_loader
python -m src.features
python -m src.labeling
```

Các output chính:

```text
data/processed/sp500_clean.csv
data/processed/stock_features.csv
data/processed/market_features.csv
data/processed/final_dataset.csv
```

## Huấn luyện mô hình

```bash
python -m src.modeling
python -m src.evaluation
```

Các artifact chính:

```text
data/models/logistic_regression.joblib
data/models/random_forest.joblib
data/models/xgboost.joblib
reports/model_results.csv
reports/xgboost_test_predictions.csv
reports/xgboost_confusion_matrix.csv
reports/xgboost_classification_report.csv
reports/xgboost_feature_importance.csv
```

## Chạy tests

```bash
pytest -q
```

## Chạy dashboard và API local

### Backend API

```bash
uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
```

### Dashboard Streamlit

```bash
streamlit run app/dashboard.py
```

### Frontend React

```bash
cd frontend
npm install
npm run dev
```

## Cấu trúc thư mục chính

```text
app/            # API FastAPI và dashboard Streamlit
frontend/       # giao diện React + Vite
src/            # pipeline dữ liệu, feature engineering, labeling, modeling
data/           # dữ liệu thô, đã xử lý và model artifacts
reports/        # kết quả đánh giá và báo cáo
```

## Ghi chú

- Dự án đang dùng XGBoost như mô hình chính để dự báo regime tiếp theo.
- Các tập dữ liệu và mô hình được lưu trong thư mục data/ và reports/ để có thể dùng trực tiếp cho dashboard.
- Nếu muốn chạy frontend kết hợp với backend, cần đảm bảo API đang chạy ở cổng 8000 và frontend đã build hoặc chạy dev server.
