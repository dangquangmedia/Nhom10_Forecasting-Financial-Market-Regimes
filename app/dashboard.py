from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    FINAL_DATASET_PATH,
    MODEL_RESULTS_PATH,
    XGBOOST_CLASSIFICATION_REPORT_PATH,
    XGBOOST_CONFUSION_MATRIX_PATH,
    XGBOOST_FEATURE_IMPORTANCE_PATH,
    XGBOOST_TEST_PREDICTIONS_PATH,
)


REGIME_COLORS = {
    "Bull": "#2ca02c",
    "Bear": "#d62728",
    "Sideways": "#7f7f7f",
    "High Volatility": "#ff7f0e",
    "Recovery": "#1f77b4",
}


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    required_paths = {
        "final_dataset": FINAL_DATASET_PATH,
        "model_results": MODEL_RESULTS_PATH,
        "predictions": XGBOOST_TEST_PREDICTIONS_PATH,
        "confusion_matrix": XGBOOST_CONFUSION_MATRIX_PATH,
        "classification_report": XGBOOST_CLASSIFICATION_REPORT_PATH,
        "feature_importance": XGBOOST_FEATURE_IMPORTANCE_PATH,
    }
    missing = [str(path) for path in required_paths.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Thiếu file đầu vào cho dashboard: " + ", ".join(missing))

    final_dataset = pd.read_csv(FINAL_DATASET_PATH, parse_dates=["date"])
    predictions = pd.read_csv(XGBOOST_TEST_PREDICTIONS_PATH, index_col=0, parse_dates=["date"])
    return {
        "final_dataset": final_dataset,
        "model_results": pd.read_csv(MODEL_RESULTS_PATH),
        "predictions": predictions,
        "confusion_matrix": pd.read_csv(XGBOOST_CONFUSION_MATRIX_PATH, index_col=0),
        "classification_report": pd.read_csv(XGBOOST_CLASSIFICATION_REPORT_PATH, index_col=0),
        "feature_importance": pd.read_csv(XGBOOST_FEATURE_IMPORTANCE_PATH, index_col=0),
    }


def regime_shift_warning(row: pd.Series) -> tuple[str, str]:
    if row["confidence"] < 0.45:
        return "Không chắc chắn", "Độ tin cậy của mô hình thấp. Nên xem dự đoán này như tín hiệu chưa chắc chắn."
    if row.get("prob_High Volatility", 0) >= 0.50:
        return "Cao", "Xác suất High Volatility đang cao. Rủi ro căng thẳng thị trường đáng chú ý."
    if row.get("prob_Bear", 0) >= 0.50:
        return "Cao", "Xác suất Bear đang cao. Rủi ro giảm giá đáng chú ý."
    if row["y_pred"] != row["y_true"]:
        return "Trung bình", "Regime dự đoán khác với nhãn kiểm thử thực tế."
    return "Thấp", "Chưa có cảnh báo chuyển regime mạnh tại ngày kiểm thử này."


def _probability_columns(predictions: pd.DataFrame) -> list[str]:
    return [column for column in predictions.columns if column.startswith("prob_")]


def _selected_prediction(predictions: pd.DataFrame, selected_date) -> pd.Series:
    selected = predictions[predictions["date"] == pd.Timestamp(selected_date)]
    if selected.empty:
        return predictions.iloc[-1]
    return selected.iloc[0]


def render_overview(data: dict[str, pd.DataFrame], selected_date) -> None:
    predictions = data["predictions"]
    row = _selected_prediction(predictions, selected_date)
    warning_level, warning_text = regime_shift_warning(row)

    st.subheader("Tổng quan dự đoán regime bằng XGBoost")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime dự đoán T+1", row["y_pred"])
    col2.metric("Nhãn thực tế", row["y_true"])
    col3.metric("Độ tin cậy", f"{row['confidence']:.1%}")
    col4.metric("Mức cảnh báo", warning_level)
    st.info(warning_text)

    prob_cols = _probability_columns(predictions)
    probs = pd.DataFrame(
        {
            "regime": [column.replace("prob_", "") for column in prob_cols],
            "probability": [row[column] for column in prob_cols],
        }
    )
    fig = px.bar(
        probs,
        x="regime",
        y="probability",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        title="Xác suất từng regime tại ngày được chọn",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_timeline(data: dict[str, pd.DataFrame]) -> None:
    predictions = data["predictions"].copy()
    predictions["correct"] = predictions["y_true"] == predictions["y_pred"]

    st.subheader("Dòng thời gian dự đoán")
    fig = px.scatter(
        predictions,
        x="date",
        y="confidence",
        color="y_pred",
        symbol="correct",
        color_discrete_map=REGIME_COLORS,
        hover_data=["y_true", "y_pred", "confidence"],
        title="Dự đoán XGBoost trên tập kiểm thử theo ngày",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    prob_cols = _probability_columns(predictions)
    probability_long = predictions.melt(
        id_vars=["date"],
        value_vars=prob_cols,
        var_name="regime",
        value_name="probability",
    )
    probability_long["regime"] = probability_long["regime"].str.replace("prob_", "", regex=False)
    fig_probs = px.line(
        probability_long,
        x="date",
        y="probability",
        color="regime",
        color_discrete_map=REGIME_COLORS,
        title="Xác suất các regime trong giai đoạn kiểm thử",
    )
    fig_probs.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_probs, use_container_width=True)


def render_market_context(data: dict[str, pd.DataFrame]) -> None:
    final_dataset = data["final_dataset"].copy()
    st.subheader("Bối cảnh thị trường")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=final_dataset["date"], y=final_dataset["vix"], name="VIX"))
    fig.add_trace(
        go.Scatter(
            x=final_dataset["date"],
            y=final_dataset["vix_ma_20"],
            name="VIX 20D MA",
            line={"dash": "dot"},
        )
    )
    fig.update_layout(title="VIX và trung bình động 20 ngày", yaxis_title="VIX")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_return = px.line(
            final_dataset,
            x="date",
            y="rolling_return_20",
            color="regime_current",
            color_discrete_map=REGIME_COLORS,
            title="Lợi suất rolling 20 ngày theo regime hiện tại",
        )
        st.plotly_chart(fig_return, use_container_width=True)
    with col2:
        fig_breadth = px.line(
            final_dataset,
            x="date",
            y="advance_ratio",
            color="regime_current",
            color_discrete_map=REGIME_COLORS,
            title="Độ rộng thị trường: tỷ lệ cổ phiếu tăng giá",
        )
        st.plotly_chart(fig_breadth, use_container_width=True)


def render_evaluation(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Đánh giá mô hình")
    st.dataframe(data["model_results"], use_container_width=True)

    confusion = data["confusion_matrix"]
    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Ma trận nhầm lẫn của XGBoost",
        labels={"x": "Dự đoán", "y": "Thực tế", "color": "Số lượng"},
    )
    st.plotly_chart(fig, use_container_width=True)

    report = data["classification_report"]
    st.dataframe(report, use_container_width=True)


def render_feature_importance(data: dict[str, pd.DataFrame]) -> None:
    st.subheader("Độ quan trọng đặc trưng của XGBoost")
    importance = data["feature_importance"].head(15).sort_values("importance", ascending=True)
    fig = px.bar(
        importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 đặc trưng quan trọng",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data["feature_importance"], use_container_width=True)


def render_disclaimer() -> None:
    st.subheader("Cảnh báo rủi ro")
    st.warning(
        "Dashboard này chỉ phục vụ mục đích học tập, nghiên cứu AI và tài chính định lượng. "
        "Kết quả dự đoán không phải lời khuyên đầu tư và không được hiểu là khuyến nghị mua, bán hoặc nắm giữ."
    )
    st.write(
        "Các regime tài chính là nhãn xấp xỉ được tạo từ logic gán nhãn và học máy. "
        "Mô hình có thể sai, đặc biệt khi thị trường thay đổi, dữ liệu bị drift hoặc xuất hiện regime biến động cao hiếm gặp."
    )


def main() -> None:
    st.set_page_config(page_title="RegimeLens AI", layout="wide")
    st.title("RegimeLens AI")
    st.caption("Dashboard dự đoán regime thị trường S&P 500")

    try:
        data = load_dashboard_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    predictions = data["predictions"]
    selected_date = st.sidebar.selectbox(
        "Ngày kiểm thử",
        options=predictions["date"].dt.date.tolist(),
        index=len(predictions) - 1,
    )
    st.sidebar.markdown("**Mô hình chính:** XGBoost")
    st.sidebar.markdown("**Target:** `regime_t_plus_1`")

    tabs = st.tabs(
        [
            "Tổng quan",
            "Dòng thời gian",
            "Bối cảnh thị trường",
            "Đánh giá",
            "Đặc trưng quan trọng",
            "Cảnh báo",
        ]
    )
    with tabs[0]:
        render_overview(data, selected_date)
    with tabs[1]:
        render_timeline(data)
    with tabs[2]:
        render_market_context(data)
    with tabs[3]:
        render_evaluation(data)
    with tabs[4]:
        render_feature_importance(data)
    with tabs[5]:
        render_disclaimer()


if __name__ == "__main__":
    main()
