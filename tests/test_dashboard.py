import pandas as pd

from app.dashboard import load_dashboard_data, regime_shift_warning


def test_load_dashboard_data_reads_required_tables():
    load_dashboard_data.clear()

    data = load_dashboard_data()

    assert set(data) == {
        "final_dataset",
        "model_results",
        "predictions",
        "confusion_matrix",
        "classification_report",
        "feature_importance",
    }
    assert {"date", "y_true", "y_pred", "confidence", "prob_Sideways"}.issubset(
        data["predictions"].columns
    )
    assert not data["feature_importance"].empty


def test_regime_shift_warning_flags_high_volatility_probability():
    row = pd.Series(
        {
            "confidence": 0.80,
            "prob_High Volatility": 0.65,
            "prob_Bear": 0.10,
            "y_pred": "High Volatility",
            "y_true": "Sideways",
        }
    )

    level, message = regime_shift_warning(row)

    assert level == "Cao"
    assert "High Volatility" in message
