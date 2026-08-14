"""
Secondary Rank-Conditioned XGBoost Regressor.
Maps 5 inputs (Concrete, Strand, Rebar, Span_ft, Design_Rank) + engineered features
to 7 structural design target parameters across all 5 alternative design ranks.
Uses Group-based splitting on Family_ID to prevent data leakage.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.split_and_scale import split_and_scale_all_alternatives
from src.data.load_data import TARGET_COLS

def calculate_metrics(Y_true: pd.DataFrame, Y_pred: np.ndarray) -> dict:
    """
    Computes R2, RMSE, RMSE% of range, and MAPE for each target column.
    """
    metrics = {}
    r2_scores = []
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_true.columns, index=Y_true.index)

    for col in Y_true.columns:
        y_t = Y_true[col].values
        y_p = Y_pred_df[col].values

        r2 = float(r2_score(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        val_range = float(y_t.max() - y_t.min())
        rmse_pct = float((rmse / val_range) * 100.0) if val_range > 0 else 0.0
        mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)

        metrics[col] = {
            "r2": round(r2, 5),
            "rmse": round(rmse, 4),
            "rmse_pct_range": round(rmse_pct, 4),
            "mape_pct": round(mape, 4)
        }
        r2_scores.append(r2)

    metrics["mean_r2"] = round(float(np.mean(r2_scores)), 5)
    return metrics

def train_secondary_rank_model():
    """
    Trains and evaluates the secondary rank-conditioned XGBoost model across Alt 1-5.
    """
    print("\n--- Training Secondary Rank-Conditioned XGBoost (All Alternatives 1-5) ---")
    X_train, X_test, Y_train, Y_test, scaler, feature_names = split_and_scale_all_alternatives()

    # Configure XGBoost for rank-conditioned multi-output regression
    xgb_base = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1
    )
    secondary_model = MultiOutputRegressor(xgb_base)
    secondary_model.fit(X_train, Y_train)

    test_preds = secondary_model.predict(X_test)
    metrics = calculate_metrics(Y_test, test_preds)

    print("\n=================================================================")
    print(f"SECONDARY RANK-CONDITIONED MODEL TEST MEAN R2: {metrics['mean_r2']:.5f}")
    print("=================================================================")
    for col in TARGET_COLS:
        m = metrics[col]
        print(f"Target: {col:<32} | R² = {m['r2']:.4f} | RMSE = {m['rmse']:.3f} | RMSE% = {m['rmse_pct_range']:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(secondary_model, "models/secondary_rank_model.pkl")

    os.makedirs("reports", exist_ok=True)
    with open("reports/secondary_rank_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSecondary Model artifact saved: models/secondary_rank_model.pkl")
    return secondary_model, metrics

if __name__ == "__main__":
    train_secondary_rank_model()
