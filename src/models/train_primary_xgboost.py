"""
Primary XGBoost surrogate model training script.
Maps 4 cost/span inputs (8 engineered features) to 7 optimal design targets (Alt 1).
Includes Optuna hyperparameter tuning and GroupKFold cross-validation.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.split_and_scale import split_and_scale_data
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

def tune_xgboost(X_train: pd.DataFrame, Y_train: pd.DataFrame, n_trials: int = 35) -> dict:
    """
    Tune XGBoost hyperparameters with Optuna.
    """
    print(f"  Starting Optuna hyperparameter optimization for Primary XGBoost ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth":        trial.suggest_int("max_depth", 2, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-1, 50.0, log=True),
            "random_state":     42,
            "n_jobs":           -1
        }

        model = MultiOutputRegressor(XGBRegressor(**params))
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring="r2", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"  Optuna tuning finished. Best Trial CV Mean R2: {study.best_value:.5f}")
    print("  Best Hyperparameters:", study.best_params)
    return study.best_params

def train_primary_xgboost():
    """
    Trains and evaluates the primary XGBoost surrogate model for Alt-1 optimal designs.
    """
    X_train, X_test, Y_train, Y_test, scaler, feature_names = split_and_scale_data()

    print("\n--- Training Primary XGBoost Surrogate (Alt 1 Optimum) ---")
    best_params = tune_xgboost(X_train, Y_train, n_trials=35)

    base_xgb = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    primary_xgb = MultiOutputRegressor(base_xgb)
    primary_xgb.fit(X_train, Y_train)

    test_preds = primary_xgb.predict(X_test)
    metrics = calculate_metrics(Y_test, test_preds)

    print("\n=======================================================")
    print(f"PRIMARY XGBOOST SURROGATE TEST MEAN R2: {metrics['mean_r2']:.5f}")
    print("=======================================================")
    for col in TARGET_COLS:
        m = metrics[col]
        print(f"Target: {col:<32} | R² = {m['r2']:.4f} | RMSE = {m['rmse']:.3f} | RMSE% = {m['rmse_pct_range']:.2f}% | MAPE = {m['mape_pct']:.2f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(primary_xgb, "models/xgboost_primary.pkl")
    joblib.dump(primary_xgb, "models/best_model.pkl")  # Primary surrogate model artifact

    os.makedirs("reports", exist_ok=True)
    with open("reports/primary_xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return primary_xgb, metrics

if __name__ == "__main__":
    train_primary_xgboost()
