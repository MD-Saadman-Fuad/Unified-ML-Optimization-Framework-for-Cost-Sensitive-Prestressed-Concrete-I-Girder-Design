"""
Multi-output machine learning model training, hyperparameter tuning (Optuna), evaluation, and model artifact generation.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna

# Suppress Optuna logging noise
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

def optuna_tune_xgboost(X_train: pd.DataFrame, Y_train: pd.DataFrame, n_trials: int = 10) -> dict:
    """
    Optimizes XGBoost hyperparameters using Optuna cross-validation.
    """
    print(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }
        
        base_xgb = XGBRegressor(**params)
        model = MultiOutputRegressor(base_xgb)
        model.fit(X_train, Y_train)
        
        preds = model.predict(X_train)
        mean_r2 = float(r2_score(Y_train, preds, multioutput="uniform_average"))
        return mean_r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Optuna tuning finished. Best Trial Mean R2 (train): {study.best_value:.5f}")
    print("Best Hyperparameters:", study.best_params)
    return study.best_params

def generate_scatter_plots(Y_test: pd.DataFrame, Y_pred: np.ndarray, output_dir: str = "reports/scatter_plots"):
    """
    Generates predicted vs actual scatter plots for all target variables.
    """
    os.makedirs(output_dir, exist_ok=True)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)

    sns.set_theme(style="whitegrid")
    
    for col in Y_test.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        y_true = Y_test[col]
        y_pred = Y_pred_df[col]
        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, alpha=0.7, color="#2563eb", edgecolors="k", linewidth=0.5)
        
        # Perfect prediction 45-degree line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')

        ax.set_xlabel(f"Actual {col}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Predicted {col}", fontsize=11, fontweight="bold")
        ax.set_title(f"{col}\n(R² = {r2:.4f})", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()

        safe_filename = col.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        plt.savefig(os.path.join(output_dir, f"{safe_filename}_scatter.png"), dpi=200)
        plt.close()

def train_and_evaluate():
    """
    Runs the full model benchmark, hyperparameter tuning, evaluation, and artifact creation pipeline.
    """
    X_train, X_test, Y_train, Y_test, scaler, feature_names = split_and_scale_data()

    print("\n--- 1. Baseline Model Benchmarking ---")
    models = {
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42)),
        "GradientBoosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        "XGBoost_Default": MultiOutputRegressor(XGBRegressor(random_state=42))
    }

    benchmark_results = {}
    for name, m in models.items():
        m.fit(X_train, Y_train)
        preds = m.predict(X_test)
        metrics = calculate_metrics(Y_test, preds)
        benchmark_results[name] = metrics
        print(f"Model: {name:<20} | Test Mean R2: {metrics['mean_r2']:.5f}")

    print("\n--- 2. Hyperparameter Optimization ---")
    best_params = optuna_tune_xgboost(X_train, Y_train, n_trials=50)

    print("\n--- 3. Training Best Tuned XGBoost Model ---")
    best_xgb_base = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    best_model = MultiOutputRegressor(best_xgb_base)
    best_model.fit(X_train, Y_train)

    final_preds = best_model.predict(X_test)
    final_metrics = calculate_metrics(Y_test, final_preds)
    benchmark_results["Tuned_XGBoost"] = final_metrics

    print("\n=======================================================")
    print(f"FINAL TUNED XGBOOST TEST MEAN R²: {final_metrics['mean_r2']:.5f}")
    print("=======================================================")
    for col in TARGET_COLS:
        m = final_metrics[col]
        print(f"Target: {col:<32} | R² = {m['r2']:.4f} | RMSE = {m['rmse']:.3f} | RMSE% = {m['rmse_pct_range']:.2f}% | MAPE = {m['mape_pct']:.2f}%")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    joblib.dump(best_model, "models/best_model.pkl")

    # Save benchmark metrics to JSON
    with open("reports/model_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save summary CSV
    summary_rows = []
    for model_name, m_dict in benchmark_results.items():
        row = {"Model": model_name, "Mean_R2": m_dict["mean_r2"]}
        for target_col in TARGET_COLS:
            row[f"{target_col}_R2"] = m_dict[target_col]["r2"]
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("reports/model_comparison.csv", index=False)

    # Generate scatter plots
    generate_scatter_plots(Y_test, final_preds)
    print("\nArtifacts saved:")
    print(" - Model: models/best_model.pkl")
    print(" - Metrics: reports/model_benchmark.json")
    print(" - Comparison CSV: reports/model_comparison.csv")
    print(" - Scatter Plots: reports/scatter_plots/*.png")

    return best_model, final_metrics

if __name__ == "__main__":
    train_and_evaluate()
