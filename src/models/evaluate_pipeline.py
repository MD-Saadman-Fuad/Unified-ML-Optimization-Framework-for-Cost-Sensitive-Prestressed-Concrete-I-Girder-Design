"""
Comprehensive Evaluation & Reporting Suite.
Executes primary surrogate training, RSM benchmark fitting, secondary rank-conditioned model training,
generates comparison matrices, json reports, and scatter plots.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.train_primary_xgboost import train_primary_xgboost
from src.equations.train_rsm_benchmark import train_rsm_benchmark
from src.models.train_secondary_rank_model import train_secondary_rank_model
from src.preprocessing.split_and_scale import split_and_scale_data
from src.data.load_data import TARGET_COLS
from sklearn.metrics import r2_score

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

        ax.scatter(y_true, y_pred, alpha=0.75, color="#1d4ed8", edgecolors="k", linewidth=0.5)

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

def run_full_pipeline_evaluation():
    """
    Runs full evaluation across Primary XGBoost, RSM Benchmark, and Secondary Rank-Conditioned Model.
    """
    print("\n=======================================================")
    print("      RUNNING FULL ML & PHYSICS PIPELINE EVALUATION    ")
    print("=======================================================\n")

    # 1. Primary XGBoost Model
    primary_xgb, primary_metrics = train_primary_xgboost()

    # 2. RSM Benchmark Model
    rsm_model, rsm_metrics, rsm_eqs = train_rsm_benchmark()

    # 3. Secondary Rank-Conditioned Model
    secondary_model, secondary_metrics = train_secondary_rank_model()

    # Consolidate benchmark comparison table
    benchmark_all = {
        "Primary_XGBoost_Alt1": primary_metrics,
        "RSM_Polynomial_Benchmark": rsm_metrics,
        "Secondary_Rank_Conditioned_XGBoost": secondary_metrics
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_benchmark.json", "w") as f:
        json.dump(benchmark_all, f, indent=2)

    # Build CSV Comparison
    rows = []
    for model_name, m_dict in benchmark_all.items():
        row = {"Model": model_name, "Mean_R2": m_dict["mean_r2"]}
        for target in TARGET_COLS:
            row[f"{target}_R2"] = m_dict[target]["r2"]
            row[f"{target}_RMSE"] = m_dict[target]["rmse"]
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv("reports/model_comparison.csv", index=False)

    # Scatter Plots for Primary Model
    X_tr, X_te, Y_tr, Y_te, sc, cols = split_and_scale_data()
    primary_preds = primary_xgb.predict(X_te)
    generate_scatter_plots(Y_te, primary_preds)

    print("\n=======================================================")
    print("              FINAL PIPELINE COMPARISON SUMMARY        ")
    print("=======================================================")
    print(df_comp[["Model", "Mean_R2"]].to_string(index=False))
    print("\nReports & Artifacts successfully generated:")
    print(" - Comparison Matrix: reports/model_comparison.csv")
    print(" - Benchmark JSON:   reports/model_benchmark.json")
    print(" - RSM Equations:    reports/equations/rsm_equations.json")
    print(" - Scatter Plots:    reports/scatter_plots/*.png")

if __name__ == "__main__":
    run_full_pipeline_evaluation()
