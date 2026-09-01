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

def generate_scatter_plots(Y_test: pd.DataFrame, Y_pred: np.ndarray, X_test_raw: pd.DataFrame = None, output_dir: str = "reports/scatter_plots"):
    """
    Generates high-precision Actual vs. Predicted comparison plots:
    1. Per-target Scatter Plots (Actual vs Predicted) with ideal y=x line, +-10% error bands, and metrics.
    2. Per-target Sample Series Plots (Actual in Blue vs Predicted in Orange).
    3. Master 7-target Grid Dashboard comparing Actual vs Predicted across all parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)
    
    # Set clean aesthetic style
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams['font.size'] = 10
    
    target_list = list(Y_test.columns)

    # 1. Individual Target Plots (Scatter & Sample Series)
    for col in target_list:
        y_true = Y_test[col].values
        y_pred = Y_pred_df[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

        # --- A. Scatter Plot: Actual vs Predicted ---
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Plot points
        scatter = ax.scatter(
            y_true, y_pred, 
            c="#2563eb", alpha=0.8, edgecolors="#1e3a8a", 
            s=45, label="Predicted vs Actual"
        )

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        buffer = (max_val - min_val) * 0.05
        min_val -= buffer
        max_val += buffer

        # Ideal 45-degree line y = x
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal Fit (y = x)")
        
        # +-10% Error Bounds
        ax.plot([min_val, max_val], [min_val * 0.90, max_val * 0.90], 'k:', alpha=0.4, label="±10% Error Limit")
        ax.plot([min_val, max_val], [min_val * 1.10, max_val * 1.10], 'k:', alpha=0.4)

        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

        ax.set_xlabel(f"Actual {col}", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_ylabel(f"Predicted {col}", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_title(f"Target: {col}\nR² = {r2:.4f} | RMSE = {rmse:.3f} | MAPE = {mape:.2f}%", fontsize=12, fontweight="bold", pad=12)
        
        ax.legend(loc="upper left", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")
        plt.tight_layout()

        safe_filename = col.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        plt.savefig(os.path.join(output_dir, f"{safe_filename}_scatter.png"), dpi=300)
        plt.close()

        # --- B. Sample Series Plot: Actual (Blue) vs Predicted (Orange) ---
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sample_indices = np.arange(1, len(y_true) + 1)

        ax.plot(sample_indices, y_true, 'o-', color="#2563eb", label="Actual Value", linewidth=1.5, markersize=5, alpha=0.85)
        ax.plot(sample_indices, y_pred, 's--', color="#f97316", label="Predicted Value", linewidth=1.5, markersize=5, alpha=0.85)

        ax.set_xlabel("Test Sample Index", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{col}", fontsize=11, fontweight="bold")
        ax.set_title(f"Sample-by-Sample Comparison: Actual vs Predicted ({col})\nR² = {r2:.4f}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{safe_filename}_series_comparison.png"), dpi=300)
        plt.close()

    # 2. Master Dashboard Plot Grid (All 7 Parameters in 1 Image)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for idx, col in enumerate(target_list):
        ax = axes[idx]
        y_true = Y_test[col].values
        y_pred = Y_pred_df[col].values
        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, color="#2563eb", alpha=0.7, edgecolors="#1e3a8a", s=30, label="Predictions")
        
        m_val = min(np.min(y_true), np.min(y_pred))
        M_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([m_val, M_val], [m_val, M_val], 'r--', linewidth=1.8, label="y = x")

        ax.set_title(f"{col}\n(R² = {r2:.4f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

    # Hide extra subplots
    for i in range(len(target_list), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Master Evaluation: Primary XGBoost Actual vs. Predicted Across All 7 Girder Parameters", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "master_actual_vs_predicted_dashboard.png"), dpi=300)
    plt.close()

    # 3. Master Sample Series Overlay Dashboard
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, col in enumerate(target_list):
        ax = axes[idx]
        y_true = Y_test[col].values
        y_pred = Y_pred_df[col].values
        sample_idx = np.arange(1, len(y_true) + 1)

        ax.plot(sample_idx, y_true, 'o-', color="#2563eb", label="Actual", linewidth=1.2, markersize=3.5, alpha=0.85)
        ax.plot(sample_idx, y_pred, 's--', color="#f97316", label="Predicted", linewidth=1.2, markersize=3.5, alpha=0.85)
        
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{col} (R² = {r2:.4f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Sample Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.legend(fontsize=8, loc="best")

    for i in range(len(target_list), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Master Sample Comparison: Actual (Blue) vs. Predicted (Orange) Across Test Set", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "master_sample_comparison_dashboard.png"), dpi=300)
    plt.close()

    print(f"\n[Visualization] Generated all scatter and series comparison graphs in: {output_dir}")

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
