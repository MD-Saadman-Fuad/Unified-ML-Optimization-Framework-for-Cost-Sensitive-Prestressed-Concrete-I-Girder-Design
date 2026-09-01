"""
Stand-alone generator for high-resolution Actual vs. Predicted comparison plots.
Generates:
1. Per-target Scatter plots with ideal y=x line, 10% error bands, and metrics.
2. Per-target Sample Series comparison plots (Actual in Dark Blue vs Predicted in Vibrant Orange).
3. Master 7-target Grid Dashboard scatter plot image.
4. Master 7-target Grid Sample comparison series image.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.split_and_scale import split_and_scale_data
from src.data.load_data import TARGET_COLS

def generate_all_comparison_plots(output_dir: str = "reports/scatter_plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit Primary MultiOutput XGBoost Model & Save
    print("[Plots Generator] Fitting Primary XGBoost Model on 14 domain features...")
    X_train, X_test, Y_train, Y_test, scaler, cols = split_and_scale_data()
    
    base_xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model = MultiOutputRegressor(base_xgb)
    model.fit(X_train, Y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_primary.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=Y_test.columns, index=Y_test.index)

    # Set aesthetics
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams['font.size'] = 10
    
    target_list = list(Y_test.columns)

    print(f"[Plots Generator] Rendering high-resolution plots for {len(target_list)} targets...")

    # 1. Individual Target Plots
    for col in target_list:
        y_true = Y_test[col].values
        y_pred = Y_pred_df[col].values
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100

        safe_filename = col.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")

        # --- A. Scatter Plot: Actual vs Predicted ---
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        
        ax.scatter(
            y_true, y_pred, 
            c="#2563eb", alpha=0.85, edgecolors="#1e3a8a", 
            s=55, label="Model Predictions"
        )

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        buffer = (max_val - min_val) * 0.06
        min_val -= buffer
        max_val += buffer

        # Ideal 45-degree reference line y = x
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal Fit (y = x)")
        
        # +-10% Error Limit Bounds
        ax.plot([min_val, max_val], [min_val * 0.90, max_val * 0.90], 'k:', alpha=0.45, label="±10% Tolerance Limit")
        ax.plot([min_val, max_val], [min_val * 1.10, max_val * 1.10], 'k:', alpha=0.45)

        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

        ax.set_xlabel(f"Actual Value ({col})", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_ylabel(f"Predicted Value ({col})", fontsize=11, fontweight="bold", color="#1e293b")
        ax.set_title(f"Target: {col}\nR² = {r2:.4f}  |  RMSE = {rmse:.3f}  |  MAPE = {mape:.2f}%", fontsize=12, fontweight="bold", pad=12)
        
        ax.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{safe_filename}_scatter.png"), dpi=300)
        plt.close()

        # --- B. Sample Series Plot: Actual (Blue) vs Predicted (Orange) ---
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        sample_indices = np.arange(1, len(y_true) + 1)

        ax.plot(sample_indices, y_true, 'o-', color="#2563eb", label="Actual Ground-Truth", linewidth=1.8, markersize=5.5, alpha=0.90)
        ax.plot(sample_indices, y_pred, 's--', color="#f97316", label="Predicted Surrogate ML", linewidth=1.8, markersize=5.5, alpha=0.90)

        ax.set_xlabel("Test Sample Index", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{col}", fontsize=11, fontweight="bold")
        ax.set_title(f"Sample-by-Sample Comparison: Actual (Blue) vs Predicted (Orange) [{col}]\nR² = {r2:.4f}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"{safe_filename}_series_comparison.png"), dpi=300)
        plt.close()

    # 2. Master Dashboard Plot Grid (All 7 Parameters Scatter in 1 Image)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes = axes.flatten()

    for idx, col in enumerate(target_list):
        ax = axes[idx]
        y_true = Y_test[col].values
        y_pred = Y_pred_df[col].values
        r2 = r2_score(y_true, y_pred)

        ax.scatter(y_true, y_pred, color="#2563eb", alpha=0.75, edgecolors="#1e3a8a", s=35, label="Predictions")
        
        m_val = min(np.min(y_true), np.min(y_pred))
        M_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([m_val, M_val], [m_val, M_val], 'r--', linewidth=1.8, label="Ideal y = x")

        ax.set_title(f"{col}\n(R² = {r2:.4f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")

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

        ax.plot(sample_idx, y_true, 'o-', color="#2563eb", label="Actual (Blue)", linewidth=1.2, markersize=3.5, alpha=0.85)
        ax.plot(sample_idx, y_pred, 's--', color="#f97316", label="Predicted (Orange)", linewidth=1.2, markersize=3.5, alpha=0.85)
        
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

    print(f"\n[Plots Generator] SUCCESS: All 16 high-resolution comparison plots saved to: {output_dir}")

if __name__ == "__main__":
    generate_all_comparison_plots()
