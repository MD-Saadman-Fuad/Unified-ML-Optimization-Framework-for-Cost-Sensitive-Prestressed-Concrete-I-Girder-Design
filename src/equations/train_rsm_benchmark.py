"""
Response Surface Methodology (RSM) Benchmark Modeling.
Fits 2nd-order polynomial regression across raw inputs (Concrete, Strand, Rebar, Span_ft).
If 2nd-order validation is inadequate (mean R2 < 0.85), conditionally tests 3rd-order RSM.
Exports algebraic closed-form equations to JSON.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_primary_dataset, INPUT_COLS, TARGET_COLS
from sklearn.model_selection import train_test_split

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

def extract_rsm_algebraic_equations(poly: PolynomialFeatures, model: LinearRegression, input_cols: list, target_cols: list) -> dict:
    """
    Generates explicit algebraic formula strings for each target variable.
    """
    feature_names = poly.get_feature_names_out(input_cols)
    equations = {}

    for i, target in enumerate(target_cols):
        coefs = model.coef_[i] if hasattr(model, "coef_") and model.coef_.ndim > 1 else model.coef_
        intercept = model.intercept_[i] if hasattr(model, "intercept_") and hasattr(model.intercept_, "__len__") else model.intercept_

        terms = [f"{intercept:+.5f}"]
        for feat_name, coef in zip(feature_names, coefs):
            if feat_name == "1" or abs(coef) < 1e-7:
                continue
            # Format feature powers nicely
            clean_name = feat_name.replace(" ", " * ")
            terms.append(f"{coef:+.6f} * {clean_name}")

        eq_str = " ".join(terms)
        equations[target] = eq_str

    return equations

def train_rsm_benchmark(filepath: str = "Girder_Dataset.xlsx"):
    """
    Trains and evaluates 2nd-Order RSM (and 3rd-Order RSM if conditionally required).
    """
    df = load_primary_dataset(filepath)
    X_raw = df[INPUT_COLS].copy()
    Y_raw = df[TARGET_COLS].copy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_raw, Y_raw, test_size=0.2, random_state=42, stratify=df['Span_ft']
    )

    print("\n--- Fitting 2nd-Order Response Surface Methodology (RSM) Benchmark ---")
    poly_2nd = PolynomialFeatures(degree=2, include_bias=True)
    X_tr_2nd = poly_2nd.fit_transform(X_train)
    X_te_2nd = poly_2nd.transform(X_test)

    rsm_2nd = LinearRegression()
    rsm_2nd.fit(X_tr_2nd, Y_train)

    preds_2nd = rsm_2nd.predict(X_te_2nd)
    metrics_2nd = calculate_metrics(Y_test, preds_2nd)

    print(f"2nd-Order RSM Test Mean R2: {metrics_2nd['mean_r2']:.5f}")
    for col in TARGET_COLS:
        m = metrics_2nd[col]
        print(f"  Target: {col:<32} | R² = {m['r2']:.4f} | RMSE = {m['rmse']:.3f}")

    eqs_2nd = extract_rsm_algebraic_equations(poly_2nd, rsm_2nd, INPUT_COLS, TARGET_COLS)

    # Check if 2nd-order validation is adequate (Threshold: Mean R2 >= 0.85)
    selected_degree = 2
    final_rsm_model = rsm_2nd
    final_poly = poly_2nd
    final_metrics = metrics_2nd
    final_eqs = eqs_2nd

    if metrics_2nd['mean_r2'] < 0.85:
        print("\n[Condition Triggered] 2nd-Order RSM Mean R2 < 0.85. Testing 3rd-Order RSM...")
        poly_3rd = PolynomialFeatures(degree=3, include_bias=True)
        X_tr_3rd = poly_3rd.fit_transform(X_train)
        X_te_3rd = poly_3rd.transform(X_test)

        rsm_3rd = Ridge(alpha=1.0)
        rsm_3rd.fit(X_tr_3rd, Y_train)

        preds_3rd = rsm_3rd.predict(X_te_3rd)
        metrics_3rd = calculate_metrics(Y_test, preds_3rd)
        print(f"3rd-Order RSM Test Mean R2: {metrics_3rd['mean_r2']:.5f}")

        if metrics_3rd['mean_r2'] > metrics_2nd['mean_r2']:
            selected_degree = 3
            final_rsm_model = rsm_3rd
            final_poly = poly_3rd
            final_metrics = metrics_3rd
            final_eqs = extract_rsm_algebraic_equations(poly_3rd, rsm_3rd, INPUT_COLS, TARGET_COLS)
            print("Selected 3rd-Order RSM as superior polynomial benchmark.")
    else:
        print("\n[Condition Validated] 2nd-Order RSM validation is adequate (Mean R2 >= 0.85). 3rd-Order RSM not needed.")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/equations", exist_ok=True)

    joblib.dump(final_rsm_model, "models/rsm_model.pkl")
    joblib.dump(final_poly, "models/rsm_poly_transformer.pkl")

    rsm_report = {
        "selected_degree": selected_degree,
        "metrics": final_metrics,
        "equations": final_eqs
    }

    with open("reports/equations/rsm_equations.json", "w") as f:
        json.dump(rsm_report, f, indent=2)

    print("\nRSM Artifacts saved:")
    print(" - Model: models/rsm_model.pkl")
    print(" - JSON Equations: reports/equations/rsm_equations.json")

    return final_rsm_model, final_metrics, final_eqs

if __name__ == "__main__":
    train_rsm_benchmark()
