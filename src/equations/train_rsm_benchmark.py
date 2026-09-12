"""
Response Surface Methodology (RSM) Benchmark Modeling.
Fits 2nd-order Sparse LassoCV polynomial regression across normalized inputs (Concrete, Strand, Rebar, Span_ft).
Improvements:
  1. LassoCV sparse pruning  — auto-zeros near-zero cross-terms per target.
  2. MinMaxScaler [0,1] normalization — interpretable, comparable coefficients.
  3. Ng and Hp physics-derived — these two targets are physically constrained
     (Ng = round(37/S)+1, Hp = 0.40*L) and do not respond to polynomial regression.
If 2nd-order validation is inadequate (mean R2 < 0.85), conditionally tests 3rd-order.
Exports minimal algebraic closed-form equations to JSON.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_primary_dataset, INPUT_COLS, TARGET_COLS
from sklearn.model_selection import train_test_split

# Targets that ARE regressed (continuous, cost-driven)
RSM_TARGETS = [
    'Gir Dep (in)',
    'Lat Spac (ft)',
    'bot flange bot part depth (in)',
    'bot flange bot part width (in)',
    'Number of strand per girder'
]

# Targets replaced by physics formulas (regression adds noise here)
PHYS_TARGETS = ['No. of Gir', 'Harp Pos (ft)']

LASSO_ALPHAS = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


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


def _extract_equations(poly, lasso_dict, input_cols, scale_info, order_label):
    """Build simplified equation strings from fitted Lasso models."""
    feat_names = poly.get_feature_names_out(input_cols)
    equations = {}

    for col, lasso in lasso_dict.items():
        coefs = lasso.coef_
        inter = float(lasso.intercept_)
        terms = [f"{inter:+.5f}"]
        active = 0
        for fname, coef in zip(feat_names, coefs):
            if fname == "1" or abs(coef) < 1e-6:
                continue
            clean = fname.replace(" ", " * ")
            terms.append(f"{coef:+.6f} * {clean}")
            active += 1
        equations[col] = {
            "equation": " ".join(terms),
            "active_terms": active,
            "lasso_alpha": round(float(lasso.alpha_), 6),
            "degree": order_label,
            "note": f"Inputs normalized to [0,1]. Scale ranges: {scale_info}"
        }

    # Physics-derived entries
    equations["No. of Gir"] = {
        "equation": "round(37.0 / Lat_Spac_ft) + 1",
        "active_terms": 1,
        "note": "Physics-derived from bridge deck width = 37.0 ft. Not a regression target."
    }
    equations["Harp Pos (ft)"] = {
        "equation": "0.40 * Span_ft",
        "active_terms": 1,
        "note": "Physics-derived: harping at 40% of span (within AASHTO 0.30–0.50L zone). Not a regression target."
    }
    return equations


def train_rsm_benchmark(filepath: str = "Girder_Dataset.xlsx"):
    """
    Trains and evaluates Sparse LassoCV 2nd-Order RSM with normalized inputs.
    Physics-derives Ng and Hp. Conditionally tests 3rd-Order LassoCV RSM.
    """
    df = load_primary_dataset(filepath)
    X_raw = df[INPUT_COLS].copy()
    Y_raw = df[TARGET_COLS].copy()

    X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
        X_raw, Y_raw, test_size=0.2, random_state=42, stratify=df['Span_ft']
    )

    # ── Step 1: Normalize inputs to [0, 1] ───────────────────────────────────
    input_scaler = MinMaxScaler()
    input_scaler.fit(X_train_raw)

    X_train_norm = pd.DataFrame(input_scaler.transform(X_train_raw), columns=INPUT_COLS)
    X_test_norm  = pd.DataFrame(input_scaler.transform(X_test_raw),  columns=INPUT_COLS)

    scale_info = {col: {"min": float(input_scaler.data_min_[i]),
                        "max": float(input_scaler.data_max_[i])}
                  for i, col in enumerate(INPUT_COLS)}

    # ── Step 2: 2nd-Order Polynomial Expansion ───────────────────────────────
    print("\n--- Fitting Sparse LassoCV 2nd-Order RSM (normalized inputs) ---")
    poly_2nd = PolynomialFeatures(degree=2, include_bias=True)
    X_tr_poly = poly_2nd.fit_transform(X_train_norm)
    X_te_poly  = poly_2nd.transform(X_test_norm)

    # ── Step 3: LassoCV per RSM target ───────────────────────────────────────
    lasso_models = {}
    preds = np.zeros((len(Y_test), len(TARGET_COLS)))

    for col in RSM_TARGETS:
        lasso = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=15000, random_state=42)
        lasso.fit(X_tr_poly, Y_train[col])
        i = TARGET_COLS.index(col)
        preds[:, i] = lasso.predict(X_te_poly)
        lasso_models[col] = lasso

    # ── Step 4: Physics-derived Ng and Hp ────────────────────────────────────
    S_pred = preds[:, TARGET_COLS.index('Lat Spac (ft)')]
    L_vals = X_test_raw['Span_ft'].values

    Ng_phys = np.array([int(round(37.0 / max(s, 2.0))) + 1 for s in S_pred])
    Ng_phys = np.clip(Ng_phys, 6, 13).astype(float)
    Hp_phys = 0.40 * L_vals

    preds[:, TARGET_COLS.index('No. of Gir')]    = Ng_phys
    preds[:, TARGET_COLS.index('Harp Pos (ft)')] = Hp_phys

    metrics_2nd = calculate_metrics(Y_test, preds)

    print(f"Sparse LassoCV 2nd-Order RSM Test Mean R2: {metrics_2nd['mean_r2']:.5f}")
    for col in TARGET_COLS:
        m = metrics_2nd[col]
        tag = " [physics-derived]" if col in PHYS_TARGETS else \
              f" [alpha={lasso_models[col].alpha_:.5f}, terms={int(np.sum(lasso_models[col].coef_ != 0))}]"
        print(f"  Target: {col:<36} | R2 = {m['r2']:.4f} | RMSE = {m['rmse']:.3f}{tag}")

    # ── Step 5: Conditional 3rd-Order check ──────────────────────────────────
    selected_degree = 2
    final_poly      = poly_2nd
    final_metrics   = metrics_2nd
    final_lasso     = lasso_models
    final_preds     = preds

    if metrics_2nd['mean_r2'] < 0.85:
        print("\n[Condition Triggered] 2nd-Order Mean R2 < 0.85. Testing 3rd-Order Sparse LassoCV RSM...")
        poly_3rd   = PolynomialFeatures(degree=3, include_bias=True)
        X_tr_3rd   = poly_3rd.fit_transform(X_train_norm)
        X_te_3rd   = poly_3rd.transform(X_test_norm)

        lasso3_models = {}
        preds_3rd = np.zeros((len(Y_test), len(TARGET_COLS)))

        for col in RSM_TARGETS:
            l3 = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=25000, random_state=42)
            l3.fit(X_tr_3rd, Y_train[col])
            i = TARGET_COLS.index(col)
            preds_3rd[:, i] = l3.predict(X_te_3rd)
            lasso3_models[col] = l3

        preds_3rd[:, TARGET_COLS.index('No. of Gir')]    = Ng_phys
        preds_3rd[:, TARGET_COLS.index('Harp Pos (ft)')] = Hp_phys

        metrics_3rd = calculate_metrics(Y_test, preds_3rd)
        print(f"3rd-Order Sparse LassoCV RSM Test Mean R2: {metrics_3rd['mean_r2']:.5f}")

        if metrics_3rd['mean_r2'] > metrics_2nd['mean_r2']:
            selected_degree = 3
            final_poly      = poly_3rd
            final_metrics   = metrics_3rd
            final_lasso     = lasso3_models
            final_preds     = preds_3rd
            print("Selected 3rd-Order LassoCV RSM as superior benchmark.")
        else:
            print("2nd-Order remains superior — keeping 2nd-Order.")
    else:
        print("\n[Validated] LassoCV Sparse RSM adequate (Mean R2 >= 0.85).")

    # ── Step 6: Build equation strings ───────────────────────────────────────
    eqs_detail = _extract_equations(final_poly, final_lasso, INPUT_COLS, scale_info, selected_degree)

    # ── Step 7: Save artifacts ────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/equations", exist_ok=True)

    model_artifact = {
        "lasso_per_target": final_lasso,
        "poly": final_poly,
        "input_scaler": input_scaler,
        "phys_targets": PHYS_TARGETS
    }
    joblib.dump(model_artifact, "models/rsm_model.pkl")
    joblib.dump(final_poly, "models/rsm_poly_transformer.pkl")

    rsm_report = {
        "selected_degree": selected_degree,
        "normalization": "MinMaxScaler [0,1] on all 4 raw inputs before polynomial expansion",
        "method": "LassoCV sparse regression — near-zero cross-terms auto-pruned per target",
        "physics_derived": {
            "No. of Gir": "round(37.0 / S) + 1   (bridge deck width = 37.0 ft)",
            "Harp Pos (ft)": "0.40 * Span_ft   (AASHTO midspan harping zone)"
        },
        "input_scale_ranges": scale_info,
        "metrics": final_metrics,
        "equations": {k: v["equation"] for k, v in eqs_detail.items()},
        "equations_detail": eqs_detail
    }

    with open("reports/equations/rsm_equations.json", "w") as f:
        json.dump(rsm_report, f, indent=2)

    print("\nRSM Artifacts saved:")
    print(" - Model:     models/rsm_model.pkl")
    print(" - Equations: reports/equations/rsm_equations.json")

    return model_artifact, final_metrics, {k: v["equation"] for k, v in eqs_detail.items()}


if __name__ == "__main__":
    train_rsm_benchmark()
