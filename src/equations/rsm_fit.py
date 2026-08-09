"""
Response Surface Methodology (RSM) 2nd-order polynomial equation derivation module.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_dataset, INPUT_COLS, TARGET_COLS

def fit_rsm_equations(filepath: str = "Girder_Dataset.xlsx"):
    """
    Fits 2nd-order Response Surface Methodology models for each target variable
    and extracts closed-form explicit mathematical equations.
    """
    df = load_dataset(filepath)
    X_raw = df[INPUT_COLS].copy()
    Y_raw = df[TARGET_COLS].copy()

    # Create 2nd-order polynomial features: 1, x1, x2, x3, x4, x1^2, x1*x2, ..., x4^2
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_raw)
    feature_names = poly.get_feature_names_out(INPUT_COLS)

    rsm_results = {}
    md_lines = [
        "# Response Surface Methodology (RSM) Explicit Design Equations",
        "",
        "Second-order polynomial formulations derived across raw material costs and span length inputs.",
        "",
        "---",
        ""
    ]

    for col in TARGET_COLS:
        y = Y_raw[col].values
        model = LinearRegression()
        model.fit(X_poly, y)
        preds = model.predict(X_poly)

        r2 = float(r2_score(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))

        coefs = model.coef_
        coefs[0] = model.intercept_  # PolynomialFeatures include_bias=True has bias term at index 0

        # Build human-readable equation string
        terms = []
        coef_dict = {}
        for fname, c in zip(feature_names, coefs):
            coef_dict[fname] = round(float(c), 6)
            if abs(c) > 1e-6:
                formatted_fname = fname.replace(" ", " * ")
                if fname == "1":
                    terms.append(f"{c:+.4f}")
                else:
                    terms.append(f"{c:+.6f} * {formatted_fname}")

        eq_str = " ".join(terms)

        rsm_results[col] = {
            "r2": round(r2, 5),
            "rmse": round(rmse, 4),
            "coefficients": coef_dict,
            "formula": f"{col} = {eq_str}"
        }

        md_lines.append(f"## Target: `{col}`")
        md_lines.append(f"- **R² Score:** {r2:.4f}")
        md_lines.append(f"- **RMSE:** {rmse:.4f}")
        md_lines.append(f"- **Equation:**")
        md_lines.append(f"```text\n{col} = {eq_str}\n```")
        md_lines.append("")

    os.makedirs("reports/equations", exist_ok=True)
    with open("reports/equations/rsm_equations.json", "w") as f:
        json.dump(rsm_results, f, indent=2)

    with open("reports/equations/rsm_equations.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"[Phase 4] RSM Equation derivation complete.")
    print("Derived RSM equations saved to:")
    print(" - JSON: reports/equations/rsm_equations.json")
    print(" - Markdown: reports/equations/rsm_equations.md")

    return rsm_results

if __name__ == "__main__":
    fit_rsm_equations()
