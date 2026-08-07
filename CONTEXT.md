# Agent Context — Unified ML Optimization Framework for Prestressed Concrete I-Girder Design

> This document is the authoritative context for AI agents and developers working on this codebase.
> Read this fully before making any code changes.

---

## Project Goal

Build a 6-phase surrogate ML pipeline that maps 4 material cost + span length inputs directly to 7 structural design parameters for a prestressed concrete I-girder bridge, bypassing computationally expensive iterative optimization.

---

## Domain Background

### What is a Prestressed Concrete I-Girder?

A structural beam used in bridge construction, made of concrete pre-tensioned with high-strength steel strands. The cross-section is shaped like the letter "I" (flanges + web) and must satisfy AASHTO/structural code constraints.

### Why Surrogate Modeling?

Classical optimization (Differential Evolution, EVOP, Box Complex) finds the minimum-cost design but requires full reruns when material costs change. A trained surrogate model returns predictions in milliseconds for any new cost combination.

---

## Data Schema

### Dataset: 675 rows × 11 columns

#### Input Features (X_raw — 4 columns)

| Column | Symbol | Description | Unit | Typical Range |
|--------|--------|-------------|------|---------------|
| A | Cc | Concrete Unit Cost | $/m³ | Low / Mid / High |
| B | Cs | Steel Rebar Unit Cost | $/ton | Low / Mid / High |
| C | Cp | Prestressing Strand Unit Cost | $/ton | Low / Mid / High |
| D | L  | Span Length | m | 20 – 40 m |

#### Target Outputs (Y_target — 7 columns)

| Column | Symbol | Description | Type | Constraint |
|--------|--------|-------------|------|------------|
| K | Gd | Girder Depth | Continuous | Gd = L/20 |
| N | S  | Lateral Spacing Between Girders | Continuous | Snap to 25mm |
| O | Ng | Number of Girders | Integer | Round to nearest int |
| P | P  | Bottom Flange Bottom Depth | Continuous | Snap to 25mm |
| Q | Q  | Bottom Flange Bottom Width | Continuous | Snap to 25mm |
| R | Ns | Number of Prestressing Strands | Integer | Round to nearest int |
| S | Hp | Harping Position | Continuous | — |

#### Dataset Generation

- 27 Cost Combinations (3 levels each for Cc, Cs, Cp — full 3³ factorial)
- 5 Span Lengths
- 5 Optimization Runs per combination (to account for stochastic solver variance)
- Total = 27 × 5 × 5 = **675 samples**

---

## Feature Engineering (Phase 1)

These engineered features must be added to the raw inputs before training or inference:

| Feature Name | Formula | Physical Meaning |
|---|---|---|
| `L_squared` | L² | Bending moment proxy (Mu ? L²) |
| `ratio_p_c` | Cp / Cc | Strand-to-concrete cost ratio |
| `ratio_s_c` | Cs / Cc | Rebar-to-concrete cost ratio |
| `interaction_pL` | (Cp / Cc) × L² | Strand cost penalty on long spans |

**Final feature vector (8 features):**
```
[Cc, Cs, Cp, L, L², Cp/Cc, Cs/Cc, (Cp/Cc)×L²]
```

All features must pass through a fitted `StandardScaler` or `RobustScaler` before being fed into any gradient-based model (XGBoost, MLP).

---

## Preprocessing Rules (Phase 2)

- **Train/Test Split:** 80/20 ? 540 train / 135 test
- **Stratification:** Stratify on all 5 span length bins AND cost combination groups to prevent data leakage
- **Scaler:** Fit scaler ONLY on training data; transform test data using the fitted scaler
- **No target scaling** required unless MLP loss is unstable

---

## Model Architecture (Phase 3)

### Candidate Models (evaluate all three)

#### 1. MultiOutput XGBoost Regressor
```python
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

model = MultiOutputRegressor(
    XGBRegressor(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
)
```
Tune with Optuna: max_depth ? [3,8], lr ? [0.01,0.1], n_estimators ? [100,500].

#### 2. MultiOutput Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
```

#### 3. Multi-Task MLP (PyTorch)
- Shared encoder: Linear(8 ? 128) ? ReLU ? Linear(128 ? 64) ? ReLU
- 7 separate output heads: Linear(64 ? 1) each
- Loss: Sum of MSE losses across all 7 heads

### Evaluation Metrics (per target output)

| Metric | Formula | Target |
|--------|---------|--------|
| R² Score | 1 - SS_res/SS_tot | > 0.98 |
| RMSE | sqrt(mean((y-y)²)) | < 2% of range |
| MAPE | mean(|y-y|/y) × 100 | Minimize |

---

## Symbolic Regression (Phase 4)

Two parallel approaches:

### PySR
- Runs genetic algorithm over algebraic operators (+, ×, /, sqrt, log)
- Outputs explicit equations per target
- Use `pysr.PySRRegressor` with `niterations=100`, `maxsize=20`

### Response Surface Methodology (RSM)
Second-order polynomial model across the 3³ factorial design:

```
y_k = ß0 + S ßi·xi + S ßii·xi² + S ßij·xi·xj
      (i=1..4)       (i=1..4)     (i<j)
```

Fit using `statsmodels.formula.api.ols` or `sklearn.preprocessing.PolynomialFeatures`.

---

## Physical Constraint Enforcement (Phase 5)

All ML predictions must be post-processed before returning to the user:

```python
def enforce_constraints(pred: dict, L: float) -> dict:
    # Integer rounding for discrete parameters
    pred["Ng"] = int(round(pred["Ng"]))
    pred["Ns"] = int(round(pred["Ns"]))

    # Snap continuous dimensions to 25mm construction increments
    for key in ["Gd", "S", "P", "Q"]:
        pred[key] = round(pred[key] / 25) * 25

    # AASHTO hard bounds
    assert pred["Gd"] >= L / 20, "Girder depth violates L/20 minimum"
    # Add minimum web thickness check: Ww >= 150mm

    return pred
```

---

## Inference API (Phase 6)

### FastAPI Endpoint

```
POST /predict
Content-Type: application/json

{
  "Cc": 150.0,
  "Cs": 800.0,
  "Cp": 1200.0,
  "L": 30.0
}

Response:
{
  "Gd": 1750,
  "S": 2000,
  "Ng": 5,
  "P": 200,
  "Q": 600,
  "Ns": 42,
  "Hp": 0.4
}
```

### Client-Side Alternative
Derived closed-form equations (from PySR/RSM) can be embedded directly in JavaScript — no backend required for basic deployments.

---

## Web UI Requirements (Phase 6)

- **Inputs:** Sliders or numeric fields for Cc, Cs, Cp, L
- **Outputs:** Numeric result cards for all 7 design parameters
- **Visualization:** SVG canvas of the I-girder cross-section that dynamically:
  - Adjusts beam height (Gd)
  - Adjusts flange width/depth (P, Q)
  - Renders strand positions (Ns dots)
  - Updates harping position (Hp)

---

## Coding Conventions

- All scripts in `src/` must be importable as Python modules (no bare scripts)
- Feature engineering must be encapsulated in a `build_features(X_raw: pd.DataFrame) -> pd.DataFrame` function
- Scaler and model artifacts saved as `.pkl` using `joblib.dump`
- All random seeds set to `42` for reproducibility
- Type hints required on all public functions

---

## Key Constraints & Gotchas

- **Never fit the scaler on test data** — always fit on train, then transform both
- **Ng and Ns are discrete** — always round model outputs before reporting
- **Dimensions must snap to 25mm grid** — raw float predictions are not directly usable
- **AASHTO minimum depth:** Gd = L/20 must be enforced in post-processing, not the model
- **Stochastic data:** 5 runs per combination means the dataset has natural variance — the model predicts the "expected optimum" not a single deterministic value
