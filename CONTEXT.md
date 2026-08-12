# Agent Context — Unified ML Optimization Framework for Prestressed Concrete I-Girder Design

> This document is the authoritative context for AI agents and developers working on this codebase.
> Read this fully before making any code changes.

---

## Project Goal

Build a 6-phase surrogate ML pipeline that maps 4 material cost + span length inputs directly to 7 structural design parameters for a prestressed concrete I-girder bridge, bypassing computationally expensive iterative optimization.

---

## Domain Background

### What is a Prestressed Concrete I-Girder?

A structural beam used in bridge construction, made of concrete pre-tensioned with high-strength steel strands. The cross-section is shaped like the letter "I" (flanges + web) and must satisfy AASHTO structural code constraints.

### Why Surrogate Modeling?

Classical optimization (Differential Evolution, EVOP, Box Complex) finds the minimum-cost design but requires full reruns when material costs change. A trained surrogate model returns predictions in milliseconds for any new cost combination.

---

## Data Schema

### Source File: `Girder_Dataset.xlsx`

The file has 5 data sheets (one per span) plus one empty `Sheet3`. Sheet name = span length in feet.

| Sheet | Span (ft) | Header Row Index | Data Rows (clean) |
|-------|-----------|------------------|-------------------|
| `100` | 100 | 3 (4th row) | 135 |
| `120` | 120 | 2 (3rd row) | 135 |
| `140` | 140 | 3 (4th row) | 135 |
| `160` | 160 | 2 (3rd row) | 130 |
| `180` | 180 | 3 (4th row) | 135 |

**Total usable rows: 595** (after dropping outlier rows from sheet `160` where `No. of Gir > 20`, filtering Rebar contamination rows where `Rebar = 1.26`, and dropping any remaining NaN rows in core columns).

---

### Input Features — X_raw (4 columns)

All units are **imperial**. Do not assume SI.

| Exact Column Name (after strip) | Symbol | Description | Unit | Observed Levels |
|----------------------------------|--------|-------------|------|-----------------|
| `Concrete` | Cc | Concrete Unit Cost | $/yd3 | 405, 505, 600 |
| `Strand` | Cp | Prestressing Strand Unit Cost | $/linear ft per strand | 1.26, 1.73, 2.23 |
| `Rebar` | Cs | Steel Rebar Unit Cost | $/lb | 2.18, 2.82, 3.45 |
| `Span_ft` | L | Span Length (added from sheet name) | ft | 100, 120, 140, 160, 180 |

> IMPORTANT: `Strand` and `Rebar` have trailing spaces in the raw Excel file (`'Strand '`, `'Rebar '`). Always call `df.columns = df.columns.str.strip()` immediately after loading.

> CONFIRMED: Rebar has 3 price levels (2.18, 2.82, 3.45). The value 1.26 observed in raw data is a contamination from the Strand column — exclude any rows where Rebar = 1.26. The design is a confirmed 3x3x3 factorial = 27 cost combinations.

---

### Target Outputs — Y_target (7 columns)

| Exact Column Name (after strip) | Symbol | Description | Unit | Type | Post-Processing |
|----------------------------------|--------|-------------|------|------|-----------------|
| `Gir Dep (in)` | Gd | Girder Depth | inches | Continuous | Snap to nearest 0.5 in |
| `Lat Spac (ft)` | S | Lateral Spacing Between Girders | feet | Continuous | As-is |
| `No. of Gir` | Ng | Number of Girders | count | Integer | Round to nearest int |
| `bot flange bot part depth (in)` | P | Bottom Flange Bottom Depth | inches | Continuous | As-is |
| `bot flange bot part width (in)` | Q | Bottom Flange Bottom Width | inches | Continuous | As-is |
| `Number of strand per girder` | Ns | Number of Prestressing Strands | count | Even integer (steps of 2) | Round to nearest even int |
| `Harp Pos (ft)` | Hp | Harping Position | feet | Continuous | As-is |

---

### Columns to IGNORE (present only in some sheets)

| Column | Sheets | Reason |
|--------|--------|--------|
| `Gir + Deck Dep (in)` | 120, 160 | Derived value (Gd + deck thickness). Not a target. |
| `Deck thickness (in)` | 160 | Structural detail not used in model. |
| `Combo` | All | Cost combination index (metadata only). |

---

### Descriptive Statistics (670 clean rows)

| Target | Min | Mean | Max | Notes |
|--------|-----|------|-----|-------|
| `Gir Dep (in)` | 47.6 | 68.0 | 72.0 | Hard-capped at 72.0 in (AASHTO Type VI standard beam depth limit) |
| `Lat Spac (ft)` | 2.3 | 6.1 | 9.3 | Continuous |
| `No. of Gir` | 6 | 7.7 | 13 | Integer, round model output |
| `bot flange depth (in)` | 0.0 | 8.2 | 98.0 | Check if 0-value rows are valid |
| `bot flange width (in)` | 7.9 | 38.4 | 81.1 | Continuous |
| `Number of strand per girder` | 32 | 70.1 | 122 | Always even integers |
| `Harp Pos (ft)` | 34.0 | 49.9 | 88.9 | Continuous |

---

## Data Loading (Phase 0 — must run first)

```python
# src/data/load_data.py

import pandas as pd

SHEET_CONFIG = {
    '100': {'header_row': 3, 'span_ft': 100},
    '120': {'header_row': 2, 'span_ft': 120},
    '140': {'header_row': 3, 'span_ft': 140},
    '160': {'header_row': 2, 'span_ft': 160},
    '180': {'header_row': 3, 'span_ft': 180},
}

INPUT_COLS  = ['Concrete', 'Strand', 'Rebar', 'Span_ft']
TARGET_COLS = [
    'Gir Dep (in)', 'Lat Spac (ft)', 'No. of Gir',
    'bot flange bot part depth (in)', 'bot flange bot part width (in)',
    'Number of strand per girder', 'Harp Pos (ft)'
]

def load_dataset(filepath: str) -> pd.DataFrame:
    frames = []
    for sheet, cfg in SHEET_CONFIG.items():
        df = pd.read_excel(filepath, sheet_name=sheet, header=cfg['header_row'])
        df.columns = df.columns.str.strip()   # remove trailing spaces
        df['Span_ft'] = cfg['span_ft']
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Keep only the columns we need
    keep_cols = INPUT_COLS + TARGET_COLS + ['Combo']
    combined = combined[[c for c in keep_cols if c in combined.columns]]

    # Drop rows with any NaN in core columns
    combined = combined.dropna(subset=INPUT_COLS + TARGET_COLS)

    # Drop outlier rows where No. of Gir is physically impossible (> 20)
    combined = combined[combined['No. of Gir'] <= 20]

    return combined.reset_index(drop=True)
```

---

## Feature Engineering (Phase 1)

These engineered features must be added to the raw inputs before training or inference.
All formulas use the actual imperial-unit values — no unit conversion is needed.

| Feature Name | Formula | Physical Meaning |
|---|---|---|
| `L_sq` | Span_ft ** 2 | Bending moment proxy (Mu proportional to L2) |
| `ratio_strand_concrete` | Strand / Concrete | Strand-to-concrete cost ratio |
| `ratio_rebar_concrete` | Rebar / Concrete | Rebar-to-concrete cost ratio |
| `interaction_strand_L` | (Strand / Concrete) * Span_ft**2 | Strand cost penalty on long spans |

```python
# src/features/build_features.py

import pandas as pd

INPUT_COLS = ['Concrete', 'Strand', 'Rebar', 'Span_ft']

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[INPUT_COLS].copy()
    X['L_sq'] = X['Span_ft'] ** 2
    X['ratio_strand_concrete'] = X['Strand'] / X['Concrete']
    X['ratio_rebar_concrete']  = X['Rebar']  / X['Concrete']
    X['interaction_strand_L']  = X['ratio_strand_concrete'] * X['L_sq']
    return X

# Final feature vector: 8 columns
# [Concrete, Strand, Rebar, Span_ft, L_sq, ratio_strand_concrete, ratio_rebar_concrete, interaction_strand_L]
```

---

## Preprocessing Rules (Phase 2)

- **Input data:** Use `load_dataset_averaged()` (not `load_dataset()`). This groups the 595 raw rows by (Concrete, Strand, Rebar, Span_ft) and averages target values, yielding **119 deterministic rows** — one per unique cost-span combination.
- **Why average?** The raw dataset has 5 stochastic optimizer runs per combination. Without averaging, 100% of test combinations also appear in training (pure leakage), and R² measures noise prediction rather than signal. Averaging removes within-combination variance and enables genuine generalization.
- **Train/Test Split:** Stratified 80/20 on 119 averaged rows = **~95 train / ~24 test**. Stratify by `Span_ft`.
- **Scaler:** Fit `StandardScaler` ONLY on training features; transform both train and test using it.
- **No target scaling** required for tree-based models.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

X = build_features(df)
Y = df[TARGET_COLS]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=df['Span_ft']
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.pkl')
```

---

## Model Architecture (Phase 3)

### Candidate Models

#### 1. MultiOutput RandomForest Regressor (Primary — best for 95-row averaged data)
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))
```
Tune with Optuna: n_estimators [200,1000], max_depth [3,20], min_samples_split [2,10], min_samples_leaf [1,5], max_features ['sqrt','log2',None].
5-fold CV R2 on averaged data: Ns≈0.96, LatSpac≈0.83, Ng≈0.75, Hp≈0.66, Q≈0.49, Gd≈0.37, P≈-0.14 (inherently noisy).

#### 2. MultiOutput XGBoost Regressor (Baseline comparison)
```python
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

model = MultiOutputRegressor(
    XGBRegressor(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        random_state=42
    )
)
```
> Note: XGBoost overfits on only 95 training rows unless heavily regularized (reg_lambda ≥ 5). RandomForest is preferred.

#### 3. Multi-Task MLP (PyTorch)
- Shared encoder: Linear(8 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU
- 7 separate output heads: Linear(64 -> 1) each
- Loss: Sum of MSE losses across all 7 heads
- Input size is 8 (the 8-feature engineered vector)

### Evaluation Metrics (per target output)

| Metric | Target |
|--------|--------|
| R2 Score | > 0.98 |
| RMSE | < 2% of each target's range |
| MAPE | Minimize |

---

## Symbolic Regression (Phase 4)

Two parallel approaches to derive explicit equations:

### PySR
```python
from pysr import PySRRegressor
model = PySRRegressor(niterations=100, maxsize=20,
                      binary_operators=["+", "-", "*", "/"],
                      unary_operators=["sqrt", "log", "exp"])
model.fit(X_train_scaled, Y_train['Gir Dep (in)'])
```

### Response Surface Methodology (RSM)
Second-order polynomial across the full input space:
```
y_k = b0 + sum(bi*xi) + sum(bii*xi^2) + sum(bij*xi*xj)   for i=1..4, i<j
```
Fit using `sklearn.preprocessing.PolynomialFeatures(degree=2)` then `statsmodels.OLS`.

---

## Physical Constraint Enforcement (Phase 5)

All ML predictions must pass through constraint enforcement before returning to the user.
All values remain in **imperial units** throughout.

```python
# src/postprocess/constraints.py

def enforce_constraints(pred: dict, L_ft: float) -> dict:
    """
    L_ft: span length in feet
    All dimension inputs/outputs in inches or feet as labeled.
    """
    # Discrete parameters — round to nearest integer
    pred['Ng'] = int(round(pred['Ng']))
    pred['Ns'] = int(round(pred['Ns'] / 2) * 2)   # round to nearest even int

    # AASHTO minimum girder depth: Gd >= 0.045 * L  (both in feet; convert Gd to ft first)
    min_gd_in = 0.045 * L_ft * 12    # 0.045 * L in feet, convert to inches
    if pred['Gd'] < min_gd_in:
        pred['Gd'] = min_gd_in       # clip to minimum, do not raise

    return pred
```

> Note: The 25mm snapping constraint from earlier versions does not apply here — the data is in imperial units, not metric. Snap to the nearest 0.5-inch increment instead if rounding is needed.

---

## Inference API (Phase 6)

### FastAPI Endpoints`n`n- `GET /`: Automatic redirect to `/ui/index.html``n- `GET /health`: Health check (`status`, `scaler_loaded`, `model_loaded`)`n- `GET /equations`: Returns derived RSM equations JSON`n- `POST /predict`: Main ML surrogate prediction endpoint

```
POST /predict
Content-Type: application/json

Request (all imperial units):
{
  "Concrete": 505.0,    # $/yd3
  "Strand":   1.73,     # $/linear ft per strand
  "Rebar":    2.18,     # $/lb
  "Span_ft":  140.0     # feet
}

Response:
{
  "Gir_Dep_in":  75.6,   # inches
  "Lat_Spac_ft":  6.14,  # feet
  "No_of_Gir":    7,     # integer
  "Bot_flange_depth_in": 8.2,   # inches
  "Bot_flange_width_in": 38.4,  # inches
  "Num_strands":  72,    # even integer
  "Harp_Pos_ft": 47.2    # feet
}
```

### Input Validation Bounds (Pydantic)
```python
class PredictRequest(BaseModel):
    Concrete: float = Field(..., ge=405, le=600)   # $/yd3
    Strand:   float = Field(..., ge=1.26, le=2.23) # $/linear ft per strand
    Rebar:    float = Field(..., ge=2.18, le=3.45) # $/lb
    Span_ft:  float = Field(..., ge=100, le=180)   # feet
```

---

## Web UI Requirements (Phase 6)

- **Input controls:** Sliders or numeric fields for Concrete ($/yd3, range 405-600), Strand ($/linear ft per strand, range 1.26-2.23), Rebar ($/lb, range 2.18-3.45), Span (ft, range 100-180)
- **Output cards:** All 7 predicted parameters with their units displayed (in / ft / count)
- **SVG visualization:** Dynamic I-girder cross-section that:
  - Scales beam height to `Gir Dep (in)`
  - Scales bottom flange width/depth to `bot flange bot part width` and `depth`
  - Renders `Number of strand per girder` dots for strand positions
  - Marks `Harp Pos (ft)` as a reference line

---

## Coding Conventions

- All scripts in `src/` must be importable Python modules (no bare scripts with side effects)
- `load_dataset()` lives in `src/data/load_data.py`
- `build_features()` lives in `src/features/build_features.py` with signature `(df: pd.DataFrame) -> pd.DataFrame`
- Model and scaler artifacts saved as `.pkl` using `joblib.dump`
- All random seeds set to `42`
- Type hints required on all public functions

---

## Key Constraints & Gotchas

1. **Never fit the scaler on test data** — always fit on train, transform both.
2. **Column names have trailing spaces in raw Excel** — always strip after loading.
3. **Units are imperial throughout** — inches, feet, $/yd3, $/lb. No SI conversion needed.
4. **Ng is integer, Ns is even integer** — always apply rounding before reporting output.
5. **Outlier rows in sheet 160** — `No. of Gir > 20`; drop them before any processing.
6. **Ignore `Gir + Deck Dep (in)` and `Deck thickness (in)`** — these are derived/extra columns.
7. **Girder depth is capped at 72.0 in** — the optimization used AASHTO Type VI standard beam sections with a 72-inch maximum depth. Do NOT use `0.045 * L * 12` as a minimum depth floor — that formula produces values up to 97.2 in for the 180-ft span which exceeds the dataset maximum. Apply bounds [45.0, 72.0] in for Gd in post-processing.
8. **Stochastic data** — 5 runs per cost+span combination means natural variance. The model predicts the expected optimum, not a single deterministic value.
9. **Span_ft is not a column in the Excel file** — it must be added programmatically from the sheet name during loading.
10. **Strand count is span-dependent** — ranges are 32–50 (100 ft), 42–76 (120 ft), 58–90 (140 ft), 72–116 (160 ft), 68–122 (180 ft). Client-side formulas must reflect this linear-dominant relationship with span, not a quadratic one.
