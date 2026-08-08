# Project Roadmap — Unified ML Optimization Framework

> Structured 6-Phase Engineering Pipeline
> Target: R2 > 0.98, RMSE < 2% across all 7 structural design outputs
> All units are imperial (inches, feet, $/yd3, $/lb) throughout the entire pipeline.

---

## Phase Overview

```
[Phase 0: Data Loading & Cleaning]
               |
               v
[Phase 1: EDA & Feature Engineering]
               |
               v
[Phase 2: Validation Strategy & Preprocessing]
               |
               v
[Phase 3: Multi-Output Model Benchmark & Tuning]
               |
               v
[Phase 4: Equation Derivation (PySR / RSM)]
               |
               v
[Phase 5: Physics & Bound Constraints Filtering]
               |
               v
[Phase 6: Deployment & Web UI Integration]
```

---

## Phase 0 — Data Loading & Cleaning

**Goal:** Load all 5 sheets from `Girder_Dataset.xlsx`, clean them into one unified DataFrame, and verify integrity.

### Tasks

- [ ] **0.1** Create `src/data/` directory
- [ ] **0.2** Write `src/data/load_data.py` with `load_dataset(filepath: str) -> pd.DataFrame`:
  - Load all 5 sheets with correct header row per sheet:
    - Sheet `100`: header row index 3
    - Sheet `120`: header row index 2
    - Sheet `140`: header row index 3
    - Sheet `160`: header row index 2
    - Sheet `180`: header row index 3
  - After loading each sheet, call `df.columns = df.columns.str.strip()` to remove trailing spaces
  - Add `Span_ft` column from the sheet name (integer: 100, 120, 140, 160, 180)
  - Concatenate all 5 sheets into one DataFrame
- [ ] **0.3** Select only the 11 required columns: `Concrete`, `Strand`, `Rebar`, `Span_ft`, `Combo`, `Gir Dep (in)`, `Lat Spac (ft)`, `No. of Gir`, `bot flange bot part depth (in)`, `bot flange bot part width (in)`, `Number of strand per girder`, `Harp Pos (ft)`
- [ ] **0.4** Drop rows with any NaN in the 4 input columns or 7 target columns
- [ ] **0.5** Drop the 5 outlier rows where `No. of Gir > 20` (these are failed solver runs in sheet `160`)
- [ ] **0.6** Assert final row count is 670
- [ ] **0.7** Assert each span appears exactly: 100=135, 120=135, 140=135, 160=130, 180=135
- [ ] **0.8** Assert `No. of Gir` unique values are all integers in range [6, 13]
- [ ] **0.9** Assert `Number of strand per girder` values are all even integers in range [32, 122]
- [ ] **0.10** Save cleaned DataFrame to `data/processed/clean_dataset.csv`

**Deliverable:** `src/data/load_data.py`, `data/processed/clean_dataset.csv`

---

## Phase 1 — EDA & Feature Engineering

**Goal:** Understand the data distributions and correlations, then build the full 8-feature input matrix.

### Tasks

#### Exploratory Data Analysis

- [ ] **1.1** Open `notebooks/01_eda.ipynb`
- [ ] **1.2** Load clean dataset using `load_dataset()`
- [ ] **1.3** Print descriptive stats (`df.describe()`) for all 11 columns
- [ ] **1.4** Plot distribution histograms for all 4 inputs and 7 targets (use matplotlib/seaborn)
- [ ] **1.5** Plot correlation heatmap between all inputs and targets
- [ ] **1.6** Group by `Span_ft` and plot how each target varies across spans
- [ ] **1.7** Group by `Concrete` level and verify expected cost-driven design trends
- [ ] **1.8** Flag and investigate any `bot flange bot part depth (in)` = 0.0 rows — decide whether to drop or keep
- [ ] **1.9** Filter out raw data contamination (exclude rows where Rebar = 1.26); verify valid Rebar cost range [2.18, 3.45]

#### Feature Engineering

- [ ] **1.10** Write `src/features/build_features.py` with `build_features(df: pd.DataFrame) -> pd.DataFrame`:
  - Add `L_sq` = `Span_ft ** 2`
  - Add `ratio_strand_concrete` = `Strand / Concrete`
  - Add `ratio_rebar_concrete` = `Rebar / Concrete`
  - Add `interaction_strand_L` = `ratio_strand_concrete * L_sq`
- [ ] **1.11** Unit test `build_features()` with at least 3 known input rows and hand-calculated expected outputs
- [ ] **1.12** Save engineered feature matrix to `data/processed/features.csv`

**Final X matrix (8 features):**
`[Concrete, Strand, Rebar, Span_ft, L_sq, ratio_strand_concrete, ratio_rebar_concrete, interaction_strand_L]`

**Deliverable:** `notebooks/01_eda.ipynb`, `src/features/build_features.py`, `data/processed/features.csv`

---

## Phase 2 — Data Preprocessing & Validation Strategy

**Goal:** Produce a reproducible, leakage-free train/test split with a properly fitted scaler.

### Tasks

- [ ] **2.1** Write `src/preprocessing/split_and_scale.py`
- [ ] **2.2** Perform 80/20 stratified split using `train_test_split` with `stratify=df['Span_ft']`
  - Expected: ~536 train rows, ~134 test rows
- [ ] **2.3** Verify split balance: each span value appears in both train and test sets
- [ ] **2.4** Fit `StandardScaler` on training features ONLY (all 8 engineered features)
- [ ] **2.5** Transform both train and test using the fitted scaler
- [ ] **2.6** Save fitted scaler to `models/scaler.pkl` using `joblib.dump`
- [ ] **2.7** Save train/test index arrays to `data/splits/train_idx.npy` and `data/splits/test_idx.npy`
- [ ] **2.8** Write unit test: assert that no test-set row index appears in the training set

**Deliverable:** `src/preprocessing/split_and_scale.py`, `models/scaler.pkl`

---

## Phase 3 — Multi-Output Model Benchmark & Hyperparameter Tuning

**Goal:** Train and compare three model architectures; tune and select the best performer.

### Tasks

#### Baseline Benchmark

- [ ] **3.1** Train `MultiOutputRegressor(XGBRegressor(random_state=42))` with default parameters
- [ ] **3.2** Train `MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=42))` with defaults
- [ ] **3.3** Train Multi-Task MLP (PyTorch): shared encoder (8->128->64) + 7 output heads
- [ ] **3.4** Evaluate all three models on test set; compute R2, RMSE, MAPE for each of the 7 targets independently
- [ ] **3.5** Log all baseline results to `reports/baseline_benchmark.csv`

#### Hyperparameter Tuning — XGBoost (Primary)

- [ ] **3.6** Set up Optuna study for XGBoost with 100 trials
  - `max_depth` in [3, 8]
  - `learning_rate` in [0.01, 0.1]
  - `n_estimators` in [100, 500]
  - `subsample` in [0.7, 1.0]
  - `colsample_bytree` in [0.7, 1.0]
- [ ] **3.7** Retrain XGBoost with best found hyperparameters; evaluate on test set
- [ ] **3.8** Compare tuned XGBoost to baseline; log to `reports/xgb_tuning_results.csv`

#### MLP Tuning

- [ ] **3.9** Experiment with hidden sizes: [64,32], [128,64], [256,128]
- [ ] **3.10** Experiment with learning rates: 1e-3, 5e-4, 1e-4
- [ ] **3.11** Add dropout (0.1–0.3) if training loss << validation loss
- [ ] **3.12** Log validation loss curves to `reports/mlp_training_curves.png`

#### Model Selection

- [ ] **3.13** Compare all tuned models across all 7 targets; select the architecture with best average R2
- [ ] **3.14** Save best model: `models/best_model.pkl` (XGBoost/RF) or `models/best_model.pt` (MLP)
- [ ] **3.15** Generate per-target predicted vs actual scatter plots; save to `reports/scatter_plots/`

**Deliverable:** `src/models/train.py`, `models/best_model.pkl`, `reports/model_comparison.csv`

---

## Phase 4 — Symbolic Regression & Explicit Equation Derivation

**Goal:** Extract human-readable closed-form equations for each of the 7 design targets.

### Tasks

#### PySR — Genetic Symbolic Regression

- [ ] **4.1** Install `pysr` package
- [ ] **4.2** For each of the 7 targets, run `PySRRegressor(niterations=100, maxsize=20)` on scaled training data
- [ ] **4.3** Evaluate each PySR equation on test set; record R2 per target
- [ ] **4.4** Export best equation strings to `reports/pysr_equations.md`

#### Response Surface Methodology (RSM)

- [ ] **4.5** Build second-order polynomial features using `PolynomialFeatures(degree=2)` on raw (unscaled) inputs
- [ ] **4.6** Fit `statsmodels.OLS` for each of the 7 targets
- [ ] **4.7** Extract and tabulate all beta coefficients (b0, bi, bii, bij)
- [ ] **4.8** Evaluate RSM equations on test set; record R2 per target
- [ ] **4.9** Export coefficients to `reports/rsm_equations.md`

#### Comparison

- [ ] **4.10** Build summary table: PySR vs RSM vs ML model R2 for each target
- [ ] **4.11** Select best equation set for Web UI embedding (balance complexity vs accuracy)

**Deliverable:** `src/equations/pysr_fit.py`, `src/equations/rsm_fit.py`, `reports/equations/`

---

## Phase 5 — Post-Processing & Physical Constraint Enforcement

**Goal:** Ensure all model predictions are physically valid and AASHTO code-compliant.
All units remain imperial throughout this phase.

### Tasks

- [ ] **5.1** Write `src/postprocess/constraints.py` with `enforce_constraints(pred: dict, L_ft: float) -> dict`:
  - Round `Ng` to nearest integer (valid range: 6–13)
  - Round `Ns` to nearest even integer (valid range: 32–122)
  - Enforce AASHTO minimum depth: `Gd_in >= (L_ft / 20) * 12` (convert L/20 from ft to in)
  - Clip violations (do not raise) — clip to minimum valid value and log warning
- [ ] **5.2** Apply constraint enforcement to all test-set predictions; count and log how many predictions violated each rule
- [ ] **5.3** Log pre/post constraint metrics to `reports/constraint_analysis.csv`
- [ ] **5.4** Unit test all constraint rules with edge cases:
  - Span = 100 ft (shortest)
  - Span = 180 ft (longest)
  - Predicted Ng = 4 (below minimum — should clip to 6)
  - Predicted Ng = 15 (above maximum — should clip to 13)
  - Predicted Ns = 33 (odd — should round to 34)

**Deliverable:** `src/postprocess/constraints.py`, `tests/test_constraints.py`

---

## Phase 6 — Deployment & Web UI Integration

**Goal:** Wrap the trained model in a web-based calculator with live SVG I-girder visualization.

### Tasks

#### FastAPI Backend

- [ ] **6.1** Create `api/main.py` with FastAPI app
- [ ] **6.2** Implement `POST /predict` endpoint:
  - Accept JSON with fields: `Concrete` ($/yd3), `Strand` ($/linear ft per strand), `Rebar` ($/lb), `Span_ft` (ft)
  - Validate inputs: Concrete in [405,600], Strand in [1.26,2.23], Rebar in [2.18,3.45], Span_ft in [100,180]
  - Run: `build_features` -> `scaler.transform` -> `model.predict` -> `enforce_constraints`
  - Return JSON with all 7 predicted values with their units
- [ ] **6.3** Load scaler and model from `.pkl` files at startup using `joblib.load`
- [ ] **6.4** Add Pydantic request model with Field bounds
- [ ] **6.5** Add CORS middleware
- [ ] **6.6** Write integration tests using `httpx.AsyncClient`
- [ ] **6.7** Containerize: `Dockerfile` + `docker-compose.yml`

#### Web Frontend

- [ ] **6.8** Create `ui/index.html` with input controls:
  - Slider + numeric field for `Concrete` (405–600 $/yd3)
  - Slider + numeric field for `Strand` (1.26–2.23 $/lb)
  - Slider + numeric field for `Rebar` (1.26–3.45 $/lb)
  - Slider + numeric field for `Span` (100–180 ft)
- [ ] **6.9** Display all 7 predicted outputs in labeled result cards with units (in / ft / count)
- [ ] **6.10** Implement dynamic SVG I-girder cross-section:
  - Beam height scales proportionally to `Gir Dep (in)`
  - Bottom flange dimensions scale to `bot flange depth` and `bot flange width`
  - Render `Number of strand per girder` dots for strand layout
  - Show `Harp Pos (ft)` as a reference marker line
- [ ] **6.11** Add export-to-CSV button for predicted design parameters
- [ ] **6.12** Test UI on Chrome, Firefox, Edge

**Deliverable:** `api/`, `ui/`, `Dockerfile`

---

## Cross-Cutting Tasks

- [ ] **C.1** Create full directory structure as defined in README
- [ ] **C.2** Write `requirements.txt` with pinned package versions
- [ ] **C.3** Set global `random_state=42` everywhere (sklearn, XGBoost, PyTorch seed)
- [ ] **C.4** Set up `pytest` suite; run tests for Phase 0 through Phase 5 modules
- [ ] **C.5** Add docstrings to all public functions
- [ ] **C.6** Final validation: confirm R2 > 0.98 on hold-out test set for all 7 targets
- [ ] **C.7** Write final research report / paper section from RSM equation coefficients and ML benchmark table

---

## Milestone Summary

| Milestone | Phases | Key Deliverable |
|-----------|--------|-----------------|
| M1 — Data Ready | 0–1 | 670-row clean dataset, 8-feature matrix |
| M2 — Split Ready | 2 | Stratified 536/134 split, fitted scaler saved |
| M3 — Model Ready | 3 | Tuned best model .pkl, R2 > 0.98 |
| M4 — Equations Ready | 4 | Closed-form equations for all 7 targets |
| M5 — Physics-Safe | 5 | Constraint-enforced inference pipeline |
| M6 — Deployed | 6 | Live web app with SVG visualization |
