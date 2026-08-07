# Project Roadmap — Unified ML Optimization Framework

> Structured 6-Phase Engineering Pipeline
> Target: R² > 0.98, RMSE < 2% across all 7 structural design outputs

---

## Phase Overview

```
[Phase 1: Ingestion & Feature Engineering]
               ¦
               ?
[Phase 2: Validation Strategy & Preprocessing]
               ¦
               ?
[Phase 3: Multi-Output Model Benchmark & Tuning]
               ¦
               ?
[Phase 4: Equation Derivation (PySR / RSM)]
               ¦
               ?
[Phase 5: Physics & Bound Constraints Filtering]
               ¦
               ?
[Phase 6: Deployment & Web UI Integration]
```

---

## Phase 1 — Data Ingestion & Feature Engineering

**Goal:** Load raw dataset, validate integrity, and construct the full feature matrix.

### Tasks

- [ ] **1.1** Load raw Excel/CSV dataset (675 rows) into a pandas DataFrame
- [ ] **1.2** Validate row count: assert exactly 675 rows (27 × 5 × 5)
- [ ] **1.3** Validate all 11 expected columns are present (A, B, C, D, K, N, O, P, Q, R, S)
- [ ] **1.4** Check for missing values or NaNs across all columns; log and handle any found
- [ ] **1.5** Check for outliers in target columns using IQR or z-score analysis
- [ ] **1.6** Run Exploratory Data Analysis (EDA):
  - [ ] Distribution plots for all 4 inputs and 7 targets
  - [ ] Correlation heatmap (inputs vs targets)
  - [ ] Pair plots for key variable pairs
- [ ] **1.7** Implement `build_features(X_raw: pd.DataFrame) -> pd.DataFrame` function:
  - [ ] Add `L_squared` = L²
  - [ ] Add `ratio_p_c` = Cp / Cc
  - [ ] Add `ratio_s_c` = Cs / Cc
  - [ ] Add `interaction_pL` = (Cp / Cc) × L²
- [ ] **1.8** Unit test `build_features()` with known input values
- [ ] **1.9** Save engineered feature matrix to `data/features.csv`

**Deliverable:** `src/features/build_features.py`, `notebooks/01_eda.ipynb`

---

## Phase 2 — Data Preprocessing & Validation Strategy

**Goal:** Produce reproducible, leakage-free train/test splits with properly fitted scalers.

### Tasks

- [ ] **2.1** Define stratification groups: combine span length bin + cost combination index as a single group label
- [ ] **2.2** Perform 80/20 stratified train/test split using `sklearn.model_selection.train_test_split` with `stratify=group_label`
- [ ] **2.3** Verify split balance: assert that all span lengths appear in both train and test sets
- [ ] **2.4** Fit `StandardScaler` on training features ONLY (never test data)
- [ ] **2.5** Transform both train and test feature matrices using the fitted scaler
- [ ] **2.6** Save fitted scaler to `models/scaler.pkl` using `joblib.dump`
- [ ] **2.7** Save train/test index splits to `data/splits/train_idx.npy` and `data/splits/test_idx.npy`
- [ ] **2.8** Write unit test: confirm no test-set indices appear in training set

**Deliverable:** `src/preprocessing/split_and_scale.py`, `models/scaler.pkl`

---

## Phase 3 — Multi-Output Model Benchmark & Hyperparameter Tuning

**Goal:** Train, compare, and tune three candidate model architectures; select the best performer.

### Tasks

#### Baseline Benchmark

- [ ] **3.1** Train `MultiOutputRegressor(XGBRegressor())` with default hyperparameters
- [ ] **3.2** Train `MultiOutputRegressor(RandomForestRegressor())` with default hyperparameters
- [ ] **3.3** Train Multi-Task MLP (PyTorch): shared encoder (8?128?64) + 7 output heads
- [ ] **3.4** Evaluate all three models on test set; compute R², RMSE, MAPE per target output
- [ ] **3.5** Log results to `reports/baseline_benchmark.csv`

#### Hyperparameter Tuning (XGBoost — Primary)

- [ ] **3.6** Set up Optuna study for XGBoost hyperparameter search
  - [ ] Tune: `max_depth` ? [3, 8]
  - [ ] Tune: `learning_rate` ? [0.01, 0.1]
  - [ ] Tune: `n_estimators` ? [100, 500]
  - [ ] Tune: `subsample` ? [0.7, 1.0]
  - [ ] Tune: `colsample_bytree` ? [0.7, 1.0]
- [ ] **3.7** Run Optuna study with 100 trials; log all trial results
- [ ] **3.8** Retrain XGBoost with best hyperparameters found
- [ ] **3.9** Evaluate tuned XGBoost on test set; compare to baseline

#### MLP Tuning

- [ ] **3.10** Experiment with hidden layer sizes: [64,32], [128,64], [256,128]
- [ ] **3.11** Experiment with learning rates: 1e-3, 5e-4, 1e-4
- [ ] **3.12** Add dropout (0.1–0.3) and batch normalization layers if overfitting observed
- [ ] **3.13** Log validation loss curves per epoch to `reports/mlp_training_curves.png`

#### Model Selection

- [ ] **3.14** Compare all tuned models on all 7 targets; select best architecture
- [ ] **3.15** Save best model to `models/best_model.pkl` (XGBoost) or `models/best_model.pt` (MLP)
- [ ] **3.16** Generate per-target prediction vs actual scatter plots; save to `reports/scatter_plots/`

**Deliverable:** `src/models/train.py`, `models/best_model.pkl`, `reports/model_comparison.csv`

---

## Phase 4 — Symbolic Regression & Explicit Equation Derivation

**Goal:** Extract human-readable, closed-form equations for each of the 7 design parameters.

### Tasks

#### PySR — Genetic Symbolic Regression

- [ ] **4.1** Install and configure `pysr` package
- [ ] **4.2** For each of the 7 targets, run `PySRRegressor` on training data:
  - Set `niterations=100`, `maxsize=20`
  - Allow operators: +, -, ×, /, sqrt, log, exp
- [ ] **4.3** Evaluate PySR equations on test set; record R² per target
- [ ] **4.4** Export best equation strings to `reports/pysr_equations.md`

#### Response Surface Methodology (RSM)

- [ ] **4.5** Build second-order polynomial features: `PolynomialFeatures(degree=2)`
- [ ] **4.6** Fit OLS regression model (statsmodels) for each of the 7 targets
- [ ] **4.7** Extract and tabulate all ß coefficients: `ß0, ßi, ßii, ßij`
- [ ] **4.8** Evaluate RSM equations on test set; record R² per target
- [ ] **4.9** Export RSM equations and coefficients to `reports/rsm_equations.md`

#### Comparison

- [ ] **4.10** Compare PySR vs RSM accuracy vs ML model for each target
- [ ] **4.11** Select which equation set to embed in web UI (complexity vs accuracy trade-off)

**Deliverable:** `src/equations/pysr_fit.py`, `src/equations/rsm_fit.py`, `reports/equations/`

---

## Phase 5 — Post-Processing & Physical Constraint Enforcement

**Goal:** Ensure all model predictions are physically valid and AASHTO code-compliant.

### Tasks

- [ ] **5.1** Implement `enforce_constraints(pred: dict, L: float) -> dict`:
  - [ ] Round `Ng` and `Ns` to nearest integer
  - [ ] Snap `Gd`, `S`, `P`, `Q` to nearest 25mm increment
  - [ ] Assert `Gd >= L / 20` (AASHTO minimum depth)
  - [ ] Assert minimum web thickness `Ww >= 150 mm`
  - [ ] Handle constraint violations gracefully (clip vs raise)
- [ ] **5.2** Apply `enforce_constraints` to all test-set predictions; quantify how often violations occur
- [ ] **5.3** Log pre/post constraint comparison metrics to `reports/constraint_analysis.csv`
- [ ] **5.4** Unit test all constraint rules with edge case inputs:
  - Very short span (L=20m)
  - Very long span (L=40m)
  - Extreme cost ratios

**Deliverable:** `src/postprocess/constraints.py`, `tests/test_constraints.py`

---

## Phase 6 — Deployment & Web UI Integration

**Goal:** Wrap the trained model in a web-based calculator with live SVG visualization.

### Tasks

#### FastAPI Backend

- [ ] **6.1** Create `api/main.py` with FastAPI app
- [ ] **6.2** Implement `POST /predict` endpoint:
  - Accept JSON: `{ Cc, Cs, Cp, L }`
  - Run `build_features` ? scale ? model inference ? `enforce_constraints`
  - Return JSON: `{ Gd, S, Ng, P, Q, Ns, Hp }`
- [ ] **6.3** Load scaler and model from `.pkl` files at startup
- [ ] **6.4** Add input validation with Pydantic (min/max bounds for each input)
- [ ] **6.5** Add CORS middleware for frontend requests
- [ ] **6.6** Write API integration tests using `httpx.AsyncClient`
- [ ] **6.7** Containerize with Docker: `Dockerfile` + `docker-compose.yml`

#### Web Frontend

- [ ] **6.8** Create `ui/index.html` with input controls:
  - Sliders + numeric inputs for Cc, Cs, Cp, L
  - Submit button triggers API call or runs JS equations
- [ ] **6.9** Display all 7 predicted design parameter values in result cards
- [ ] **6.10** Implement dynamic SVG canvas for I-girder cross-section visualization:
  - [ ] Draw I-girder outline (top flange, web, bottom flange)
  - [ ] Scale height proportional to `Gd`
  - [ ] Scale bottom flange dimensions to `P` and `Q`
  - [ ] Render `Ns` dots representing strand positions
  - [ ] Show `Hp` as a reference line for harping point
- [ ] **6.11** Add cost breakdown panel: compute total estimated cost using input unit costs and predicted quantities
- [ ] **6.12** Add export-to-PDF or export-to-CSV functionality for design report

#### Polish & Testing

- [ ] **6.13** Test UI on Chrome, Firefox, and Edge
- [ ] **6.14** Validate SVG renders correctly at all span length bounds (L=20 to L=40)
- [ ] **6.15** Confirm API response time < 200ms for XGBoost model
- [ ] **6.16** Write README usage instructions for the web app

**Deliverable:** `api/`, `ui/`, `Dockerfile`

---

## Cross-Cutting Tasks

- [ ] **C.1** Set up project directory structure as defined in README
- [ ] **C.2** Create `requirements.txt` with pinned dependencies
- [ ] **C.3** Set up `pytest` test suite; add tests for each phase module
- [ ] **C.4** Set global random seed = 42 across all scripts
- [ ] **C.5** Add inline docstrings to all public functions
- [ ] **C.6** Final validation: confirm R² > 0.98 on hold-out test set for all 7 targets
- [ ] **C.7** Write final research report / paper draft from RSM equations and benchmark results

---

## Milestone Summary

| Milestone | Phases | Key Deliverable |
|-----------|--------|-----------------|
| M1 — Data Ready | 1–2 | Engineered features, clean split, fitted scaler |
| M2 — Model Ready | 3 | Tuned best model saved as .pkl, R² > 0.98 |
| M3 — Equations Ready | 4 | Closed-form equations for all 7 targets |
| M4 — Physics-Safe | 5 | Constraint-enforced inference pipeline |
| M5 — Deployed | 6 | Live web app with SVG visualization |
