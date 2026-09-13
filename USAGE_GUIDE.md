# User Usage Guide
## Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

---

## 1. System Architecture

This framework replaces expensive iterative structural optimizers (Differential Evolution, Box Complex) with a **Surrogate ML Pipeline**. Given material unit costs and span length, it instantly predicts optimal prestressed concrete bridge I-girder design parameters.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER INPUTS                                 │
│   Concrete ($/yd³) | Strand ($/ft/strand) | Rebar ($/lb) | Span (ft)   │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│          Phase 0-1 · DATA LOADING & FEATURE ENGINEERING                  │
│  load_dataset_averaged() -> 119 unique cost-span families                │
│  11 features: [Cc, Cp, Cs, L, L^2, L^3, cost-ratios, interactions]       │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           Phase 3 · RANDOMFOREST SURROGATE MODEL                         │
│   Predicts 7 structural design targets (Mean R2 = 0.630, CV R2 = 0.605)  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           Phase 5 · PHYSICAL CONSTRAINT ENFORCEMENT                      │
│   Gd in [45, 72] in  |  Ns = even int [32,122]  |  Ng in [6,13]        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           Phase 6 · FastAPI BACKEND + WEB UI                             │
│   POST /predict  |  Dynamic SVG I-girder cross-section                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase Breakdown

### Phase 0 — Data Loading (`src/data/load_data.py`)
- Reads all 5 span sheets (`100`, `120`, `140`, `160`, `180` ft) from `Girder_Dataset.xlsx`.
- Strips trailing spaces from column names (`'Strand '` → `'Strand'`).
- Filters contamination rows where `Rebar == 1.26` and solver-failure outliers where `No. of Gir > 20`.
- `load_dataset()` → 595 raw rows.
- `load_dataset_averaged()` → **119 rows** (5 stochastic optimizer runs per combination averaged into one deterministic row per unique cost-span combination).

> **Why average?** The raw 595 rows contain 5 optimizer runs per cost-span combination. Training on raw rows causes 100% train/test leakage — every test combination also exists in training — so R² measures noise prediction, not signal. Averaging produces genuine, unseen test combos and yields true generalization performance.

### Phase 1 — Feature Engineering (`src/features/build_features.py`)
Expands 4 raw inputs into **11 validated domain-engineered features** (reduced from 18 after engineering review):

| Feature Name | Formula | Rationale |
|---|---|---|
| `Concrete` | Cc | Raw concrete unit cost ($/yd³) |
| `Strand` | Cp | Raw strand unit cost ($/ft/strand) |
| `Rebar` | Cs | Raw rebar unit cost ($/lb) |
| `Span_ft` | L | Bridge span length (ft) |
| `L_sq` | Span_ft ** 2 | Bending moment proxy ($M_u \propto wL^2$) |
| `L_cube` | Span_ft ** 3 | Nonlinear deflection/demand proxy (cubic span) |
| `ratio_strand_concrete` | Cp / Cc | Strand-to-concrete unit cost ratio |
| `ratio_rebar_concrete` | Cs / Cc | Rebar-to-concrete unit cost ratio |
| `ratio_rebar_strand` | Cs / Cp | Cross-steel unit cost ratio |
| `interaction_strand_L` | (Cp / Cc) * Span_ft² | Strand cost × span demand (L² only) |
| `cost_interaction` | (Cp * Cs) / Cc² | Composite steel cost interaction |

### Phase 2 — Preprocessing (`src/preprocessing/split_and_scale.py`)
- Uses `load_dataset_averaged()` → 119 unique cost-span combination rows.
- Stratified 80/20 split by `Span_ft` → **95 train / 24 test** combos.
- Fits `StandardScaler` on 11 training features only → `models/scaler.pkl`.

### Phase 3 — Model Training (`src/models/train.py`)
- Benchmarks RandomForest, GradientBoosting, and XGBoost.
- **Primary model: RandomForest** (achieves Mean R² = 0.630 on test set; XGBoost achieves 0.519 on the 95-row dataset due to small-sample tree variance).
- Optuna (15 trials, 5-fold CV) tunes RF hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- Saves `models/best_model.pkl`, `reports/model_benchmark.json`, `reports/scatter_plots/`.

**Achieved Performance on Test Set (24 Unseen Combos):**

| Target | Test R² | Test RMSE | Status |
|---|---|---|---|
| Number of strand per girder | **0.946** | 4.30 strands | Excellent |
| Lateral Spacing (ft) | **0.843** | 0.44 ft | Excellent |
| Number of Girders | **0.823** | 0.64 girders | Excellent |
| Harping Position (ft) | **0.750** | 1.96 ft | Strong |
| Girder Depth (in) | **0.524** | 1.88 in | Good |
| Bottom Flange Width (in) | **0.483** | 2.45 in | Moderate |
| Bottom Flange Depth (in) | **0.040** | 0.35 in | Flat target (~7.5 in constant) |
| **Overall Mean** | **0.630** | — | **CV R²: 0.605** |

### Phase 4 — RSM Equations (`src/equations/train_rsm_benchmark.py`)
- **All 7 targets** (including $N_g$ and $H_p$) are **regressed via 3rd-order LassoCV**.
- Fixed formulas (`Ng = round(37/S)+1` and `Hp = 0.40*L`) were both removed — $N_g$ assumed an unconfirmed 37-ft deck width, while $H_p$ depends on tendon geometry and harping angle constraints (e.g. 1:7 slope limit).
- LassoCV auto-prunes near-zero cross-terms — equations shrink from 35 terms → 1–19 terms per target.
- Mean R² improved from 0.32 (old 2nd-order linear) → **0.56** (LassoCV 3rd-order, all 7 targets regressed).
- Outputs `reports/equations/rsm_equations.json`.

### Phase 5 — Constraint Enforcement (`src/postprocess/constraints.py`)
Post-processes raw ML predictions to enforce physical bounds:
- **Girder Depth:** Snapped to 0.5 in increments, clamped to **[45.0, 72.0] in** (AASHTO Type VI standard beam depth limits observed in dataset).
- **Strands (Ns):** Rounded to nearest even integer, clamped to [32, 122].
- **Girder Count (Ng):** Rounded to nearest integer, clamped to [6, 13].
- **Flange depth/width:** Snapped to 0.5 in increments.
- **Lateral spacing:** Snapped to 0.25 ft increments.
- **Harp position:** Clamped to [0, L].

### Phase 6 — API + Web UI (`api/main.py`, `ui/`)
- FastAPI endpoints: `POST /predict`, `GET /health`, `GET /equations`, `GET /` (redirects to UI).
- Web UI: Dark glassmorphism dashboard with real-time parametric SVG I-girder rendering.
- Client-side RSM fallback activates automatically when backend is offline.

---

## 3. Running the Project — All Commands in Order

### Prerequisites
- Python 3.10+
- `Girder_Dataset.xlsx` must be in the **project root folder**

---

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 2 — Verify Data Loading *(optional check)*
```bash
python src/data/load_data.py
```
Expected output:
```
[Phase 0] Data loading complete. Clean dataset shape: (595, 12)
Saved to data/processed/clean_dataset.csv
```

---

### Step 3 — Build Features *(optional check)*
```bash
python src/features/build_features.py
```
Expected output:
```
[Phase 1] Feature engineering complete. Features shape: (595, 11)
Features list: ['Concrete', 'Strand', 'Rebar', 'Span_ft', 'L_sq', 'L_cube', 'ratio_strand_concrete', ...]
```

---

### Step 4 — Preprocess and Split *(optional check)*
```bash
python src/preprocessing/split_and_scale.py
```
Expected output:
```
  [Preprocessing] Using averaged dataset: 119 unique cost-span combinations
[Phase 2] Preprocessing & splitting complete.
X_train shape: (95, 11), X_test shape: (24, 11)
Fitted scaler saved to models/scaler.pkl
```

---

### Step 5 — Train the Model  ← MAIN COMMAND (takes ~5–10 min)
```bash
python src/models/train.py
```
Expected output:
```
  [Preprocessing] Averaged primary dataset: 119 unique cost-span families (noise reduced via mean of 5 optimizer runs)

--- 1. Baseline Model Benchmarking ---
Model: RandomForest         | Test Mean R2: 0.58991
Model: GradientBoosting     | Test Mean R2: 0.53903
Model: XGBoost_Default      | Test Mean R2: 0.51908

--- 2. Hyperparameter Optimization (RandomForest — best on small averaged data) ---
Starting Optuna hyperparameter optimization (15 trials)...
Optuna tuning finished. Best Trial Per-Target Mean CV-R2: 0.60513
Best Hyperparameters: {'n_estimators': 1000, 'max_depth': 25, 'min_samples_split': 14, 'min_samples_leaf': 4, 'max_features': 'sqrt'}

--- 3. Training Best Tuned RandomForest Model ---

=======================================================
FINAL TUNED RANDOMFOREST TEST MEAN R2: 0.62987
=======================================================
Target: Gir Dep (in)                   | R2 = 0.52409  | RMSE = 1.88456
Target: Lat Spac (ft)                  | R2 = 0.84279  | RMSE = 0.44374
Target: No. of Gir                     | R2 = 0.82298  | RMSE = 0.63842
Target: bot flange bot part depth (in) | R2 = 0.04018  | RMSE = 0.35338
Target: bot flange bot part width (in) | R2 = 0.48270  | RMSE = 2.45398
Target: Number of strand per girder    | R2 = 0.94589  | RMSE = 4.29633
Target: Harp Pos (ft)                  | R2 = 0.75046  | RMSE = 1.95666
```

Files created:
```
models/scaler.pkl           <- Fitted StandardScaler
models/best_model.pkl       <- Tuned RandomForest
reports/model_benchmark.json
reports/model_comparison.csv
reports/scatter_plots/*.png
```

---

### Step 6 — Generate RSM Equations *(optional)*
```bash
python -m src.equations.train_rsm_benchmark
```
Fits Sparse LassoCV 3rd-order RSM with normalized inputs. Creates `reports/equations/rsm_equations.json`.
Physics-derived formulas used for `No. of Gir` and `Harp Pos (ft)` instead of polynomial regression.

---

### Step 7 — Run Tests *(optional)*
```bash
python -m pytest tests/ -v
```

---

### Step 8 — Start the API Server
```bash
python api/main.py
```
Or with auto-reload for development:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

### Step 9 — Open the Web UI
Open `ui/index.html` directly in any browser, **or** navigate to `http://localhost:8000/` when the server is running.

**Input ranges:**
| Input | Range | Unit |
|---|---|---|
| Concrete Unit Cost | 405 – 600 | $/yd³ |
| Strand Unit Cost | 1.26 – 2.23 | $/linear ft per strand |
| Rebar Unit Cost | 2.18 – 3.45 | $/lb |
| Bridge Span Length | 100 – 180 | ft |

The SVG cross-section updates in real time. Click **Export Report** to download a CSV.

> **Offline fallback:** If the backend is not running, the UI shows "Client-Side RSM Solver (Offline)" and uses calibrated analytical formulas instead.

---

## 4. Quick Reference — Minimal Command Set

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model  (the only required step before running the server)
python src/models/train.py

# 3. (Optional) Generate RSM equations
python -m src.equations.train_rsm_benchmark

# 4. Start backend server
python api/main.py
```

Then open `http://localhost:8000/` in your browser.

---

## 5. API Request / Response Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Concrete": 505.0, "Strand": 1.73, "Rebar": 2.18, "Span_ft": 140.0}'
```

**Response:**
```json
{
  "Gir_Dep_in": 69.5,
  "Lat_Spac_ft": 6.25,
  "No_of_Gir": 7,
  "bot_flange_depth_in": 7.5,
  "bot_flange_width_in": 38.0,
  "Number_of_strands": 70,
  "Harp_Pos_ft": 49.5
}
```

> Note: `Gir_Dep_in` will always be in [45.0, 72.0] in. `Number_of_strands` will always be an even integer in [32, 122].

---

## 6. Project File Layout

```
project-root/
├── Girder_Dataset.xlsx              <- Source data (must be in root)
├── requirements.txt
│
├── api/
│   └── main.py                      <- FastAPI server (Phase 6)
│
├── src/
│   ├── data/
│   │   └── load_data.py             <- load_dataset() + load_dataset_averaged()
│   ├── features/
│   │   └── build_features.py        <- 11 validated domain-engineered features
│   ├── preprocessing/
│   │   └── split_and_scale.py       <- Averaged 95/24 split + scaler (11 features)
│   ├── models/
│   │   └── train.py                 <- RF benchmark + Optuna tuning
│   ├── equations/
│   │   └── train_rsm_benchmark.py   <- Sparse LassoCV RSM equations
│   └── postprocess/
│       └── constraints.py           <- Physical bounds enforcement
│
├── models/                          <- Created by train.py
│   ├── scaler.pkl
│   └── best_model.pkl
│
├── reports/                         <- Created by train.py + train_rsm_benchmark.py
│   ├── model_benchmark.json
│   ├── model_comparison.csv
│   ├── scatter_plots/
│   └── equations/
│       ├── rsm_equations.json
│       └── rsm_equations.md
│
├── ui/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── tests/
│   └── test_pipeline.py
│
├── CONTEXT.md                       <- Authoritative agent/dev context
├── EQUATIONS.md                     <- Full math reference
└── USAGE_GUIDE.md                   <- This file
```
