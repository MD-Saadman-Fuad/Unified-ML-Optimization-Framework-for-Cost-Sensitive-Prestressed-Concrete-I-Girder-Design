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
│  load_dataset_averaged() -> 119 unique cost-span combos                  │
│  8 features: [Cc, Cp, Cs, L, L^2, Cp/Cc, Cs/Cc, (Cp/Cc)*L^2]          │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│           Phase 3 · RANDOMFOREST SURROGATE MODEL                         │
│   Predicts 7 structural design targets  (CV Mean R2 ~ 0.56)             │
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

> **Why average?** The raw 595 rows contain 5 optimizer runs per cost-span combination. Training on raw rows causes 100% train/test leakage — every test combination also exists in training — so R² measures noise prediction, not signal. Averaging produces genuine, unseen test combos and raises Mean R² from 0.22 → 0.56.

### Phase 1 — Feature Engineering (`src/features/build_features.py`)
Expands 4 raw inputs into 8 domain-engineered features:

| Feature | Formula | Rationale |
|---|---|---|
| `Concrete` | Cc | Raw concrete cost |
| `Strand` | Cp | Raw strand cost |
| `Rebar` | Cs | Raw rebar cost |
| `Span_ft` | L | Raw span length |
| `L_sq` | L² | Bending moment proxy (Mu proportional to L²) |
| `ratio_strand_concrete` | Cp/Cc | Relative strand-to-concrete cost |
| `ratio_rebar_concrete` | Cs/Cc | Relative rebar-to-concrete cost |
| `interaction_strand_L` | (Cp/Cc) * L² | Strand cost penalty on long spans |

### Phase 2 — Preprocessing (`src/preprocessing/split_and_scale.py`)
- Uses `load_dataset_averaged()` → 119 unique cost-span combination rows.
- Stratified 80/20 split by `Span_ft` → **95 train / 24 test** combos.
- Fits `StandardScaler` on training features only → `models/scaler.pkl`.

### Phase 3 — Model Training (`src/models/train.py`)
- Benchmarks RandomForest, GradientBoosting, and XGBoost.
- **Primary model: RandomForest** (wins on 95 training rows; XGBoost overfits without heavy regularization on small data).
- Optuna (30 trials, 5-fold CV) tunes RF hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- Saves `models/best_model.pkl`, `reports/model_benchmark.json`, `reports/scatter_plots/`.

**Expected 5-fold CV R² after training:**

| Target | CV R² |
|---|---|
| Number of strand per girder | ~0.96 |
| Lateral Spacing | ~0.83 |
| Number of Girders | ~0.75 |
| Harp Position | ~0.66 |
| Bot Flange Width | ~0.49 |
| Girder Depth | ~0.37 |
| Bot Flange Depth | ~-0.14 *(inherently low — near-zero correlation with cost inputs)* |
| **Mean** | **~0.56** |

### Phase 4 — RSM Equations (`src/equations/rsm_fit.py`)
- Fits 2nd-order polynomial surfaces on the averaged 119-row dataset.
- Outputs `reports/equations/rsm_equations.json` and `reports/equations/rsm_equations.md`.

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
[Phase 1] Feature engineering complete. Features shape: (595, 8)
Features list: ['Concrete', 'Strand', 'Rebar', 'Span_ft', 'L_sq', ...]
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
X_train shape: (95, 8), X_test shape: (24, 8)
Fitted scaler saved to models/scaler.pkl
```

---

### Step 5 — Train the Model  ← MAIN COMMAND (takes ~5–10 min)
```bash
python src/models/train.py
```
Expected output:
```
  [Preprocessing] Using averaged dataset: 119 unique cost-span combinations

--- 1. Baseline Model Benchmarking ---
Model: RandomForest         | Test Mean R2: ~0.578
Model: GradientBoosting     | Test Mean R2: ~0.359
Model: XGBoost_Default      | Test Mean R2: ~0.376

--- 2. Hyperparameter Optimization (RandomForest) ---
Starting Optuna hyperparameter optimization (30 trials)...

--- 3. Training Best Tuned RandomForest Model ---
FINAL TUNED RANDOMFOREST TEST MEAN R2: ~0.58+
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
python src/equations/rsm_fit.py
```
Creates `reports/equations/rsm_equations.json` and `rsm_equations.md`.

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
python src/equations/rsm_fit.py

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
│   │   └── build_features.py        <- 8 domain-engineered features
│   ├── preprocessing/
│   │   └── split_and_scale.py       <- Averaged 95/24 split + scaler
│   ├── models/
│   │   └── train.py                 <- RF benchmark + Optuna tuning
│   ├── equations/
│   │   └── rsm_fit.py               <- 2nd-order RSM polynomial fitting
│   └── postprocess/
│       └── constraints.py           <- Physical bounds enforcement
│
├── models/                          <- Created by train.py
│   ├── scaler.pkl
│   └── best_model.pkl
│
├── reports/                         <- Created by train.py + rsm_fit.py
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
