# Detailed Development Breakdown & User Usage Guide
## Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

---

## 1. Executive Summary & System Architecture

This framework replaces traditional, computationally expensive iterative structural optimization solvers (e.g., Differential Evolution, Box Complex) with a **Surrogate Machine Learning Pipeline**. Given real-time material unit costs and target span length, the system instantly predicts optimal prestressed concrete bridge I-girder cross-section dimensions and prestressing strand configurations.

```
┌─────────────────────────────────────────────────────────┐
│                     USER INPUTS                         │
│  Concrete ($/yd³) | Strand ($/ft) | Rebar ($/lb) | Span (ft) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│           1. DOMAIN FEATURE ENGINEERING                 │
│      L², Strand/Concrete, Rebar/Concrete, (Strand/Cc)*L²│
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        2. MULTI-OUTPUT XGBOOST SURROGATE MODEL          │
│       Predicts 7 structural design target variables     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        3. AASHTO PHYSICS & CODE POST-PROCESSING         │
│  Girder Depth Gd >= L/20 | Even Strands Ns | Ng integer │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│        4. FASTAPI BACKEND & DYNAMIC SVG WEB UI          │
│   Real-time parametric I-girder cross-section rendering  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Breakdown of Engineering Phases

### Phase 0: Data Loading & Cleaning (`src/data/load_data.py`)
- **Multi-Sheet Ingestion:** Reads all 5 span sheets (`100`, `120`, `140`, `160`, `180` ft) from `Girder_Dataset.xlsx` with custom per-sheet header indices.
- **Whitespaces Cleanup:** Calls `.str.strip()` on all column names to resolve trailing space bugs (e.g., `'Strand '`, `'Rebar '`).
- **Data Contamination Filtering:** Removes raw dataset error rows where `Rebar == 1.26` (pricing data entry contamination).
- **Outlier Filtering:** Excludes solver failure rows where `No. of Gir > 20` (valid structural range is 6 to 13 girders).
- **Output:** Exports clean dataset (`595` rows × 12 columns) to `data/processed/clean_dataset.csv`.

### Phase 1: Domain-Driven Feature Engineering (`src/features/build_features.py`)
Encapsulates 8 domain-engineered physical features based on structural mechanics:
1. `Concrete`: Concrete unit cost ($/yd³)
2. `Strand`: Prestressing strand unit cost ($/linear ft per strand)
3. `Rebar`: Steel rebar unit cost ($/lb)
4. `Span_ft`: Bridge span length (ft)
5. `L_sq`: Span length squared ($L^2$), representing bending moment demand ($M_u \propto L^2$)
6. `ratio_strand_concrete`: Strand-to-concrete cost ratio ($\frac{C_p}{C_c}$)
7. `ratio_rebar_concrete`: Rebar-to-concrete cost ratio ($\frac{C_s}{C_c}$)
8. `interaction_strand_L`: Long-span strand cost penalty term ($\frac{C_p}{C_c} \times L^2$)

### Phase 2: Stratified Preprocessing & Scaler Strategy (`src/preprocessing/split_and_scale.py`)
- **Stratified Split:** 80% train / 20% test split stratified on `Span_ft` (`random_state=42`) to ensure balanced representation across all span lengths with zero data leakage.
- **Scaler Isolation:** Fits `StandardScaler` on training features ONLY, saving the fitted object to `models/scaler.pkl`.
- **Artifacts:** Exports train/test feature and target matrices to `data/splits/`.

### Phase 3: Multi-Output ML Benchmark & Optuna Tuning (`src/models/train.py`)
- **Model Benchmarking:** Trains and compares MultiOutput XGBoost, Random Forest, and Gradient Boosting Regressors.
- **Optuna Tuning:** Executes hyperparameter search over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
- **Artifact Generation:** Saves best model to `models/best_model.pkl`, evaluation metrics to `reports/model_benchmark.json` and `reports/model_comparison.csv`, and per-target predicted vs. actual scatter plots to `reports/scatter_plots/`.

### Phase 4: Explicit Closed-Form Equation Derivation (`src/equations/rsm_fit.py`)
- Fits 2nd-order Response Surface Methodology (RSM) polynomial models ($\hat{y}_k = \beta_0 + \sum \beta_i x_i + \sum \beta_{ii} x_i^2 + \sum \beta_{ij} x_i x_j$).
- Exports derived algebraic equations to `reports/equations/rsm_equations.json` and `reports/equations/rsm_equations.md`.

### Phase 5: AASHTO Physics & Code Constraint Enforcement (`src/postprocess/constraints.py`)
Applies post-processing constraints to raw model predictions:
- **AASHTO Code Minimum Depth:** Enforces $G_{d,min} = \frac{L_{ft}}{20} \times 12$ inches.
- **Strand Rounding:** Rounds $N_s$ to the nearest EVEN integer (`[32, 122]`).
- **Girder Count Rounding:** Rounds $N_g$ to integer (`[6, 13]`).
- **Dimension Snapping:** Snaps flange depth/width and lateral spacing to practical construction increments.
- **Unit Test Suite:** 100% test pass rate in `tests/test_pipeline.py`.

### Phase 6: FastAPI Backend (`api/main.py`)
- Provides RESTful API endpoints: `POST /predict`, `GET /health`, and `GET /equations`.
- Validates inputs using Pydantic Field schemas.

### Phase 7: Dynamic Interactive Web UI (`ui/`)
- Modern dark glassmorphism dashboard (`ui/index.html`, `ui/style.css`, `ui/app.js`).
- Parametric SVG Canvas: Real-time dynamic rendering of the I-girder cross-section, bottom flange geometry, strand grid dots, and harping position.
- Features client-side RSM solver fallback and CSV report export.

---

## 3. Step-by-Step User Usage Guide

### Prerequisites
- Python 3.10+ installed.

### Step 1: Install Dependencies
Open terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

---

### Step 2: Run Data Pipeline & Train Model
To execute data loading, feature engineering, scaler fitting, and model training:
```bash
python src/models/train.py
```
This produces:
- `models/scaler.pkl`
- `models/best_model.pkl`
- `reports/model_benchmark.json`
- `reports/scatter_plots/*.png`

To generate explicit closed-form RSM equations:
```bash
python src/equations/rsm_fit.py
```

---

### Step 3: Run Automated Test Suite
To verify data pipeline, feature builder, physical constraints, and split integrity:
```bash
python -m pytest tests/test_pipeline.py
```

---

### Step 4: Launch FastAPI Inference Backend Server
Start the backend server on port 8000:
```bash
python api/main.py
```
Or via uvicorn directly:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Access Interactive API Docs (Swagger UI) at: `http://localhost:8000/docs`

---

### Step 5: Launch & Use the Web Interface
1. Open [ui/index.html](file:///e:/civil/Unified-ML-Optimization-Framework-for-Cost-Sensitive-Prestressed-Concrete-I-Girder-Design/ui/index.html) directly in any modern web browser (Chrome, Edge, Firefox).
2. Adjust the input sliders:
   - **Concrete Unit Cost:** $405 – 600 $/yd³
   - **Strand Unit Cost:** $1.26 – 2.23 $/linear ft per strand
   - **Steel Rebar Unit Cost:** $2.18 – 3.45 $/lb
   - **Bridge Span Length:** 100 – 180 ft
3. Observe:
   - **Dynamic SVG Cross-Section:** Beam height, bottom flange depth/width ($P, Q$), tendon strand grid ($N_s$), and harping line ($H_p$) update in real time.
   - **Target Output Cards:** Instantly view predicted values with AASHTO depth bounds enforced.
4. Click **Export Report** to download a CSV design report.

---

## 4. API Reference & Request Example

### Endpoint: `POST /predict`

**Request Headers:** `Content-Type: application/json`

**Example Request Payload:**
```json
{
  "Concrete": 505.0,
  "Strand": 1.73,
  "Rebar": 2.18,
  "Span_ft": 140.0
}
```

**Example Response Payload:**
```json
{
  "Gir_Dep_in": 84.0,
  "Lat_Spac_ft": 6.5,
  "No_of_Gir": 7,
  "bot_flange_depth_in": 6.0,
  "bot_flange_width_in": 42.0,
  "Number_of_strands": 68,
  "Harp_Pos_ft": 47.0,
  "raw_predictions": {
    "Gir Dep (in)": 68.57,
    "Lat Spac (ft)": 6.48,
    "No. of Gir": 6.56,
    "bot flange bot part depth (in)": 5.87,
    "bot flange bot part width (in)": 41.75,
    "Number of strand per girder": 68.51,
    "Harp Pos (ft)": 47.03
  }
}
```

---

## 5. Complete Project Directory Layout

```
Unified-ML-Optimization-Framework/
├── data/
│   ├── raw/                        # Girder_Dataset.xlsx
│   ├── processed/                  # clean_dataset.csv, features.csv
│   └── splits/                     # train/test index arrays & CSVs
├── src/
│   ├── data/                       # load_data.py (Excel loader & cleaner)
│   ├── features/                   # build_features.py (8 physical features)
│   ├── preprocessing/              # split_and_scale.py (Stratified split & scaler)
│   ├── models/                     # train.py (Optuna tuning & benchmarking)
│   ├── postprocess/                # constraints.py (AASHTO & physical bounds)
│   └── equations/                  # rsm_fit.py (Derived RSM polynomial equations)
├── api/
│   └── main.py                     # FastAPI REST server
├── ui/
│   ├── index.html                  # Glassmorphism HTML dashboard
│   ├── style.css                   # Dark mode styling & layout
│   └── app.js                     # SVG renderer & API integration
├── models/
│   ├── scaler.pkl                  # Fitted StandardScaler artifact
│   └── best_model.pkl              # Trained MultiOutput XGBoost artifact
├── reports/
│   ├── model_benchmark.json        # Evaluation metrics per target
│   ├── model_comparison.csv       # Baseline vs Tuned summary
│   ├── scatter_plots/              # Per-target scatter plots
│   └── equations/                  # Derived RSM formulas (JSON & MD)
├── tests/
│   └── test_pipeline.py            # Pytest test suite
├── requirements.txt
├── README.md
├── CONTEXT.md
├── ROADMAP.md
├── EXPLAINER.md
└── USAGE_GUIDE.md
```
