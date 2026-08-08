# Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

## Overview

This project builds a **Surrogate Machine Learning Model** that replaces traditional, computationally expensive structural optimization algorithms for prestressed concrete bridge I-girder design. Given real-time material unit costs and a target span length, the model instantly predicts a complete set of optimal structural design parameters — eliminating the need to re-run full iterative optimization every time market prices fluctuate.

---

## The Engineering Problem

In prestressed concrete bridge design, engineers traditionally rely on classical optimization algorithms — such as **Differential Evolution (DE)**, **EVOP**, or **Box Complex** methods — to find the optimal cross-section geometry for a fixed set of material costs and span lengths.

**The bottleneck:** Every time material unit costs change (concrete, steel rebar, or prestressing strands), the entire optimization loop must be re-executed. This is:

- Computationally slow — iterative solvers can take minutes to hours per design scenario
- Operationally expensive — requires engineers to manually trigger reruns for each cost update
- Inflexible — not suitable for real-time cost-sensitivity analysis or design dashboards

---

## The ML Solution

Instead of re-optimizing from scratch each time, this framework **trains a surrogate ML model on pre-computed optimal designs** spread across multiple span lengths and material cost combinations.

The trained model learns the **non-linear mapping** from 4 primary inputs directly to 7 structural design parameters — enabling instant prediction at inference time.

```
f: (Concrete, Strand, Rebar, Span_ft)  --ML Pipeline-->  (Gd, S, Ng, P, Q, Ns, Hp)
```

---

## Mathematical System Definition

### Inputs — X_raw (4 features)

| Symbol | Excel Column Name | Description | Unit | Levels |
|--------|-------------------|-------------|------|--------|
| Concrete | `Concrete` | Concrete Unit Cost | $/yd3 | 405, 505, 600 |
| Strand | `Strand` | Prestressing Strand Unit Cost | $/linear ft per strand | 1.26, 1.73, 2.23 |
| Rebar | `Rebar` | Steel Rebar Unit Cost | $/lb | 2.18, 2.82, 3.45 |
| Span_ft | Sheet name (added programmatically) | Span Length | ft | 100, 120, 140, 160, 180 |

> Note: The dataset follows a full 3x3x3x5 factorial design (Concrete x Strand x Rebar x Span) = 27 cost combinations x 5 spans x 5 runs = 675 gross rows. The value 1.26 seen in raw Rebar data is a data contamination from the Strand column and should be excluded. Valid Rebar range is 2.18 to 3.45.

### Targets — Y_target (7 outputs)

| Symbol | Excel Column Name | Description | Unit | Type |
|--------|-------------------|-------------|------|------|
| Gd | `Gir Dep (in)` | Girder Depth | inches | Continuous |
| S | `Lat Spac (ft)` | Lateral Spacing Between Girders | feet | Continuous |
| Ng | `No. of Gir` | Number of Girders | count | Integer |
| P | `bot flange bot part depth (in)` | Bottom Flange Bottom Depth | inches | Continuous |
| Q | `bot flange bot part width (in)` | Bottom Flange Bottom Width | inches | Continuous |
| Ns | `Number of strand per girder` | Number of Prestressing Strands | count | Even integer (multiples of 2) |
| Hp | `Harp Pos (ft)` | Harping Position | feet | Continuous |

---

## Dataset

### File: `Girder_Dataset.xlsx`

The file has 5 data sheets (plus one empty sheet). Each sheet represents one span length.

| Sheet Name | Span (ft) | Header Row | Data Rows |
|------------|-----------|------------|-----------|
| `100` | 100 | Row 4 | 135 |
| `120` | 120 | Row 3 | 135 |
| `140` | 140 | Row 4 | 135 |
| `160` | 160 | Row 3 | 130 (5 outlier rows dropped) |
| `180` | 180 | Row 4 | 135 |

- **Gross rows before cleaning:** 675
- **Rows after dropping outliers:** 670
- **Structure:** 27 cost combinations (3x3x3 factorial) x 5 span lengths x 5 runs/variants = 675 gross rows
- **Source:** Pre-optimized using classical structural optimization algorithms

### Known Data Quality Issues

1. **5 outlier rows in Sheet `160`** — `No. of Gir` values of 42–68 are physically impossible (valid range: 6–13). These are failed solver runs and must be dropped before training.
2. **Trailing spaces in column names** — `Strand` and `Rebar` columns have a trailing space in the raw file. Always call `.str.strip()` on column names after loading.
3. **Extra columns in sheets 120 and 160** — `Gir + Deck Dep (in)` and `Deck thickness (in)` are derived/structural detail columns. Do not use them as targets.
4. **All units are imperial** — inches, feet, $/yd3, $/lb. Do not assume SI units.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| R2 Score | > 0.98 per target |
| RMSE | < 2% of each target's range |
| MAPE | Minimized per target |

---

## Why This Matters

This framework enables:

- **Real-time cost sensitivity analysis** — change material prices on a slider, get updated design parameters instantly
- **Design automation** — no manual re-optimization required for procurement decisions
- **Interpretable equations** — symbolic regression extracts closed-form formulas usable in spreadsheets or research papers
- **Web UI integration** — a browser-based calculator with dynamic SVG I-girder visualization

---

## Project Structure

```
Unified-ML-Optimization-Framework/
├── data/
│   ├── raw/                    # Girder_Dataset.xlsx lives here
│   └── splits/                 # train/test index arrays
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data/                   # load_data.py — Excel loader + cleaner
│   ├── features/               # build_features.py — feature engineering
│   ├── models/                 # train.py, tune.py
│   ├── postprocess/            # constraints.py — physical constraint enforcement
│   └── equations/              # pysr_fit.py, rsm_fit.py
├── api/                        # FastAPI inference backend
├── ui/                         # HTML/JS/SVG frontend
├── models/                     # Saved .pkl / .pt model artifacts
├── reports/                    # Metrics, plots, equation outputs
├── tests/                      # pytest test suite
├── requirements.txt
├── README.md
├── CONTEXT.md
└── ROADMAP.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Loading | pandas, openpyxl |
| ML Core | scikit-learn, XGBoost, PyTorch |
| Hyperparameter Tuning | Optuna |
| Symbolic Regression | PySR |
| Response Surface | statsmodels, sklearn PolynomialFeatures |
| API Backend | FastAPI + uvicorn |
| Frontend | HTML / CSS / JavaScript + SVG |
| Utilities | NumPy, joblib, matplotlib, seaborn |
