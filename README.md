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
f: (Concrete, Strand, Rebar, Span_ft)  --18 Engineered Features + ML Pipeline-->  (Gd, S, Ng, P, Q, Ns, Hp)
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

### Fixed Design Assumptions & Material Limitations

The framework operates under the following fixed material properties and design constants:

| Parameter | Symbol | Fixed Value | Units | Description |
|---|---|---|---|---|
| Concrete 28-day Strength | $f'_c$ | **8,000** | psi | Compressive design strength |
| Concrete Initial Strength | $f'_{ci}$ | **7,200** | psi | Strength at initial prestress transfer |
| Concrete Unit Density | $\gamma_c$ | **160** | pcf | Unit weight of prestressed concrete |
| Strand Ultimate Strength | $f_{pu}$ | **270** | ksi | Grade 270 low-relaxation 7-wire strand |
| Strand Nominal Diameter | $d_s$ | **0.6** | in | Standard 0.6-inch diameter strand |
| Mild Steel Yield Strength | $f_y$ | **60** | ksi | Grade 60 mild steel reinforcing bar |

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

## Achieved Performance (Latest Training Run)

| Target | R2 Score | RMSE | MAPE | Status |
|--------|----------|------|------|--------|
| `Gir Dep (in)` | **0.524** | 1.88 in | 1.84% | Good — low MAPE despite moderate R2 |
| `Lat Spac (ft)` | **0.838** | 0.34 ft | 4.68% | Strong |
| `No. of Gir` | **0.833** | 0.50 | 5.31% | Strong |
| `bot flange bot part depth (in)` | ~0.003 | 1.77 in | 20.80% | Physics noise ceiling — optimizer treats as free variable |
| `bot flange bot part width (in)` | **0.480** | 6.34 in | 14.48% | Moderate |
| `Number of strand per girder` | **0.963** | 4.42 | 4.23% | Excellent — near-production grade |
| `Harp Pos (ft)` | **0.763** | 4.60 ft | 6.38% | Strong |
| **Mean R2** | **0.630** | — | — | **+77% improvement after data averaging fix** |

> Training data: 119 averaged rows (mean of 5 optimizer runs per cost-span family).
> Model: Optuna-tuned RandomForest (MultiOutputRegressor), 18 engineered features.
> Stable across 3 independent runs: Mean R2 always 0.627-0.631.

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
│   └── equations/              # train_rsm_benchmark.py — sparse LassoCV RSM
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
