# Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

## Overview

This project builds a **Surrogate Machine Learning Model** that replaces traditional, computationally expensive structural optimization algorithms for prestressed concrete bridge I-girder design. Given real-time material unit costs and a target span length, the model instantly predicts a complete set of optimal structural design parameters — eliminating the need to re-run full iterative optimization every time market prices fluctuate.

---

## The Engineering Problem

In prestressed concrete bridge design, engineers traditionally rely on classical optimization algorithms — such as **Differential Evolution (DE)**, **EVOP**, or **Box Complex** methods — to find the optimal cross-section geometry for a fixed set of material costs and span lengths.

**The bottleneck:** Every time material unit costs change (concrete, steel rebar, or prestressing strands), the entire optimization loop must be re-executed. This is:

- ?? **Computationally slow** — iterative solvers can take minutes to hours per design scenario
- ?? **Operationally expensive** — requires engineers to manually trigger reruns for each cost update
- ?? **Inflexible** — not suitable for real-time cost-sensitivity analysis or design dashboards

---

## The ML Solution

Instead of re-optimizing from scratch each time, this framework **trains a surrogate ML model on 675 pre-computed optimal designs** (27 cost combinations × 5 span lengths × 5 runs/variants).

The trained model learns the **non-linear mapping** from 4 primary inputs directly to 7 structural design parameters — enabling instant prediction at inference time.

```
f: (Cc, Cs, Cp, L)  --ML Pipeline--?  (Gd, S, Ng, P, Q, Ns, Hp)
```

---

## Mathematical System Definition

### Inputs — X_raw

| Symbol | Description | Unit |
|--------|-------------|------|
| Cc | Concrete Unit Cost | $/m³ |
| Cs | Steel Rebar Unit Cost | $/ton |
| Cp | Prestressing Strand Unit Cost | $/ton |
| L  | Span Length | m |

### Targets — Y_target

| Symbol | Description | Excel Column |
|--------|-------------|--------------|
| Gd | Girder Depth | Column K |
| S  | Lateral Spacing Between Girders | Column N |
| Ng | Number of Girders | Column O |
| P  | Bottom Flange Bottom Depth | Column P |
| Q  | Bottom Flange Bottom Width | Column Q |
| Ns | Number of Prestressing Strands | Column R |
| Hp | Harping Position | Column S |

---

## Dataset

- **Total Samples:** 675 rows
  - 27 Cost Combinations × 5 Span Lengths × 5 Runs/Variants
- **Source:** Pre-optimized using classical structural optimization algorithms
- **Split:** 540 training / 135 test (80/20 stratified)

---

## Performance Targets

| Metric | Target |
|--------|--------|
| R² Score | > 0.98 |
| RMSE | < 2% of range |
| MAPE | Minimized per target |

---

## Why This Matters

This framework enables:

- **Real-time cost sensitivity analysis** — change Cc/Cs/Cp on a slider and get updated design parameters instantly
- **Design automation** — no manual re-optimization required for procurement decisions
- **Interpretable equations** — symbolic regression extracts closed-form formulas usable in spreadsheets or research papers
- **Web UI integration** — a browser-based calculator with dynamic SVG I-girder visualization

---

## Project Structure

```
+-- data/                   # Raw and processed dataset files
+-- notebooks/              # EDA, training, and evaluation notebooks
+-- src/
¦   +-- features/           # Feature engineering pipeline
¦   +-- models/             # Model training and tuning scripts
¦   +-- postprocess/        # Physical constraint enforcement
¦   +-- equations/          # Symbolic regression outputs
+-- api/                    # FastAPI backend for model inference
+-- ui/                     # Web frontend with SVG visualization
+-- models/                 # Saved .pkl model files
+-- reports/                # Evaluation metrics and comparison tables
+-- README.md
+-- CONTEXT.md
+-- ROADMAP.md
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML Core | Python, scikit-learn, XGBoost, PyTorch |
| Hyperparameter Tuning | Optuna / GridSearchCV |
| Symbolic Regression | PySR |
| Response Surface | statsmodels / sklearn |
| API Backend | FastAPI |
| Frontend | HTML/CSS/JavaScript + SVG |
| Data | pandas, NumPy |
| Visualization | matplotlib, seaborn |
