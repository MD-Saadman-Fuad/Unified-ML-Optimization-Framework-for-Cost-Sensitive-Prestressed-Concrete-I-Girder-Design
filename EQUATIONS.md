# Complete Mathematical Equations Reference Manual
## Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

> **Document Status:** Verified & Audited  
> This manual documents all mathematical equations, domain-engineered physical expressions, machine learning formulations, Response Surface Methodology (RSM) 2nd-order polynomial equations, AASHTO code constraints, client-side fallback heuristics, and model evaluation metrics.

---

## 1. System Notation & Definitions

### Input Variables (X_raw)
| Symbol | Variable Name | Description | Units | Observed Range |
|---|---|---|---|---|
| C_c | `Concrete` | Concrete Unit Cost | $/yd³ | 405 – 600 |
| C_p | `Strand` | Prestressing Strand Unit Cost | $/linear ft per strand | 1.26 – 2.23 |
| C_s | `Rebar` | Steel Rebar Unit Cost | $/lb | 2.18 – 3.45 |
| L | `Span_ft` | Bridge Span Length | ft | 100 – 180 |

### Target Output Variables (Y_target)
| Symbol | Column Name | Description | Units | Type / Bounds |
|---|---|---|---|---|
| G_d | `Gir Dep (in)` | Girder Total Depth | inches | Continuous, G_d ≥ 0.54 L |
| S | `Lat Spac (ft)` | Lateral Spacing Between Girders | feet | Continuous, S ≥ 2.0 |
| N_g | `No. of Gir` | Number of Girders | count | Integer in [6, 13] |
| P | `bot flange bot part depth (in)` | Bottom Flange Bottom Depth | inches | Continuous, P ≥ 0.0 |
| Q | `bot flange bot part width (in)` | Bottom Flange Bottom Width | inches | Continuous, Q ≥ 6.0 |
| N_s | `Number of strand per girder` | Total Prestressing Strands | count | Even Integer in [32, 122] |
| H_p | `Harp Pos (ft)` | Harping Position Along Span | feet | Continuous in [0, L] |

---

## 2. Feature Engineering Equations

Raw inputs are expanded into an 8-dimensional feature vector (X_feat) incorporating structural mechanics and cost interaction dynamics (`src/features/build_features.py`).

$$
X_{\text{feat}} = \begin{bmatrix} C_c, & C_p, & C_s, & L, & L^2, & \frac{C_p}{C_c}, & \frac{C_s}{C_c}, & \left(\frac{C_p}{C_c}\right) L^2 \end{bmatrix}^T
$$

### Equation 2.1: Bending Moment Demand Proxy (L²)
$$
f_{L^2}(L) = L^2
$$
* **Mechanics Rationale:** Mid-span bending moment (M_u) scales quadratically with span length under uniform load (M_u ∝ w L²).

### Equation 2.2: Strand-to-Concrete Unit Cost Ratio
$$
f_{\text{ratio,strand}}(C_p, C_c) = \frac{C_p}{C_c}
$$
* **Economic Rationale:** Captures relative unit price sensitivity between high-strength steel strands and bulk concrete.

### Equation 2.3: Rebar-to-Concrete Unit Cost Ratio
$$
f_{\text{ratio,rebar}}(C_s, C_c) = \frac{C_s}{C_c}
$$
* **Economic Rationale:** Captures relative cost trade-off between mild shear reinforcement steel and concrete volume.

### Equation 2.4: Long-Span Strand Cost Penalty Interaction
$$
f_{\text{inter,strand-L}}(C_p, C_c, L) = \left( \frac{C_p}{C_c} \right) \times L^2
$$
* **Interaction Rationale:** Models the compounding financial impact of high strand unit costs on long spans requiring high prestressing forces.

---

## 3. Feature Normalization & Standardization Equation

### Equation 3.1: Standard Score (Z-Score) Normalization
$$
z_{j} = \frac{x_{j} - \mu_{j}}{\sigma_{j}} \quad \text{for } j = 1, 2, \dots, 8
$$

Where:
* x_{j} is the raw value of feature j.
* μ_{j} is the sample mean computed **strictly on training data** (N_train = 476).
* σ_{j} is the sample standard deviation computed on training data (`models/scaler.pkl`).

---

## 4. Primary Machine Learning Surrogate Formulation

The primary surrogate model ([models/best_model.pkl](file:///e:/civil/Unified-ML-Optimization-Framework-for-Cost-Sensitive-Prestressed-Concrete-I-Girder-Design/models/best_model.pkl)) uses a `MultiOutputRegressor` wrapping gradient boosted decision tree ensembles (XGBoost).

### Equation 4.1: Gradient Boosted Tree Prediction
$$
\hat{Y}_{\text{target}} = \begin{bmatrix} \hat{y}_{1}(Z), & \hat{y}_{2}(Z), & \dots, & \hat{y}_{7}(Z) \end{bmatrix}^T
$$

Where for each target output k in {1, 2, ..., 7}:
$$
\hat{y}_{k}(Z) = \eta \sum_{m=1}^{M} f_{m,k}(Z)
$$

* M is the number of boosted trees (`n_estimators`).
* f_{m,k}(Z) is the individual decision tree mapping standardized feature vector Z ∈ ℝ⁸ to leaf values.
* η is the learning rate shrinkage factor (`learning_rate`).

---

## 5. Closed-Form Response Surface Methodology (RSM) Equations

Response Surface Methodology fits 2nd-order polynomial models across raw inputs (C_c, C_p, C_s, L), providing explicit algebraic formulas (`reports/equations/rsm_equations.json`).

### General 2nd-Order Polynomial Structure
$$
\hat{y}_{k} = \beta_{0} + \beta_{1} C_c + \beta_{2} C_p + \beta_{3} C_s + \beta_{4} L + \beta_{5} C_c^2 + \beta_{6} (C_c C_p) + \beta_{7} (C_c C_s) + \beta_{8} (C_c L) + \beta_{9} C_p^2 + \beta_{10} (C_p C_s) + \beta_{11} (C_p L) + \beta_{12} C_s^2 + \beta_{13} (C_s L) + \beta_{14} L^2
$$

---

### Equation 5.1: Girder Depth (G_d, in) — R² = 0.1941

$$
G_d = 44.5074 + 0.002748 C_c + 22.369118 C_p - 4.760544 C_s + 0.099547 L + 0.000011 C_c^2 - 0.004663 (C_c C_p) - 0.002801 (C_c C_s) + 0.000009 (C_c L) - 3.920456 C_p^2 - 0.149617 (C_p C_s) - 0.030436 (C_p L) + 0.484014 C_s^2 + 0.019657 (C_s L) - 0.000179 L^2
$$

```text
Gir Dep (in) = +44.5074 +0.002748 * Concrete +22.369118 * Strand -4.760544 * Rebar +0.099547 * Span_ft +0.000011 * Concrete^2 -0.004663 * Concrete * Strand -0.002801 * Concrete * Rebar +0.000009 * Concrete * Span_ft -3.920456 * Strand^2 -0.149617 * Strand * Rebar -0.030436 * Strand * Span_ft +0.484014 * Rebar^2 +0.019657 * Rebar * Span_ft -0.000179 * Span_ft^2
```

---

### Equation 5.2: Lateral Spacing Between Girders (S, ft) — R² = 0.5274

$$
S = -4.9523 - 0.001228 C_c + 0.824492 C_p - 0.077151 C_s + 0.187315 L - 0.000001 C_c^2 + 0.000539 (C_c C_p) + 0.000501 (C_c C_s) - 0.000002 (C_c L) - 0.217914 C_p^2 + 0.063843 (C_p C_s) - 0.003307 (C_p L) - 0.141892 C_s^2 + 0.002765 (C_s L) - 0.000750 L^2
$$

```text
Lat Spac (ft) = -4.9523 -0.001228 * Concrete +0.824492 * Strand -0.077151 * Rebar +0.187315 * Span_ft -0.000001 * Concrete^2 +0.000539 * Concrete * Strand +0.000501 * Concrete * Rebar -0.000002 * Concrete * Span_ft -0.217914 * Strand^2 +0.063843 * Strand * Rebar -0.003307 * Strand * Span_ft -0.141892 * Rebar^2 +0.002765 * Rebar * Span_ft -0.000750 * Span_ft^2
```

---

### Equation 5.3: Number of Girders (N_g, count) — R² = 0.5628

$$
N_g = 23.5064 + 0.003969 C_c - 0.764795 C_p - 0.062802 C_s - 0.285145 L + 0.000001 C_c^2 - 0.000935 (C_c C_p) - 0.001346 (C_c C_s) + 0.000003 (C_c L) + 0.229808 C_p^2 - 0.132333 (C_p C_s) + 0.005558 (C_p L) + 0.297839 C_s^2 - 0.004194 (C_s L) + 0.001140 L^2
$$

```text
No. of Gir = +23.5064 +0.003969 * Concrete -0.764795 * Strand -0.062802 * Rebar -0.285145 * Span_ft +0.000001 * Concrete^2 -0.000935 * Concrete * Strand -0.001346 * Concrete * Rebar +0.000003 * Concrete * Span_ft +0.229808 * Strand^2 -0.132333 * Strand * Rebar +0.005558 * Strand * Span_ft +0.297839 * Rebar^2 -0.004194 * Rebar * Span_ft +0.001140 * Span_ft^2
```

---

### Equation 5.4: Bottom Flange Bottom Depth (P, in) — R² = 0.0323

$$
P = 24.0323 - 0.058186 C_c - 2.921621 C_p - 2.514897 C_s + 0.051839 L + 0.000046 C_c^2 + 0.006486 (C_c C_p) + 0.001850 (C_c C_s) - 0.000024 (C_c L) - 0.644544 C_p^2 - 0.525141 (C_p C_s) + 0.020309 (C_p L) + 0.673785 C_s^2 - 0.008590 (C_s L) - 0.000161 L^2
$$

```text
bot flange bot part depth (in) = +24.0323 -0.058186 * Concrete -2.921621 * Strand -2.514897 * Rebar +0.051839 * Span_ft +0.000046 * Concrete^2 +0.006486 * Concrete * Strand +0.001850 * Concrete * Rebar -0.000024 * Concrete * Span_ft -0.644544 * Strand^2 -0.525141 * Strand * Rebar +0.020309 * Strand * Span_ft +0.673785 * Rebar^2 -0.008590 * Rebar * Span_ft -0.000161 * Span_ft^2
```

---

### Equation 5.5: Bottom Flange Bottom Width (Q, in) — R² = 0.2566

$$
Q = -138.6298 + 0.026408 C_c - 19.847148 C_p + 25.923205 C_s + 2.023460 L - 0.000098 C_c^2 + 0.013797 (C_c C_p) + 0.011088 (C_c C_s) + 0.000046 (C_c L) + 7.683385 C_p^2 - 1.547968 (C_p C_s) - 0.055343 (C_p L) - 4.435635 C_s^2 - 0.017589 (C_s L) - 0.006079 L^2
$$

```text
bot flange bot part width (in) = -138.6298 +0.026408 * Concrete -19.847148 * Strand +25.923205 * Rebar +2.023460 * Span_ft -0.000098 * Concrete^2 +0.013797 * Concrete * Strand +0.011088 * Concrete * Rebar +0.000046 * Concrete * Span_ft +7.683385 * Strand^2 -1.547968 * Strand * Rebar -0.055343 * Strand * Span_ft -4.435635 * Rebar^2 -0.017589 * Rebar * Span_ft -0.006079 * Span_ft^2
```

---

### Equation 5.6: Prestressing Strands Per Girder (N_s, count) — R² = 0.8530

$$
N_s = -38.2086 - 0.108069 C_c - 24.262612 C_p - 4.315810 C_s + 1.623666 L + 0.000061 C_c^2 + 0.007040 (C_c C_p) + 0.013147 (C_c C_s) + 0.000026 (C_c L) + 4.286573 C_p^2 + 1.927072 (C_p C_s) - 0.021130 (C_p L) - 1.201516 C_s^2 + 0.008894 (C_s L) - 0.003164 L^2
$$

```text
Number of strand per girder = -38.2086 -0.108069 * Concrete -24.262612 * Strand -4.315810 * Rebar +1.623666 * Span_ft +0.000061 * Concrete^2 +0.007040 * Concrete * Strand +0.013147 * Concrete * Rebar +0.000026 * Concrete * Span_ft +4.286573 * Strand^2 +1.927072 * Strand * Rebar -0.021130 * Strand * Span_ft -1.201516 * Rebar^2 +0.008894 * Rebar * Span_ft -0.003164 * Span_ft^2
```

---

### Equation 5.7: Harping Position (H_p, ft) — R² = 0.3481

$$
H_p = 24.3191 - 0.101973 C_c + 4.769822 C_p + 25.886897 C_s - 0.094973 L + 0.000079 C_c^2 - 0.006320 (C_c C_p) + 0.014837 (C_c C_s) - 0.000041 (C_c L) + 3.631444 C_p^2 - 4.130234 (C_p C_s) - 0.027149 (C_p L) - 3.278885 C_s^2 - 0.070170 (C_s L) + 0.002237 L^2
$$

```text
Harp Pos (ft) = +24.3191 -0.101973 * Concrete +4.769822 * Strand +25.886897 * Rebar -0.094973 * Span_ft +0.000079 * Concrete^2 -0.006320 * Concrete * Strand +0.014837 * Concrete * Rebar -0.000041 * Concrete * Span_ft +3.631444 * Strand^2 -4.130234 * Strand * Rebar -0.027149 * Strand * Span_ft -3.278885 * Rebar^2 -0.070170 * Rebar * Span_ft +0.002237 * Span_ft^2
```

---

## 6. AASHTO Physics & Structural Post-Processing Equations

Raw model predictions pass through post-processing functions (`src/postprocess/constraints.py`) to enforce physical bounds, construction tolerances, and AASHTO design standards.

### Equation 6.1: AASHTO Minimum Girder Depth (G_d,min)
$$
G_{d,\text{min}}(L) = 0.045 \times L \times 12 = 0.54 L \quad \text{(inches for L in ft)}
$$

$$
G_{d,\text{final}} = \max\left( G_{d,\text{min}}(L), \; \frac{\text{round}(2.0 \times G_{d,\text{raw}})}{2.0} \right)
$$
* **Explanation:** Enforces the AASHTO minimum height requirement (0.045 · L). The lower bound is applied **after** rounding to the nearest 0.5-inch increment to prevent rounding below the code minimum.

### Equation 6.2: Discrete Girder Count Bounding (N_g)
$$
N_{g,\text{final}} = \max\left(6, \; \min\left(13, \; \text{round}(N_{g,\text{raw}})\right)\right)
$$
* **Explanation:** Rounds raw model predictions to the nearest integer and clips within standard bridge girder counts [6, 13].

### Equation 6.3: Prestressing Strand Count Even-Integer Bounding (N_s)
$$
N_{s,\text{final}} = \max\left(32, \; \min\left(122, \; 2 \times \text{round}\left( \frac{N_{s,\text{raw}}}{2.0} \right)\right)\right)
$$
* **Explanation:** Rounds raw strand counts to the nearest even integer (required for cross-sectional symmetry) and clips to [32, 122].

### Equation 6.4: Bottom Flange Depth Snapping (P)
$$
P_{\text{final}} = \max\left(0.0, \; \frac{\text{round}(2.0 \times P_{\text{raw}})}{2.0}\right)
$$
* **Explanation:** Snaps flange depth to 0.5-inch construction increments.

### Equation 6.5: Bottom Flange Width Snapping (Q)
$$
Q_{\text{final}} = \max\left(6.0, \; \frac{\text{round}(2.0 \times Q_{\text{raw}})}{2.0}\right)
$$
* **Explanation:** Snaps flange width to 0.5-inch construction increments with a 6.0-inch physical lower bound.

### Equation 6.6: Lateral Girder Spacing Snapping (S)
$$
S_{\text{final}} = \max\left(2.0, \; \frac{\text{round}(4.0 \times S_{\text{raw}})}{4.0}\right)
$$
* **Explanation:** Snaps lateral girder spacing to 0.25-foot (3-inch) grid increments with a 2.0-foot lower bound.

### Equation 6.7: Harping Position Boundary (H_p)
$$
H_{p,\text{final}} = \max\left(0.0, \; \min\left(L, \; \frac{\text{round}(2.0 \times H_{p,\text{raw}})}{2.0}\right)\right)
$$
* **Explanation:** Restricts harping point location to the physical bridge span range [0, L], snapped to 0.5-foot increments.

---

## 7. Client-Side Analytical Fallback Equations

When the backend API server is unreachable, the Web UI ([ui/app.js](file:///e:/civil/Unified-ML-Optimization-Framework-for-Cost-Sensitive-Prestressed-Concrete-I-Girder-Design/ui/app.js#L110-L146)) uses client-side heuristic approximations:

### Girder Depth (G_d)
$$
G_{d,\text{client}} = \max\left(0.54 L, \; \frac{\text{round}(2.0 \times (45.0 + 0.18 L - 0.005 C_c + 2.5 C_p))}{2.0}\right)
$$

### Lateral Spacing (S)
$$
S_{\text{client}} = \max\left(2.0, \; \frac{\text{round}(100.0 \times (5.0 + 0.015 L - 0.001 C_c + 0.4 C_s))}{100.0}\right)
$$

### Number of Girders (N_g)
$$
N_{g,\text{client}} = \max\left(6, \; \min\left(13, \; \text{round}(6.0 + 0.02 L - 0.002 C_c)\right)\right)
$$

### Bottom Flange Depth (P)
$$
P_{\text{client}} = \max\left(4.0, \; \frac{\text{round}(2.0 \times (6.2 + 0.025 L + 0.2 C_p))}{2.0}\right)
$$

### Bottom Flange Width (Q)
$$
Q_{\text{client}} = \max\left(16.0, \; \frac{\text{round}(2.0 \times (22.0 + 0.14 L + 0.8 C_p))}{2.0}\right)
$$

### Prestressing Strands (N_s)
$$
N_{s,\text{client}} = \max\left(32, \; \min\left(122, \; 2 \times \text{round}\left(\frac{20.0 + 0.38 L + 0.002 L^2 - 0.01 C_c + 4.5 C_p}{2.0}\right)\right)\right)
$$

### Harping Position (H_p)
$$
H_{p,\text{client}} = \frac{\text{round}\left(10.0 \times \left(0.35 L + 0.05 \cdot \frac{L \cdot C_p}{C_c}\right)\right)}{10.0}
$$

---

## 8. Model Evaluation Metric Formulations

### Equation 8.1: Coefficient of Determination (R²)
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

### Equation 8.2: Root Mean Squared Error (RMSE)
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### Equation 8.3: RMSE Percentage of Range (RMSE % of Range)
$$
\text{RMSE}_{\text{pct-range}} = \left( \frac{\text{RMSE}}{y_{\max} - y_{\min}} \right) \times 100
$$

### Equation 8.4: Mean Absolute Percentage Error (MAPE)
$$
\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$
