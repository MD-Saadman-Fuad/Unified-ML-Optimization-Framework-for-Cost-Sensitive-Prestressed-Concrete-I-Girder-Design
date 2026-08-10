# Mathematical Equations Reference Manual
## Unified ML Optimization Framework for Cost-Sensitive Prestressed Concrete I-Girder Design

> This document contains all mathematical formulations, domain-engineered physical expressions, Response Surface Methodology (RSM) polynomial equations, AASHTO structural code constraints, and model evaluation metrics used in this project.

---

## 1. Variable Notation & Definitions

### Input Variables ($X_{raw}$)
| Symbol | Variable Name | Description | Units | Range |
|---|---|---|---|---|
| $C_c$ | `Concrete` | Concrete Unit Cost | $\$ /\text{yd}^3$ | 405 – 600 |
| $C_p$ | `Strand` | Prestressing Strand Unit Cost | $\$ /\text{linear ft per strand}$ | 1.26 – 2.23 |
| $C_s$ | `Rebar` | Steel Rebar Unit Cost | $\$ /\text{lb}$ | 2.18 – 3.45 |
| $L$ | `Span_ft` | Bridge Span Length | $\text{ft}$ | 100 – 180 |

### Target Output Variables ($Y_{target}$)
| Symbol | Column Name | Description | Units | Type / Range |
|---|---|---|---|---|
| $G_d$ | `Gir Dep (in)` | Girder Total Depth | inches | Continuous |
| $S$ | `Lat Spac (ft)` | Lateral Spacing Between Girders | feet | Continuous |
| $N_g$ | `No. of Gir` | Number of Girders | count | Integer $\in [6, 13]$ |
| $P$ | `bot flange bot part depth (in)` | Bottom Flange Bottom Depth | inches | Continuous |
| $Q$ | `bot flange bot part width (in)` | Bottom Flange Bottom Width | inches | Continuous |
| $N_s$ | `Number of strand per girder` | Total Prestressing Strands | count | Even Integer $\in [32, 122]$ |
| $H_p$ | `Harp Pos (ft)` | Harping Position Along Span | feet | Continuous $\in [0, L]$ |

---

## 2. Domain-Driven Feature Engineering Equations

These equations transform the raw 4-input vector into an 8-feature domain vector ($X_{feat}$) based on structural mechanics and material cost dynamics (`src/features/build_features.py`).

### Equation 2.1: Bending Moment Demand Proxy ($L^2$)
$$f_{L^2}(L) = L^2$$
* **Physical Mechanics Rationale:** Maximum mid-span bending moment ($M_u$) under uniform loading scales quadratically with span length ($M_u \propto w L^2$).

### Equation 2.2: Strand-to-Concrete Cost Ratio
$$f_{ratio,strand}(C_p, C_c) = \frac{C_p}{C_c}$$
* **Economic Rationale:** Measures the relative financial penalty of adding high-strength prestressing strands versus increasing concrete section bulk.

### Equation 2.3: Rebar-to-Concrete Cost Ratio
$$f_{ratio,rebar}(C_s, C_c) = \frac{C_s}{C_c}$$
* **Economic Rationale:** Captures the cost trade-off between mild steel reinforcement and concrete volume.

### Equation 2.4: Long-Span Strand Cost Penalty Interaction
$$f_{inter,strand-L}(C_p, C_c, L) = \left( \frac{C_p}{C_c} \right) \times L^2$$
* **Cross-Domain Rationale:** Captures the compound cost sensitivity of high strand unit costs on long-span bridge designs requiring high prestressing forces.

---

## 3. Feature Normalization & Scaling Equation

### Equation 3.1: Standard Score ($Z$-Score) Normalization
$$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}$$
Where:
* $x_{i,j}$ is the raw value of feature $j$ for sample $i$.
* $\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}$ is the sample mean of feature $j$ computed on training data (`X_train`).
* $\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2}$ is the sample standard deviation computed on training data.

---

## 4. Closed-Form Response Surface Methodology (RSM) Polynomial Equations

The 2nd-order RSM models provide explicit, closed-form algebraic polynomial formulas mapping raw inputs directly to each structural design target (`src/equations/rsm_fit.py`).

### General 2nd-Order RSM Polynomial Expression
$$\hat{y}_k = \beta_0 + \sum_{i=1}^{4} \beta_i X_i + \sum_{i=1}^{4} \beta_{ii} X_i^2 + \sum_{i<j} \beta_{ij} X_i X_j$$
Where $X_1 = C_c$, $X_2 = C_p$, $X_3 = C_s$, and $X_4 = L$.

---

### Equation 4.1: Girder Depth ($G_d$, in)
$$\begin{aligned}
G_d = & + 44.5074 + 0.002748 C_c + 22.369118 C_p - 4.760544 C_s + 0.099547 L \\
& + 0.000011 C_c^2 - 0.004663 (C_c \cdot C_p) - 0.002801 (C_c \cdot C_s) + 0.000009 (C_c \cdot L) \\
& - 3.920456 C_p^2 - 0.149617 (C_p \cdot C_s) - 0.030436 (C_p \cdot L) \\
& + 0.484014 C_s^2 + 0.019657 (C_s \cdot L) - 0.000179 L^2
\end{aligned}$$
* **Use:** Direct continuous prediction of total I-girder height.

---

### Equation 4.2: Lateral Spacing Between Girders ($S$, ft)
$$\begin{aligned}
S = & - 4.9523 - 0.001228 C_c + 0.824492 C_p - 0.077151 C_s + 0.187315 L \\
& + 0.000539 (C_c \cdot C_p) + 0.000501 (C_c \cdot C_s) - 0.000002 (C_c \cdot L) \\
& - 0.217914 C_p^2 + 0.063843 (C_p \cdot C_s) - 0.003307 (C_p \cdot L) \\
& - 0.141892 C_s^2 + 0.002765 (C_s \cdot L) - 0.000750 L^2
\end{aligned}$$
* **Use:** Direct continuous prediction of lateral center-to-center girder spacing.

---

### Equation 4.3: Number of Girders ($N_g$, count)
$$\begin{aligned}
N_g = & + 23.5064 + 0.003969 C_c - 0.764795 C_p - 0.062802 C_s - 0.285145 L \\
& + 0.000001 C_c^2 - 0.000935 (C_c \cdot C_p) - 0.001346 (C_c \cdot C_s) + 0.000003 (C_c \cdot L) \\
& + 0.229808 C_p^2 - 0.132333 (C_p \cdot C_s) + 0.005558 (C_p \cdot L) \\
& + 0.297839 C_s^2 - 0.004194 (C_s \cdot L) + 0.001140 L^2
\end{aligned}$$
* **Use:** Raw continuous estimate for total girder count across bridge deck width.

---

### Equation 4.4: Bottom Flange Bottom Part Depth ($P$, in)
$$\begin{aligned}
P = & + 24.0323 - 0.058186 C_c - 2.921621 C_p - 2.514897 C_s + 0.051839 L \\
& + 0.000046 C_c^2 + 0.006486 (C_c \cdot C_p) + 0.001850 (C_c \cdot C_s) - 0.000024 (C_c \cdot L) \\
& - 0.644544 C_p^2 - 0.525141 (C_p \cdot C_s) + 0.020309 (C_p \cdot L) \\
& + 0.673785 C_s^2 - 0.008590 (C_s \cdot L) - 0.000161 L^2
\end{aligned}$$
* **Use:** Direct continuous prediction of bottom flange bottom thickness.

---

### Equation 4.5: Bottom Flange Bottom Part Width ($Q$, in)
$$\begin{aligned}
Q = & - 138.6298 + 0.026408 C_c - 19.847148 C_p + 25.923205 C_s + 2.023460 L \\
& - 0.000098 C_c^2 + 0.013797 (C_c \cdot C_p) + 0.011088 (C_c \cdot C_s) + 0.000046 (C_c \cdot L) \\
& + 7.683385 C_p^2 - 1.547968 (C_p \cdot C_s) - 0.055343 (C_p \cdot L) \\
& - 4.435635 C_s^2 - 0.017589 (C_s \cdot L) - 0.006079 L^2
\end{aligned}$$
* **Use:** Direct continuous prediction of bottom flange width.

---

### Equation 4.6: Prestressing Strands Per Girder ($N_s$, count)
$$\begin{aligned}
N_s = & - 38.2086 - 0.108069 C_c - 24.262612 C_p - 4.315810 C_s + 1.623666 L \\
& + 0.000061 C_c^2 + 0.007040 (C_c \cdot C_p) + 0.013147 (C_c \cdot C_s) + 0.000026 (C_c \cdot L) \\
& + 4.286573 C_p^2 + 1.927072 (C_p \cdot C_s) - 0.021130 (C_p \cdot L) \\
& - 1.201516 C_s^2 + 0.008894 (C_s \cdot L) - 0.003164 L^2
\end{aligned}$$
* **Use:** Raw continuous prediction of the total number of prestressing strands required per beam ($R^2 = 0.853$).

---

### Equation 4.7: Harping Position ($H_p$, ft)
$$\begin{aligned}
H_p = & + 24.3191 - 0.101973 C_c + 4.769822 C_p + 25.886897 C_s - 0.094973 L \\
& + 0.000079 C_c^2 - 0.006320 (C_c \cdot C_p) + 0.014837 (C_c \cdot C_s) - 0.000041 (C_c \cdot L) \\
& + 3.631444 C_p^2 - 4.130234 (C_p \cdot C_s) - 0.027149 (C_p \cdot L) \\
& - 3.278885 C_s^2 - 0.070170 (C_s \cdot L) + 0.002237 L^2
\end{aligned}$$
* **Use:** Direct continuous prediction of harping point location along the span.

---

## 5. AASHTO & Physical Code Post-Processing Constraint Equations

These equations enforce structural code compliance, constructibility snapping, and discrete integer rounding on raw model outputs (`src/postprocess/constraints.py`).

### Equation 5.1: AASHTO Minimum Girder Depth Lower Bound
$$G_{d,min}(L) = 0.045 \times L \times 12 \quad \text{(inches)}$$

$$G_{d,enforced} = \max\left( G_{d,min}(L), \; \frac{\text{round}(2.0 \times G_{d,raw})}{2.0} \right)$$
* **Use:** Guarantees that the girder height never drops below AASHTO depth-to-span requirements ($0.045 \times L$), snapped to 0.5-inch increments.

### Equation 5.2: Discrete Girder Count Rounding & Bounding ($N_g$)
$$N_{g,enforced} = \max\left(6, \; \min\left(13, \; \text{round}(N_{g,raw})\right)\right)$$
* **Use:** Converts continuous ML predictions into a practical integer girder count bounded between 6 and 13.

### Equation 5.3: Prestressing Strand Count Even-Integer Rounding & Bounding ($N_s$)
$$N_{s,enforced} = \max\left(32, \; \min\left(122, \; 2 \times \text{round}\left( \frac{N_{s,raw}}{2.0} \right)\right)\right)$$
* **Use:** Forces strand count to be an even integer (required for symmetric tendon placement) bounded between 32 and 122 strands.

### Equation 5.4: Bottom Flange Depth Constructibility Snapping ($P$)
$$P_{enforced} = \max\left(0.0, \; \frac{\text{round}(2.0 \times P_{raw})}{2.0}\right)$$
* **Use:** Snaps flange depth to the nearest 0.5-inch construction tolerance.

### Equation 5.5: Bottom Flange Width Constructibility Snapping ($Q$)
$$Q_{enforced} = \max\left(6.0, \; \frac{\text{round}(2.0 \times Q_{raw})}{2.0}\right)$$
* **Use:** Snaps flange width to the nearest 0.5-inch construction tolerance.

### Equation 5.6: Lateral Girder Spacing Constructibility Snapping ($S$)
$$S_{enforced} = \max\left(2.0, \; \frac{\text{round}(4.0 \times S_{raw})}{4.0}\right)$$
* **Use:** Snaps lateral spacing to the nearest 0.25-foot (3-inch) placement grid.

### Equation 5.7: Harping Position Boundary & Snapping ($H_p$)
$$H_{p,enforced} = \max\left(0.0, \; \min\left(L, \; \frac{\text{round}(2.0 \times H_{p,raw})}{2.0}\right)\right)$$
* **Use:** Restricts the harping deviation point to lie strictly within the span range $[0, L]$, snapped to 0.5-foot steps.

---

## 6. Performance Evaluation Metric Equations

These standard statistical equations measure model accuracy across training and test sets (`src/models/train.py`).

### Equation 6.1: Coefficient of Determination ($R^2$)
$$R^2 = 1 - \frac{\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2}{\sum_{i=1}^{n} \left(y_i - \bar{y}\right)^2}$$
Where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ is the empirical mean of actual values.

### Equation 6.2: Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2 }$$

### Equation 6.3: RMSE Percentage of Range ($\text{RMSE}_{\% \text{ range}}$)
$$\text{RMSE}_{\% \text{ range}} = \left( \frac{\text{RMSE}}{y_{\max} - y_{\min}} \right) \times 100\%$$

### Equation 6.4: Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
