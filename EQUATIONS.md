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
| G_d | `Gir Dep (in)` | Girder Total Depth | inches | Continuous, 45.0 ≤ G_d ≤ 72.0 |
| S | `Lat Spac (ft)` | Lateral Spacing Between Girders | feet | Continuous, S ≥ 2.0 |
| N_g | `No. of Gir` | Number of Girders | count | Integer in [6, 13] |
| P | `bot flange bot part depth (in)` | Bottom Flange Bottom Depth | inches | Continuous, P ≥ 0.0 |
| Q | `bot flange bot part width (in)` | Bottom Flange Bottom Width | inches | Continuous, Q ≥ 6.0 |
| N_s | `Number of strand per girder` | Total Prestressing Strands | count | Even Integer in [32, 122] |
| H_p | `Harp Pos (ft)` | Harping Position Along Span | feet | Continuous in [0, L] |

### Fixed Design Assumptions & Material Limitations

The following structural material parameters were held **constant** across all optimizer training runs and define the fixed design boundaries of this ML surrogate framework:

$$
\begin{aligned}
f'_c &= 8000\text{ psi} && \text{(28-day Concrete Compressive Strength)} \\
f'_{ci} &= 7200\text{ psi} && \text{(Initial Concrete Strength at Prestress Transfer)} \\
\gamma_c &= 160\text{ pcf} && \text{(Concrete Unit Weight / Density)} \\
f_{pu} &= 270\text{ ksi} && \text{(Ultimate Tensile Strength of Prestressing Strands)} \\
d_s &= 0.6\text{ in} && \text{(Prestressing Strand Nominal Diameter)} \\
f_y &= 60\text{ ksi} && \text{(Yield Strength of Mild Reinforcing Steel)}
\end{aligned}
$$

| Symbol | Parameter | Value | Units | Description / Standard |
|---|---|---|---|---|
| $f'_c$ | Concrete Compressive Strength | **8,000** | psi | 28-day concrete compressive design strength |
| $f'_{ci}$ | Initial Concrete Strength | **7,200** | psi | Concrete strength at initial prestress transfer |
| $\gamma_c$ | Concrete Unit Density | **160** | pcf | Unit weight of reinforced/prestressed concrete |
| $f_{pu}$ | Strand Tensile Strength | **270** | ksi | Grade 270 low-relaxation 7-wire prestressing strand |
| $d_s$ | Strand Nominal Diameter | **0.6** | in | Standard 0.6-inch diameter prestressing strand |
| $f_y$ | Mild Steel Yield Strength | **60** | ksi | Grade 60 mild steel reinforcing bar yield stress |

---

## 2. Feature Engineering Equations

Raw inputs are expanded into an **11-dimensional validated feature vector** ($X_{\text{feat}}$) incorporating span demand proxies and unit cost interaction terms (`src/features/build_features.py`).

$$
X_{\text{feat}} \in \mathbb{R}^{11} = \begin{bmatrix} 
C_c, & C_p, & C_s, & L, & L^2, & L^3, \\
\frac{C_p}{C_c}, & \frac{C_s}{C_c}, & \frac{C_s}{C_p}, & \left(\frac{C_p}{C_c}\right) L^2, & \frac{C_p C_s}{C_c^2}
\end{bmatrix}^T
$$

> **Engineering Review Note:** Features were audited and reduced from 18 to 11. Removed features lacked either engineering justification or grounding in actual dataset quantities. See inline verdicts below.

---

### Equation 2.1: Bending Moment Demand Proxy ($L^2$)

$$
f_{L^2}(L) = L^2
$$

* **Rationale:** Mid-span bending moment scales quadratically with span under uniform load ($M_u \propto w L^2$). Well-established in structural mechanics and supported by dataset correlation. ✅ **KEEP**

---

### Equation 2.2: Span-Cubed Demand Proxy ($L^3$)

$$
f_{L^3}(L) = L^3
$$

* **Rationale:** Represents the nonlinear increase of beam deflection and demand with span length. Included as an optional nonlinear span feature. ⚠️ **KEEP** — note: does not claim harping position itself follows $L^3$; that claim is not established by this dataset.

---

### Equation 2.3: Strand-to-Concrete Unit Cost Ratio

$$
f_{\text{ratio,strand}}(C_p, C_c) = \frac{C_p}{C_c}
$$

* **Rationale:** Economic sensitivity feature — captures how expensive prestressing strand is relative to concrete. Dataset directly varies both $C_p$ and $C_c$, making this ratio meaningful. ✅ **KEEP**

---

### Equation 2.4: Rebar-to-Concrete Unit Cost Ratio

$$
f_{\text{ratio,rebar}}(C_s, C_c) = \frac{C_s}{C_c}
$$

* **Rationale:** Economic sensitivity feature — captures relative rebar vs concrete unit price. Relevant for cost-sensitive optimization. ✅ **KEEP**

---

### Equation 2.5: Rebar-to-Strand Cross-Steel Cost Ratio

$$
f_{\text{ratio,rebar-strand}}(C_s, C_p) = \frac{C_s}{C_p}
$$

* **Rationale:** Economic feature — represents the relative unit cost trade-off between the two steel material types. Especially useful for cost sensitivity analysis. ✅ **KEEP**

---

### Equation 2.6: Strand Cost × Span Demand Interaction

$$
f_{\text{inter,strand-L2}}(C_p, C_c, L) = \left( \frac{C_p}{C_c} \right) \times L^2
$$

* **Rationale:** Captures compounding effect of high strand unit cost on long spans. The $L^2$ variant is retained as it aligns with moment demand scaling. ⚠️ **KEEP ($L^2$ only)** — the $L^3$ interaction variant is not independently validated relative to the $L^2$ form and has been removed.

---

### Equation 2.7: Composite Cost Interaction

$$
f_{\text{cost-int}}(C_p, C_s, C_c) = \frac{C_p \cdot C_s}{C_c^2}
$$

* **Rationale:** Captures joint steel cost pressure relative to concrete cost squared. Retained as an experimental interaction term. ⚠️ **KEEP** — renamed from `elasticity_index` (previous label was misleading; this is not a structural elasticity parameter).

---

> **Removed Features (not in active model):**
>
> | Feature | Expression | Reason Removed |
> |:---|:---|:---|
> | `M_LL_proxy` | $0.72L + 10$ | Not a real AASHTO equation; redundant with $L$ |
> | `strand_span_cube` | $C_p \cdot L^3$ | $L^3$ interaction not validated vs $L^2$ variant |
> | `volumetric_stiffness` | $C_c/(C_p L^2)$ | No evidence of real engineering relationship |
> | `span_concrete_ratio` | $L/C_c$ | Likely redundant; no established basis |
> | `span_strand_stiffness` | $L^3/C_p$ | No established basis; redundant |
> | `total_cost_index` | $(C_c/100 + C_p + C_s)/L$ | Artificial scaling coefficients (100×, 1000×) — not quantity-based |
> | `concrete_cost_share` | $C_c / C_{\text{tot}}$ | Depends on artificial total cost; invalid without real quantities |


---

## 3. Feature Normalization & Standardization Equation

### Equation 3.1: Standard Score (Z-Score) Normalization
$$
z_{j} = \frac{x_{j} - \mu_{j}}{\sigma_{j}} \quad \text{for } j = 1, 2, \dots, 11
$$

Where:
* $x_{j}$ is the raw value of feature $j$.
* $\mu_{j}$ is the sample mean computed **strictly on training data** ($N_{\text{train}}$ averaged unique cost-span families).
* $\sigma_{j}$ is the sample standard deviation computed on training data (`models/scaler.pkl`).

---

## 4. Primary Machine Learning Surrogate Formulation

The primary surrogate model (`models/best_model.pkl`) is a **Tuned RandomForest Regressor** ensemble, selected through Optuna hyperparameter optimization because it outperforms gradient boosting on the 95-sample averaged training dataset (Test Mean $R^2 = 0.630$ for RF vs. $0.539$ for GradientBoosting and $0.519$ for XGBoost).

### Equation 4.1: Random Forest Ensemble Prediction
$$
\hat{Y}_{\text{target}} = \begin{bmatrix} \hat{y}_{1}(Z), & \hat{y}_{2}(Z), & \dots, & \hat{y}_{7}(Z) \end{bmatrix}^T
$$

Where for each target output $k \in \{1, 2, \dots, 7\}$:
$$
\hat{y}_{k}(Z) = \frac{1}{B} \sum_{b=1}^{B} T_{b,k}(Z)
$$

* $B$ is the number of ensemble trees (`n_estimators = 1000`).
* $T_{b,k}(Z)$ is the individual decision tree trained on a bootstrap sample of the training data, mapping standardized feature vector $Z \in \mathbb{R}^{11}$ to leaf values.
* Split decisions evaluate a random subset of features ($\sqrt{11} \approx 3$ features via `max_features = 'sqrt'`).

*(For benchmarking, MultiOutput XGBoost and GradientBoosting models are also evaluated; see `reports/model_benchmark.json`).*

---

## 5. Closed-Form Response Surface Methodology (RSM) Equations

Response Surface Methodology fits **Sparse LassoCV 3rd-order polynomial models** on **normalized inputs** (all scaled to [0, 1] via MinMaxScaler before expansion). This approach auto-prunes near-zero cross-terms via L1 regularization, producing compact, interpretable equations. All 7 target variables ($G_d, S, N_g, P, Q, N_s, H_p$) are regressed via LassoCV, with physical bounds enforced during post-processing.

**Script:** `src/equations/train_rsm_benchmark.py`  
**Output:** `reports/equations/rsm_equations.json`

---

### Variable Legend

Before reading any equation below, refer to this master legend for all symbols used:

| Symbol | Full Name | Units | Raw Range | Meaning |
|:---|:---|:---|:---|:---|
| $C_c$ | Concrete Unit Cost | \$/yd³ | 405 – 600 | Cost per cubic yard of concrete placed |
| $C_p$ | Prestressing Strand Cost | \$/linear ft per strand | 1.26 – 2.23 | Unit price of high-strength steel prestressing strand |
| $C_s$ | Steel Rebar Cost | \$/lb | 2.18 – 3.45 | Unit price of mild steel shear reinforcement bars |
| $L$ | Span Length | ft | 100 – 180 | Bridge span between supports |
| $G_d$ | Girder Depth | in | 45 – 72 | Total depth of I-girder from top flange to bottom flange |
| $S$ | Lateral Spacing | ft | 2.0 – 8.2 | Center-to-center spacing between adjacent girders |
| $N_g$ | Number of Girders | count | 6 – 13 | Total number of prestressed I-girders in cross-section |
| $P$ | Bottom Flange Depth | in | 0 – 11 | Thickness of the bottom (tension) flange bottom part |
| $Q$ | Bottom Flange Width | in | 6 – 46 | Width of the bottom (tension) flange bottom part |
| $N_s$ | Strands per Girder | count | 32 – 108 | Total number of 0.6-in diameter prestressing strands per girder (always even) |
| $H_p$ | Harping Position | ft | 30 – 90 | Distance from end of beam where straight strands are deflected (harped) upward |

**Normalized inputs** (used inside RSM equations, all dimensionless, range [0, 1]):

| Symbol | Formula | Meaning |
|:---|:---|:---|
| $C_c^*$ | $(C_c - 405) / 195$ | 0 = cheapest concrete (\$405/yd³), 1 = most expensive (\$600/yd³) |
| $C_p^*$ | $(C_p - 1.26) / 0.97$ | 0 = cheapest strand (\$1.26/ft), 1 = most expensive (\$2.23/ft) |
| $C_s^*$ | $(C_s - 2.18) / 1.27$ | 0 = cheapest rebar (\$2.18/lb), 1 = most expensive (\$3.45/lb) |
| $L^*$ | $(L - 100) / 80$ | 0 = 100-ft span, 1 = 180-ft span |

> **How to use normalized inputs:** To evaluate an equation for e.g. $C_c = 505$, $C_p = 1.73$, $C_s = 2.18$, $L = 140$:  
> $C_c^* = (505-405)/195 = 0.513$, $C_p^* = (1.73-1.26)/0.97 = 0.485$, $C_s^* = 0.0$, $L^* = (140-100)/80 = 0.5$

---

### Input Normalization Equations

$$
C_c^* = \frac{C_c - 405}{195}, \quad C_p^* = \frac{C_p - 1.26}{0.97}, \quad C_s^* = \frac{C_s - 2.18}{1.27}, \quad L^* = \frac{L - 100}{80}
$$

> All RSM equation coefficients below operate on these normalized inputs $\in [0, 1]$.

---

### General Sparse LassoCV Polynomial Structure (3rd-Order)

$$
\hat{y}_k = \sum_{j} \beta_j \cdot \phi_j(C_c^*, C_p^*, C_s^*, L^*), \quad |\beta_j| > 0 \text{ (LassoCV-selected terms only)}
$$

Where $\phi_j$ are polynomial monomials up to degree 3. A cross-validated Lasso penalty ($\alpha$) per target automatically zeros out irrelevant cross-products, keeping only terms that genuinely predict the output.

---

### Equation 5.1: Girder Depth ($G_d$, inches) — R² = 0.4878

> 3rd-order LassoCV, $\alpha = 0.001$, **29 active terms** | Moderate fit — step-like data structure limits polynomial accuracy

$$
G_d = f(C_c^*, C_p^*, C_s^*, L^*)
$$

```text
Gd [inches] =
  +64.397                          ← baseline depth when all costs/span are at minimum
  -2.612 × Cc*                     ← higher concrete cost → slightly shallower beam (less concrete used)
  +8.082 × Cp*                     ← higher strand cost → deeper beam (fewer strands, larger depth)
  -1.830 × Cs*                     ← higher rebar cost → marginally shallower
  +13.119 × L*                     ← longer span → deeper beam (dominant linear effect)
  -1.796 × (Cc* · Cp*)            ← concrete-strand cost interaction: mitigates combined cost sensitivity
  -0.092 × (Cc* · Cs*)            ← small concrete-rebar interaction
  +5.226 × (Cc* · L*)             ← high concrete cost on long spans → forces depth increase
  -0.718 × (Cp* · Cs*)            ← strand-rebar interaction
  -8.805 × (Cp* · L*)             ← high strand cost on long spans → reduces depth (substitute capacity)
  +4.948 × (Cs* · L*)             ← rebar-span interaction
  -21.510 × L*²                   ← quadratic span correction (nonlinear depth-span relationship)
  +4.705 × Cc*³                   ← cubic concrete cost effect
  ... (additional 3rd-order cross-terms follow same pattern)
  +14.605 × L*³                   ← dominant cubic span term (beam depth is ~L^1.5 in mechanics)
```

**Engineering note:** Girder depth ($G_d$) in the AASHTO dataset snaps to discrete beam depths (45, 54, 63, 72 in). A polynomial cannot model these step changes perfectly, explaining the moderate R² = 0.49. Nevertheless, this represents a large improvement over the old linear RSM (R² = −0.39).

---

### Equation 5.2: Lateral Spacing Between Girders ($S$, feet) — R² = 0.7373

> 3rd-order LassoCV, $\alpha = 0.005$, **6 active terms** — extremely sparse, very interpretable

$$
S = 6.593 + 0.018\,C_p^* + 1.012\,L^* + 0.005\,(C_c^* \cdot C_p^*) + 0.010\,(C_c^* \cdot C_s^* \cdot L^*) - 0.183\,C_s^{*3} - 2.963\,L^{*3}
$$

```text
S [feet] =
  +6.593                           ← baseline spacing at minimum span (100 ft) and minimum costs
  +0.018 × Cp*                     ← higher strand cost marginally increases spacing (fewer, wider-spaced girders)
  +1.012 × L*                      ← longer span → wider spacing (fewer girders needed for long spans)
  +0.005 × (Cc* · Cp*)            ← concrete-strand cost interaction: both expensive → slightly wider spacing
  +0.010 × (Cc* · Cs* · L*)       ← 3-way interaction: high costs on long span compounds spacing increase
  -0.183 × Cs*³                   ← high rebar cost → tighter spacing (more/closer girders used)
  -2.963 × L*³                    ← cubic span correction (spacing is not purely linear in span)
```

**Engineering note:** Lateral spacing $S$ is tied to bridge deck width and girder count. The dominant predictor is span length ($L^*$ coefficient = 1.012), followed by a cubic correction. Concrete and rebar cost effects are small but meaningful for long spans.

---

### Equation 5.3: Number of Girders ($N_g$, integer count) — R² = 0.7138 ⚠️ MODIFIED

> **ML Regression Target.** $N_g$ is predicted from data via LassoCV sparse polynomial regression — it is **not** computed from a fixed 37-ft deck width.

$$
N_g = f\!\left(\,C_c^*,\; C_p^*,\; C_s^*,\; L^*\;\right) \quad \text{(3rd-order LassoCV, 8 active terms — see } rsm\_equations.json \text{)}
$$

**Why not a fixed deck-width formula?**

The formula $N_g = \operatorname{round}(37.0 / S) + 1$ assumed a constant 37 ft deck width ($B = 37\text{ ft}$). However, the optimization framework depends on defined bridge width and geometric constraints. Unless 37 ft is explicitly confirmed as the uniform bridge width across all dataset rows, hardcoding $B = 37\text{ ft}$ imposes an unverified structural constraint.

> [!IMPORTANT]
> **Previous documentation assumed $B = 37\text{ ft}$ deck width.** This assumption has been removed. $N_g$ is now regressed directly from cost and span inputs using LassoCV (R² = **0.71**), allowing the model to adapt to dataset-inherent bridge width relationships $S = f(N_g)$.

**Validity bounds:** Post-processed (Equation 6.2) to round to integer and clamp within range **[6, 13]** per AASHTO construction feasibility limits.

---

### Equation 5.4: Bottom Flange Bottom Depth ($P$, inches) — R² ≈ 0.00

> 3rd-order LassoCV, $\alpha = 0.1$, **1 active term** — Lasso correctly identified this target is nearly unpredictable from cost inputs

$$
P = 7.614 - 0.152\,C_p^*
$$

```text
P [inches] =
  +7.614                           ← mean bottom flange depth across all designs (~7.5 in)
  -0.152 × Cp*                     ← higher strand cost → marginally thinner bottom flange
                                      (only weakly significant — all other terms pruned by Lasso)
```

**Engineering note:** Bottom flange depth $P$ is a constructibility parameter controlled primarily by cross-section geometry and local buckling constraints — **not** by material unit costs. The Lasso correctly pruned all 34 possible 3rd-order terms down to 1, as no cost input has meaningful predictive power for $P$. The near-zero R² is honest and expected — this target requires finite element geometry data, not cost data, to predict accurately.

---

### Equation 5.5: Bottom Flange Bottom Width ($Q$, inches) — R² = 0.1267

> 3rd-order LassoCV, $\alpha = 0.1$, **8 active terms** (normalized inputs)

$$
Q = 26.980 - 2.670\,C_c^* + 1.168\,C_s^* + 36.142\,L^* + 3.287\,(C_c^* \cdot C_p^{*2}) + 1.685\,(C_c^* \cdot C_s^* \cdot L^*) + 1.301\,C_p^{*3} - 1.704\,C_s^{*3} - 21.830\,L^{*3}
$$

```text
Q [inches] =
  +26.980                          ← baseline bottom flange width at minimum costs and 100-ft span
  -2.670 × Cc*                     ← higher concrete cost → narrower flange (reduce concrete volume)
  +1.168 × Cs*                     ← higher rebar cost → wider flange (more steel area needed)
  +36.142 × L*                     ← longer span → much wider flange (dominant effect; moment capacity demand)
  +3.287 × (Cc* · Cp*²)           ← concrete cost × strand cost²: both expensive → wider flange
  +1.685 × (Cc* · Cs* · L*)       ← 3-way interaction: high costs on long spans → wider flange
  +1.301 × Cp*³                   ← cubic strand cost effect
  -1.704 × Cs*³                   ← cubic rebar cost: very high rebar cost reverses flange widening
  -21.830 × L*³                   ← cubic span correction (nonlinear limit state at very long spans)
```

**Engineering note:** Span length dominates $Q$ (coefficient = 36.1 for linear, −21.8 for cubic — a nonlinear arch pattern). The low R² = 0.13 reflects that flange width is highly geometry-dependent and the 4-variable cost model cannot fully capture it.

---

### Equation 5.6: Prestressing Strands Per Girder ($N_s$, even count) — R² = 0.9631

> 3rd-order LassoCV, $\alpha = 0.01$, **19 active terms** | Excellent fit — strands are the most cost-driven design parameter

```text
Ns [count] =
  +40.689                          ← baseline strands at minimum span (100 ft) and minimum costs
  +0.168 × Cc*                     ← higher concrete cost → slightly more strands (compensate with prestress)
  -3.435 × Cp*                     ← higher strand cost → fewer strands selected by optimizer
  -0.152 × Cs*                     ← higher rebar cost → marginally fewer strands
  +44.723 × L*                     ← longer span → many more strands (primary structural demand driver)
  -2.523 × (Cp* · Cs*)            ← both steel types expensive → optimizer reduces strand count
  +0.181 × (Cp* · L*)             ← strand cost × span interaction
  +73.009 × L*²                   ← quadratic span term (strand count scales ~L² in mechanics)
  -1.096 × Cc*³                   ← cubic concrete cost correction
  +8.387 × (Cc*² · L*)            ← concrete cost² × span: high concrete cost on long spans → more strands
  ... (additional 3rd-order cross-terms)
  -57.590 × L*³                   ← cubic span correction (plateau effect at very long spans)
```

**Engineering note:** $N_s$ has R² = **0.96** — by far the best-fitting target. This makes physical sense: the optimizer directly trades strand count for cost (more expensive strands → fewer selected). Span length is the dominant predictor (linear coefficient 44.7, quadratic 73.0), consistent with the AASHTO prestress demand formula where $N_s \propto M_u / (f_{pe} \cdot e)$ and $M_u \propto L^2$.

**Output rounding:** Always rounded to the nearest even integer (strand pairs) and clamped to [32, 108].

---

### Equation 5.7: Harping Position ($H_p$, feet from beam end) — R² = 0.7175 ⚠️ MODIFIED

> **ML Regression Target.** Harping position is predicted from data via LassoCV — it is **not** a fixed percentage of span length.

$$
H_p = f\!\left(\,C_c^*,\; C_p^*,\; C_s^*,\; L^*\;\right) \quad \text{(3rd-order LassoCV, 13 active terms — see } rsm\_equations.json \text{)}
$$

**Why not a fixed formula?**

The harping position depends on:
- **Tendon geometry** — the drape of the prestressing strand from the harping point to the beam end
- **Harping angle constraints** — e.g. a **1:7** slope limit (rise:run) is common in practice; this directly ties $H_p$ to the eccentricity profile, not just span length
- **Design configuration** — number of harped strands, eccentricity at midspan vs. end, cross-section depth

The approximation $H_p = 0.40 \times L$ was a dataset mean observed across 119 averaged cost-span families, but this is **not a universal equation**. The actual harping fraction varies with the above factors and is specific to each design scenario.

> [!IMPORTANT]
> **Previous documentation incorrectly labelled $H_p = 0.40L$ as "physics-derived".** This has been corrected. The value 0.40 was a statistical artifact of the dataset, not an AASHTO engineering formula.

**AASHTO-required validity range:** Regardless of what the model predicts, $H_p$ is post-processed (see Equation 6.7) to ensure it lies within $[0,\; L]$ and is snapped to 0.5 ft increments. The physical harping zone $[0.30L,\; 0.50L]$ is a common design guideline, not a hard constraint in all configurations.

---

### RSM Performance Summary (Sparse LassoCV 3rd-Order vs. Old 2nd-Order Linear)

| Target | Symbol | Old R² (15-term Linear) | New R² (LassoCV) | Change | Method |
|:---|:--:|:--:|:--:|:--:|:---|
| Girder Depth | $G_d$ | −0.39 | **0.49** | +0.88 🔼 | 3rd-order Lasso, 14 terms |
| Lateral Spacing | $S$ | 0.63 | **0.74** | +0.11 🔼 | 3rd-order Lasso, 7 terms |
| No. of Girders | $N_g$ | 0.59 | **0.71** | +0.12 🔼 | 3rd-order Lasso, 8 terms (was `round(37/S)+1`) |
| Bot Flange Depth | $P$ | 0.07 | 0.00 | — | Lasso pruned to **1 term** |
| Bot Flange Width | $Q$ | 0.44 | 0.13 | — | 3rd-order Lasso, 14 terms |
| Strands per Girder | $N_s$ | 0.87 | **0.96** | +0.09 🔼 | 3rd-order Lasso, 7 terms |
| Harp Position | $H_p$ | 0.06 | **0.72** | +0.66 🔼 | 3rd-order Lasso, 13 terms (was `0.40×L`) |
| **Mean R²** | — | **0.32** | **0.56** | **+0.24 🔼** | — |



---


## 6. AASHTO Physics & Structural Post-Processing Equations

Raw model predictions pass through post-processing functions (`src/postprocess/constraints.py`) to enforce physical bounds, construction tolerances, and AASHTO design standards.

### Equation 6.1: Girder Depth Physical Bounds (G_d)
$$
G_{d,\text{final}} = \max\left(45.0, \; \min\left(72.0, \; \frac{\text{round}(2.0 \times G_{d,\text{raw}})}{2.0}\right)\right)
$$
* **Explanation:** Snaps girder depth to the nearest 0.5-inch increment and clamps within the physically observed dataset bounds. The upper bound of **72.0 in** is the AASHTO Type VI standard beam maximum depth, which was the hard design variable constraint used in the optimization that generated this dataset. The commonly cited formula G_d ≥ 0.045 × L × 12 produces values up to 97.2 in for a 180-ft span — far exceeding the 72-in dataset maximum — and therefore does **not** apply here as a binding lower bound.

### Equation 6.2: Discrete Girder Count Bounding (N_g)
$$
N_{g,\text{final}} = \max\left(N_{g,\text{min}}(L), \; \min\left(11, \; \text{round}(N_{g,\text{raw}})\right)\right)
$$
$$
N_{g,\text{min}}(L) = \begin{cases} 8 & L \geq 175\,\text{ft} \\ 6 & L < 175\,\text{ft} \end{cases}
$$
* **Explanation:** Rounds to nearest integer. Upper bound is **11** (actual maximum in averaged optimal designs, not 13 as previously stated). Span-aware lower bound: 180 ft span requires at least 8 girders (dataset minimum is 8 at that span).

### Equation 6.3: Prestressing Strand Count Even-Integer Bounding (N_s)
$$
N_{s,\text{final}} = \max\left(32, \; \min\left(108, \; 2 \times \text{round}\left( \frac{N_{s,\text{raw}}}{2.0} \right)\right)\right)
$$
* **Explanation:** Rounds to nearest even integer. Upper bound corrected to **108** (actual maximum in the averaged optimal design dataset; the raw dataset maximum of 122 appears only in stochastic outlier runs, not in the averaged optima).

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
G_{d,\text{client}} = \max\left(45.0, \; \min\left(72.0, \; \frac{\text{round}(2.0 \times (55.0 + 0.075 L - 0.002 C_c + 0.4 C_p))}{2.0}\right)\right)
$$
* **Note:** Hard-capped at 72.0 in (AASHTO Type VI beam maximum) and floored at 45.0 in. The previous formula using 0.045 × L × 12 as a minimum was incorrect — it forces outputs up to 97.2 in for 180-ft spans, exceeding the dataset maximum of 72.0 in.

### Lateral Spacing (S)
$$
S_{\text{client}} = \max\left(2.0, \; \frac{\text{round}(100.0 \times (5.0 + 0.015 L - 0.001 C_c + 0.4 C_s))}{100.0}\right)
$$

### Number of Girders (N_g)
$$
N_{g,\text{client}} = \begin{cases} 10 & L \geq 175\,\text{ft} \\ 7 & L < 175\,\text{ft} \end{cases}, \quad \text{clamped to } [6, 11]
$$
* **Note:** Ng is nearly constant at 6–7 for spans 100–160 ft (correlation with costs ≈ 0.01–0.05; only span matters). It jumps to 8–11 at 180 ft. Previous quadratic formula gave 12 for 100 ft (actual mean: 6.8) — corrected.


### Bottom Flange Depth (P)
$$
P_{\text{client}} = \max\left(4.0, \; \frac{\text{round}(2.0 \times (6.2 + 0.025 L + 0.2 C_p))}{2.0}\right)
$$

### Bottom Flange Width (Q)
$$
Q_{\text{client}} = \max\left(16.0, \; \frac{\text{round}(2.0 \times (22.0 + 0.14 L + 0.8 C_p))}{2.0}\right)
$$

### Prestressing Strands (N_s)

Piecewise-linear interpolation anchored at per-span means from averaged optimal dataset:

| Span | Base N_s |
|---|---|
| 100 ft | 39 |
| 120 ft | 55 |
| 140 ft | 71 |
| 160 ft | 92 |
| 180 ft | 95 |

$$
N_{s,\text{client}} = \text{Interp}(L) + \Delta_{Cc} + \Delta_{Cp}, \quad \text{clamped to } [32, 108]
$$
$$
\Delta_{Cc} = 2 \times \text{round}\left(\frac{C_c - 505}{100} \cdot \frac{1}{2}\right), \quad \Delta_{Cp} = 2 \times \text{round}\left(\frac{C_p - 1.73}{0.5} \cdot \frac{-2}{2}\right)
$$
* **Note:** Strand count correlates at **0.97 with span** and barely with costs. The 140→160 ft jump (70→92) is nonlinear — a single linear L coefficient cannot capture it accurately.

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
