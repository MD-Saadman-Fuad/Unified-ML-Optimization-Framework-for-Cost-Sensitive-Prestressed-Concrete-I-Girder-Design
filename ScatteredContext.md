Project Context & System Architecture
The Engineering Problem
In prestressed concrete bridge design, traditional optimization tools (e.g., Differential Evolution, EVOP, or Box Complex algorithms) compute a single optimal cross-section for a fixed set of material costs and span lengths. However, running a full iterative structural optimization every time market unit costs fluctuate is computationally expensive and slow.  

To bypass iterative optimization, we train a Surrogate Machine Learning Model. The model learns the underlying non-linear mapping from 4 primary inputs to 7 structural design parameters generated across your 27×5×5=675 dataset (27 Cost Combinations×5 Span Lengths×5 Runs/Variants).  

Mathematical System Definition
X 
raw
​
 =[ 
C 
c
​
 
​
  
C 
s
​
 
​
  
C 
p
​
 
​
  
L
​
 ] 
ML Pipeline

​
 Y 
target
​
 =[ 
G 
d
​
 
​
  
S
​
  
N 
g
​
 
​
  
P
​
  
Q
​
  
N 
s
​
 
​
  
H 
p
​
 
​
 ]
Inputs (X):

Concrete Unit Cost (C 
c
​
 )  

Steel Rebar Unit Cost (C 
s
​
 )  

Strand Unit Cost (C 
p
​
 )  

Span Length (L)  

Targets (Y):

Girder Depth (G 
d
​
 ) [Column K]  

Lateral Spacing (S) [Column N]  

No. of Girders (N 
g
​
 ) [Column O]  

Bottom Flange Bottom Depth (P) [Column P]  

Bottom Flange Bottom Width (Q) [Column Q]  

Number of Strands (N 
s
​
 ) [Column R]  

Harping Position (H 
p
​
 ) [Column S]  

Senior Machine Learning Engineer Development Roadmap
To achieve high multi-target prediction accuracy (R 
2
 >0.98, RMSE<2%), follow this structured 6-phase engineering pipeline:

[Phase 1: Ingestion & Feature Engineering]
                       │
                       ▼
[Phase 2: Validation Strategy & Preprocessing]
                       │
                       ▼
[Phase 3: Multi-Output Model Benchmark & Tuning]
                       │
                       ▼
[Phase 4: Equation Derivation (PySR / RSM)]
                       │
                       ▼
[Phase 5: Physics & Bound Constraints Filtering]
                       │
                       ▼
[Phase 6: Deployment & Web UI Integration]
Phase 1: Domain-Driven Feature Engineering
Raw inputs alone (C 
c
​
 ,C 
s
​
 ,C 
p
​
 ,L) under-represent structural mechanics. To maximize accuracy, inject domain-specific physical relationships directly into the feature matrix:  

Bending Moment Proxy: Bending moment scales quadratically with span length (M 
u
​
 ∝L 
2
 ). Adding L 
2
  provides a direct linear signal for structural capacity demand.  

Relative Material Cost Ratios:

Ratio 
p/c
​
 = 
C 
c
​
 
C 
p
​
 
​
 ,Ratio 
s/c
​
 = 
C 
c
​
 
C 
s
​
 
​
 

These ratios govern structural substitution behavior (e.g., trading deeper concrete sections G 
d
​
  for fewer steel strands N 
s
​
 ).  

Cross-Domain Interaction Terms:

Interaction 
p⋅L
​
 =( 
C 
c
​
 
C 
p
​
 
​
 )⋅L 
2
 

This term explicitly captures how strand costs penalize long-span designs.  

Phase 2: Data Preprocessing & Validation Strategy
Train-Test Split (80/20): Split the 675 rows into 540 training samples and 135 test samples.  

Stratified Grouping: Ensure that all 5 span length ranges (L) and cost combinations (C 
c
​
 ,C 
s
​
 ,C 
p
​
 ) are uniformly distributed across both training and test sets to prevent data leakage.  

Feature Scaling: Apply StandardScaler or RobustScaler to numerical inputs to ensure zero mean and unit variance before feeding them into gradient-based models.  

Phase 3: Multi-Output Model Selection & Hyperparameter Tuning
Standard single-output regressors ignore correlations between target outputs. Use multi-output frameworks that model cross-target dependencies:  

Candidate Architectures:

MultiOutput XGBoost Regressor: Best overall accuracy for tabular engineering data.  

MultiOutput Random Forest Regressor: Resilient against overfitting on small/medium datasets.  

Multi-Task Neural Network (MLP): A shared hidden layer (128→64 neurons) with 7 distinct output heads, allowing explicit gradient sharing across parameters.

Hyperparameter Optimization (Optuna / GridSearchCV):

Tune XGBoost parameters: max_depth (3 to 8), learning_rate (0.01 to 0.1), n_estimators (100 to 500), subsample (0.7 to 1.0), and colsample_bytree (0.7 to 1.0).

Evaluation Metrics: Evaluated across all 7 targets independently using R 
2
  Score, Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

Phase 4: Symbolic Regression & Explicit Equation Derivation
To provide an interpretable mathematical formulation for spreadsheets or research papers:

PySR (Python Symbolic Regression): Runs genetic algorithms to search the space of algebraic equations, balancing accuracy and complexity.  

Response Surface Methodology (RSM): Fits a second-order polynomial model across the 3 
3
  factorial cost design and span variations:
  

y
^
​
  
k
​
 =β 
0
​
 + 
i=1
∑
4
​
 β 
i
​
 x 
i
​
 + 
i=1
∑
4
​
 β 
ii
​
 x 
i
2
​
 + 
i<j
∑
​
 β 
ij
​
 x 
i
​
 x 
j
​
 
Phase 5: Post-Processing & Physical Constraint Enforcement
Raw ML predictions may yield non-physical or non-constructible values (e.g., N 
g
​
 =4.37 girders or N 
s
​
 =14.2 strands). The inference pipeline must enforce physics-based post-processing filters:  

Integer Rounding: Discrete parameters (N 
g
​
 , N 
s
​
 ) are rounded to the nearest valid integer.  

Standard Modular Steps: Dimensions (G 
d
​
 ,S,P,Q) are snapped to practical construction increments (e.g., 25 mm or 50 mm steps).  

AASHTO Code Boundary Checks: Enforce hard bounds like minimum web thickness W 
w
​
 ≥150 mm and girder depth ratios G 
d
​
 ≥L/20.  

Phase 6: Web UI Integration & Dynamic SVG Rendering
Wrap the trained model or derived equations into a single-page web calculator modal:

User Controls: Sliders or input fields for C 
c
​
 , C 
s
​
 , C 
p
​
 , and L.

Inference Engine: Runs lightweight client-side JavaScript calculations using derived closed-form equations or calls a Python FastAPI backend hosting the .pkl model.

Dynamic Visualization: An SVG canvas draws the I-girder cross-section, adjusts height/width dynamically, and updates tendon positions in real-time.