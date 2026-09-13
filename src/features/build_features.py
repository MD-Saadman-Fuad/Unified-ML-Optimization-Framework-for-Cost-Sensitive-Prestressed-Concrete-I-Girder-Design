"""
Domain-driven feature engineering module for prestressed concrete I-girder surrogate model.
Expands raw cost and span inputs into an 11-dimensional validated feature vector.

Feature selection rationale (engineering review):
  KEEP   : L², L³ (span demand proxies — well established in structural mechanics)
  KEEP   : Cp/Cc, Cs/Cc, Cs/Cp (unit-cost ratios — dataset directly varies these)
  KEEP   : (Cp/Cc)×L² (cost-span interaction — L² variant, tested and validated)
  KEEP   : Cp·Cs/Cc² (composite cost interaction term, renamed from 'elasticity_index')
  REMOVED: M_LL_proxy — not a real AASHTO equation; redundant with L
  REMOVED: strand_span_cube (Cp×L³) — L³ interaction not validated vs L² variant
  REMOVED: volumetric_stiffness, span_concrete_ratio, span_strand_stiffness — no
           engineering evidence these ratio forms represent real relationships
  REMOVED: total_cost_index — artificial 100×/1000× coefficients, not quantity-based
  REMOVED: concrete_cost_share — depends on artificial total cost; meaningless without
           real material quantities from design
"""
import sys
import os
import pandas as pd

# Add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

INPUT_COLS = ['Concrete', 'Strand', 'Rebar', 'Span_ft']

FEATURE_NAMES = [
    # Raw inputs
    'Concrete',              # Cc — concrete unit cost ($/yd³)
    'Strand',                # Cp — prestressing strand unit cost ($/linear ft per strand)
    'Rebar',                 # Cs — rebar unit cost ($/lb)
    'Span_ft',               # L  — span length (ft)
    # Span demand proxies
    'L_sq',                  # L² — bending moment demand (M ∝ wL²)
    'L_cube',                # L³ — nonlinear deflection/demand proxy
    # Unit cost ratios
    'ratio_strand_concrete', # Cp/Cc — strand-to-concrete unit cost ratio
    'ratio_rebar_concrete',  # Cs/Cc — rebar-to-concrete unit cost ratio
    'ratio_rebar_strand',    # Cs/Cp — rebar-to-strand cross-steel unit cost ratio
    # Cost-span interaction (L² only — L³ variant not validated)
    'interaction_strand_L',  # (Cp/Cc)×L² — strand cost pressure × span demand
    # Composite cost interaction
    'cost_interaction',      # Cp·Cs/Cc² — joint steel cost relative to concrete (renamed)
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs 11 domain-engineered features from raw inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ['Concrete', 'Strand', 'Rebar', 'Span_ft'].

    Returns
    -------
    pd.DataFrame
        DataFrame with 11 validated feature columns (see FEATURE_NAMES).
    """
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns for feature engineering: {missing}")

    X = df[INPUT_COLS].copy()

    # 1. Span polynomial terms
    X['L_sq']   = X['Span_ft'] ** 2
    X['L_cube'] = X['Span_ft'] ** 3  # Cubic span proxy — nonlinear deflection/demand

    # 2. Unit cost ratios (economic sensitivity features)
    X['ratio_strand_concrete'] = X['Strand'] / X['Concrete']   # Cp/Cc
    X['ratio_rebar_concrete']  = X['Rebar']  / X['Concrete']   # Cs/Cc
    X['ratio_rebar_strand']    = X['Rebar']  / X['Strand']     # Cs/Cp

    # 3. Cost-span interaction (L² variant only — validated; L³ variant removed)
    X['interaction_strand_L'] = X['ratio_strand_concrete'] * X['L_sq']  # (Cp/Cc)×L²

    # 4. Composite cost interaction (renamed from elasticity_index — avoids misleading label)
    X['cost_interaction'] = (X['Rebar'] * X['Strand']) / (X['Concrete'] ** 2)  # Cp·Cs/Cc²

    return X[FEATURE_NAMES]


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    df_clean = load_dataset()
    X_features = build_features(df_clean)
    os.makedirs("data/processed", exist_ok=True)
    X_features.to_csv("data/processed/features.csv", index=False)
    print(f"[Phase 1] Feature engineering complete. Features shape: {X_features.shape}")
    print(f"Features list: {list(X_features.columns)}")
