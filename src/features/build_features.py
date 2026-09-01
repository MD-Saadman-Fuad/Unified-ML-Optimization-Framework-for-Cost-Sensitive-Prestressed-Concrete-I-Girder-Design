"""
Domain-driven feature engineering module for prestressed concrete I-girder surrogate model.
Expands raw cost and span inputs into a 14-dimensional structural mechanics feature vector.
"""
import sys
import os
import pandas as pd

# Add project root to sys.path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

INPUT_COLS = ['Concrete', 'Strand', 'Rebar', 'Span_ft']

FEATURE_NAMES = [
    'Concrete',
    'Strand',
    'Rebar',
    'Span_ft',
    'L_sq',
    'M_LL_proxy',
    'ratio_strand_concrete',
    'ratio_rebar_concrete',
    'ratio_rebar_strand',
    'interaction_strand_L',
    'volumetric_stiffness',
    'span_concrete_ratio',
    'span_strand_stiffness',
    'elasticity_index'
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs 14 domain-engineered features from raw inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ['Concrete', 'Strand', 'Rebar', 'Span_ft'].

    Returns
    -------
    pd.DataFrame
        DataFrame with 14 physical feature columns.
    """
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns for feature engineering: {missing}")

    X = df[INPUT_COLS].copy()

    # 1. Uniform Dead Load Bending Moment Proxy (Mu proportional to L^2)
    X['L_sq'] = X['Span_ft'] ** 2

    # 2. AASHTO HS20 Live Load Bending Moment Proxy (MLL proportional to 0.72*L + 10)
    X['M_LL_proxy'] = 0.72 * X['Span_ft'] + 10.0

    # 3. Relative Material Cost Ratios
    X['ratio_strand_concrete'] = X['Strand'] / X['Concrete']
    X['ratio_rebar_concrete'] = X['Rebar'] / X['Concrete']
    X['ratio_rebar_strand'] = X['Rebar'] / X['Strand']

    # 4. Cross-Domain Interaction Terms
    X['interaction_strand_L'] = X['ratio_strand_concrete'] * X['L_sq']
    X['volumetric_stiffness'] = X['Concrete'] / (X['Strand'] * X['L_sq'] + 1.0)
    X['span_concrete_ratio'] = X['Span_ft'] / X['Concrete']
    X['span_strand_stiffness'] = (X['Span_ft'] ** 3) / (X['Strand'] * 1000.0)
    X['elasticity_index'] = (X['Rebar'] * X['Strand']) / (X['Concrete'] ** 2)

    return X[FEATURE_NAMES]

if __name__ == "__main__":
    from src.data.load_data import load_dataset
    df_clean = load_dataset()
    X_features = build_features(df_clean)
    os.makedirs("data/processed", exist_ok=True)
    X_features.to_csv("data/processed/features.csv", index=False)
    print(f"[Phase 1] Feature engineering complete. Features shape: {X_features.shape}")
    print(f"Features list: {list(X_features.columns)}")

