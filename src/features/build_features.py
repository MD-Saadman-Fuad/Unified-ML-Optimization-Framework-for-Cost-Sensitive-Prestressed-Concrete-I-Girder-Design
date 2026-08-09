"""
Domain-driven feature engineering module for prestressed concrete I-girder surrogate model.
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
    'ratio_strand_concrete',
    'ratio_rebar_concrete',
    'interaction_strand_L'
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs 8 domain-engineered features from raw inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing ['Concrete', 'Strand', 'Rebar', 'Span_ft'].

    Returns
    -------
    pd.DataFrame
        DataFrame with 8 physical feature columns.
    """
    # Verify input columns presence
    missing = [c for c in INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns for feature engineering: {missing}")

    X = df[INPUT_COLS].copy()

    # 1. Bending Moment Proxy: Mu scales quadratically with span length (L^2)
    X['L_sq'] = X['Span_ft'] ** 2

    # 2. Relative Material Cost Ratios (Strand/Concrete, Rebar/Concrete)
    X['ratio_strand_concrete'] = X['Strand'] / X['Concrete']
    X['ratio_rebar_concrete'] = X['Rebar'] / X['Concrete']

    # 3. Cross-Domain Interaction Term: Strand cost penalty on long spans
    X['interaction_strand_L'] = X['ratio_strand_concrete'] * X['L_sq']

    return X[FEATURE_NAMES]

if __name__ == "__main__":
    from src.data.load_data import load_dataset
    df_clean = load_dataset()
    X_features = build_features(df_clean)
    os.makedirs("data/processed", exist_ok=True)
    X_features.to_csv("data/processed/features.csv", index=False)
    print(f"[Phase 1] Feature engineering complete. Features shape: {X_features.shape}")
    print(f"Features list: {list(X_features.columns)}")
