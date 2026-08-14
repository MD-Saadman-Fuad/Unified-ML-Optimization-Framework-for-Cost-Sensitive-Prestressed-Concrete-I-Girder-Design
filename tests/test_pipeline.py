"""
Unit tests for data pipeline, feature engineering, and constraint enforcement.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_dataset, INPUT_COLS, TARGET_COLS
from src.features.build_features import build_features, FEATURE_NAMES
from src.postprocess.constraints import enforce_constraints

def test_load_dataset():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 500
    for col in INPUT_COLS + TARGET_COLS:
        assert col in df.columns
        assert df[col].isnull().sum() == 0
    assert (df['Rebar'] > 1.5).all()
    assert (df['No. of Gir'] <= 20).all()

def test_build_features():
    sample_df = pd.DataFrame([{
        'Concrete': 505.0,
        'Strand': 1.73,
        'Rebar': 2.18,
        'Span_ft': 140.0
    }])
    X = build_features(sample_df)
    assert list(X.columns) == FEATURE_NAMES
    assert X['L_sq'].iloc[0] == 140.0 ** 2
    assert X['ratio_strand_concrete'].iloc[0] == pytest.approx(1.73 / 505.0)
    assert X['ratio_rebar_concrete'].iloc[0] == pytest.approx(2.18 / 505.0)
    assert X['interaction_strand_L'].iloc[0] == pytest.approx((1.73 / 505.0) * (140.0 ** 2))

def test_enforce_constraints():
    raw_pred = {
        "Gir Dep (in)": 50.3,
        "Lat Spac (ft)": 6.13,
        "No. of Gir": 7.6,
        "bot flange bot part depth (in)": 8.12,
        "bot flange bot part width (in)": 37.89,
        "Number of strand per girder": 71.3,
        "Harp Pos (ft)": 49.8
    }
    L_ft = 140.0
    processed = enforce_constraints(raw_pred, L_ft)

    # Physical bounds [45.0, 72.0] and 0.5-in snapping
    assert 45.0 <= processed["Gir Dep (in)"] <= 72.0
    assert processed["Gir Dep (in)"] == 50.5
    # No. of Gir should round to 8
    assert processed["No. of Gir"] == 8
    # Strands should round to even integer 72
    assert processed["Number of strand per girder"] == 72
    assert processed["Number of strand per girder"] % 2 == 0
    # Structural checks present and satisfied
    assert "structural_checks" in processed
    assert processed["structural_checks"]["all_satisfied"] is True


def test_no_data_leakage():
    from src.preprocessing.split_and_scale import split_and_scale_data
    X_train, X_test, Y_train, Y_test, scaler, cols = split_and_scale_data()
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0
