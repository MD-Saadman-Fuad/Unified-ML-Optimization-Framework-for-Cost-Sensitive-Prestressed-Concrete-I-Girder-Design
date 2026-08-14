"""
Data preprocessing, group-based train/test splitting, and feature scaling module.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_dataset, load_primary_dataset, TARGET_COLS
from src.features.build_features import build_features

def split_and_scale_data(filepath: str = "Girder_Dataset.xlsx", test_size: float = 0.2, random_state: int = 42):
    """
    Loads Primary (Alt 1) dataset, splits into stratified train/test sets by Span_ft,
    and fits StandardScaler on training features.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, feature_names)
    """
    df = load_primary_dataset(filepath)
    X = build_features(df)
    Y = df[TARGET_COLS].copy()

    print(f"  [Preprocessing] Primary Alt-1 dataset: {len(df)} unique optimal cost-span designs")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Span_ft']
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    X_train_scaled.to_csv("data/splits/X_train_scaled.csv")
    X_test_scaled.to_csv("data/splits/X_test_scaled.csv")
    Y_train.to_csv("data/splits/Y_train.csv")
    Y_test.to_csv("data/splits/Y_test.csv")

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, X.columns.tolist()


def split_and_scale_all_alternatives(filepath: str = "Girder_Dataset.xlsx", test_size: float = 0.2, random_state: int = 42):
    """
    Loads all alternatives (595 rows), performs group-based split on Family_ID
    so that 100% of rows for a given cost-span family remain together in Train or Test.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, feature_names, train_groups, test_groups)
    """
    df = load_dataset(filepath)
    X = build_features(df)
    # Add Design_Rank as explicit feature for secondary model
    X['Design_Rank'] = df['Design_Rank']
    Y = df[TARGET_COLS].copy()
    groups = df['Family_ID'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, Y, groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    Y_train, Y_test = Y.iloc[train_idx].copy(), Y.iloc[test_idx].copy()

    # Confirm zero leakage between Train and Test families
    train_fams = set(df.iloc[train_idx]['Family_ID'])
    test_fams = set(df.iloc[test_idx]['Family_ID'])
    assert len(train_fams.intersection(test_fams)) == 0, "DATA LEAKAGE DETECTED: Overlapping family IDs!"

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    joblib.dump(scaler, "models/secondary_scaler.pkl")

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, X.columns.tolist()

if __name__ == "__main__":
    X_tr, X_te, Y_tr, Y_te, sc, cols = split_and_scale_data()
    print(f"[Phase 2] Primary splitting complete. X_train: {X_tr.shape}, X_test: {X_te.shape}")
    X_tr_all, X_te_all, Y_tr_all, Y_te_all, sc_all, cols_all = split_and_scale_all_alternatives()
    print(f"[Phase 2] All-alternatives Grouped splitting complete. X_train_all: {X_tr_all.shape}, X_test_all: {X_te_all.shape}")


