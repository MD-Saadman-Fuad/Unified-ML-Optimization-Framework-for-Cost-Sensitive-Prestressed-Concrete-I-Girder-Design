"""
Data preprocessing, train/test splitting, and feature scaling module.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.load_data import load_dataset, TARGET_COLS
from src.features.build_features import build_features

def split_and_scale_data(filepath: str = "Girder_Dataset.xlsx", test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into stratified train/test sets and fits StandardScaler on training data.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, feature_names)
    """
    df = load_dataset(filepath)
    X = build_features(df)
    Y = df[TARGET_COLS].copy()

    # Stratified 80/20 train/test split on Span_ft to ensure balanced representation across span lengths
    X_train, X_test, Y_train, Y_test, train_idx, test_idx = train_test_split(
        X, Y, df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df['Span_ft']
    )

    # Fit scaler on training features ONLY
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/splits", exist_ok=True)

    joblib.dump(scaler, "models/scaler.pkl")
    np.save("data/splits/train_idx.npy", train_idx.to_numpy())
    np.save("data/splits/test_idx.npy", test_idx.to_numpy())

    X_train_scaled.to_csv("data/splits/X_train_scaled.csv")
    X_test_scaled.to_csv("data/splits/X_test_scaled.csv")
    Y_train.to_csv("data/splits/Y_train.csv")
    Y_test.to_csv("data/splits/Y_test.csv")

    return X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, X.columns.tolist()

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, Y_train, Y_test, scaler, cols = split_and_scale_data()
    print(f"[Phase 2] Preprocessing & splitting complete.")
    print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
    print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")
    print(f"Fitted scaler saved to models/scaler.pkl")
