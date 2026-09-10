"""
Data loading and cleaning module for Girder_Dataset.xlsx.
Supports multi-alternative datasets, design ranking (1-5), and group family identification.
"""
import os
import pandas as pd

SHEET_CONFIG = {
    '100': {'header_row': 3, 'span_ft': 100},
    '120': {'header_row': 2, 'span_ft': 120},
    '140': {'header_row': 3, 'span_ft': 140},
    '160': {'header_row': 2, 'span_ft': 160},
    '180': {'header_row': 3, 'span_ft': 180},
}

INPUT_COLS = ['Concrete', 'Strand', 'Rebar', 'Span_ft']
TARGET_COLS = [
    'Gir Dep (in)',
    'Lat Spac (ft)',
    'No. of Gir',
    'bot flange bot part depth (in)',
    'bot flange bot part width (in)',
    'Number of strand per girder',
    'Harp Pos (ft)'
]

def load_dataset(filepath: str = "Girder_Dataset.xlsx") -> pd.DataFrame:
    """
    Loads, cleans, and merges all span sheets from Girder_Dataset.xlsx.
    Assigns Design_Rank (1-5) and Family_ID for group-based cross validation.
    
    Parameters
    ----------
    filepath : str
        Path to the Excel file.
        
    Returns
    -------
    pd.DataFrame
        Cleaned combined dataframe ready for feature engineering.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")

    frames = []
    for sheet, cfg in SHEET_CONFIG.items():
        df = pd.read_excel(filepath, sheet_name=sheet, header=cfg['header_row'])
        # Strip leading/trailing whitespaces from column names
        df.columns = df.columns.astype(str).str.strip()
        df['Span_ft'] = float(cfg['span_ft'])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Select core columns
    keep_cols = INPUT_COLS + TARGET_COLS
    if 'Combo' in combined.columns:
        keep_cols.append('Combo')
    
    combined = combined[[c for c in keep_cols if c in combined.columns]].copy()

    # Drop NaNs in core input and target columns
    combined = combined.dropna(subset=INPUT_COLS + TARGET_COLS).copy()

    # Data Cleaning Filters:
    # 1. Filter out Rebar == 1.26 contamination (valid Rebar range is 2.18 - 3.45)
    combined = combined[combined['Rebar'] > 1.5].copy()

    # 2. Filter out solver failure outliers where No. of Gir > 20 (valid range is 6 - 13)
    combined = combined[combined['No. of Gir'] <= 20].copy()

    # Assign Design_Rank (1-5) based on sequence within each (Concrete, Strand, Rebar, Span_ft) family
    combined['Design_Rank'] = combined.groupby(INPUT_COLS).cumcount() + 1
    
    # Assign unique string Family_ID for group-based train/test splitting
    combined['Family_ID'] = (
        combined['Concrete'].astype(str) + "_" +
        combined['Strand'].astype(str) + "_" +
        combined['Rebar'].astype(str) + "_" +
        combined['Span_ft'].astype(str)
    )

    combined = combined.reset_index(drop=True)
    return combined

def load_dataset_averaged(filepath: str = "Girder_Dataset.xlsx") -> pd.DataFrame:
    """
    Loads all 5 alternatives per (Concrete, Strand, Rebar, Span_ft) family and
    computes the MEAN of target values across all runs, yielding 119 deterministic
    noise-reduced rows (one per unique cost-span combination).

    This is the correct primary training path.

    Why average?
    -----------
    The raw dataset contains 5 stochastic optimizer runs per cost-span combination.
    Using any single run (e.g., rank=1) introduces optimizer noise as if it were signal,
    causing the model to learn random variation rather than true cost-response relationships.
    Averaging removes within-combination variance and enables genuine generalization.

    Returns
    -------
    pd.DataFrame
        119-row averaged dataframe with INPUT_COLS + TARGET_COLS.
    """
    df = load_dataset(filepath)
    averaged = (
        df.groupby(INPUT_COLS, sort=False)[TARGET_COLS]
        .mean()
        .reset_index()
    )
    averaged = averaged.reset_index(drop=True)
    return averaged


def load_primary_dataset(filepath: str = "Girder_Dataset.xlsx") -> pd.DataFrame:
    """
    Returns the averaged primary dataset (mean across all 5 optimizer runs per family).
    This is the canonical training dataset for the primary surrogate model.

    NOTE: Previously returned rank=1 only (one stochastic run).
    Now correctly delegates to load_dataset_averaged() for noise-reduced labels.
    """
    return load_dataset_averaged(filepath)

if __name__ == "__main__":
    df_clean = load_dataset()
    df_primary = load_primary_dataset()
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/clean_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"[Phase 0] Data loading complete.")
    print(f"Clean all-alternative dataset shape: {df_clean.shape}")
    print(f"Primary Alt-1 optimal dataset shape: {df_primary.shape}")
    print(f"Design Ranks present: {sorted(df_clean['Design_Rank'].unique())}")
    print(f"Unique Cost-Span Families: {df_clean['Family_ID'].nunique()}")

