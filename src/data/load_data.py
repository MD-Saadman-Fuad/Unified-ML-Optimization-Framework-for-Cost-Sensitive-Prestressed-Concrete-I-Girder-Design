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

def load_primary_dataset(filepath: str = "Girder_Dataset.xlsx") -> pd.DataFrame:
    """
    Loads only Alternative 1 (the optimal design for each cost-span combination).
    Yields 135 deterministic optimal designs across 5 span lengths.
    """
    df = load_dataset(filepath)
    df_primary = df[df['Design_Rank'] == 1].copy().reset_index(drop=True)
    return df_primary

def load_dataset_averaged(filepath: str = "Girder_Dataset.xlsx") -> pd.DataFrame:
    """
    Compatibility wrapper returning load_primary_dataset (Alternative 1).
    """
    return load_primary_dataset(filepath)

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

