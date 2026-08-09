"""
Data loading and cleaning module for Girder_Dataset.xlsx.
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
    combined = combined.dropna(subset=INPUT_COLS + TARGET_COLS)

    # Data Cleaning Filters:
    # 1. Filter out Rebar == 1.26 contamination (valid Rebar range is 2.18 - 3.45)
    combined = combined[combined['Rebar'] > 1.5].copy()

    # 2. Filter out solver failure outliers where No. of Gir > 20 (valid range is 6 - 13)
    combined = combined[combined['No. of Gir'] <= 20].copy()

    combined = combined.reset_index(drop=True)
    return combined

if __name__ == "__main__":
    df_clean = load_dataset()
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/clean_dataset.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"[Phase 0] Data loading complete. Clean dataset shape: {df_clean.shape}")
    print(f"Saved to {output_path}")
