"""
Physical and AASHTO structural code constraint enforcement module.
"""
import sys
import os
import math

def enforce_constraints(pred: dict, L_ft: float) -> dict:
    """
    Applies physical, constructibility, and AASHTO code constraints to raw ML model predictions.

    Parameters
    ----------
    pred : dict
        Dictionary of raw model predictions for target columns.
    L_ft : float
        Span length in feet.

    Returns
    -------
    dict
        Post-processed predictions with enforced physical bounds.
    """
    out = dict(pred)

    # 1. Discrete Parameter: Number of Girders (Ng) -> Integer in range [6, 13]
    if "No. of Gir" in out:
        ng_val = int(round(out["No. of Gir"]))
        out["No. of Gir"] = max(6, min(13, ng_val))

    # 2. Discrete Parameter: Number of Strands per Girder (Ns) -> Even integer in range [32, 122]
    if "Number of strand per girder" in out:
        ns_val = out["Number of strand per girder"]
        even_ns = int(round(ns_val / 2.0) * 2)
        out["Number of strand per girder"] = max(32, min(122, even_ns))

    # 3. AASHTO Code Bound: Minimum Girder Depth Gd >= 0.045 * L (both in feet; convert Gd to inches)
    # Min depth in inches = 0.045 * L_ft * 12
    if "Gir Dep (in)" in out:
        min_gd_in = 0.045 * L_ft * 12.0
        # Snap girder depth to nearest 0.5 in increment
        snapped = round(out["Gir Dep (in)"] * 2.0) / 2.0
        out["Gir Dep (in)"] = max(min_gd_in, snapped)

    # 4. Constructibility Snapping for flange dimensions
    if "bot flange bot part depth (in)" in out:
        out["bot flange bot part depth (in)"] = max(0.0, round(out["bot flange bot part depth (in)"] * 2.0) / 2.0)

    if "bot flange bot part width (in)" in out:
        out["bot flange bot part width (in)"] = max(6.0, round(out["bot flange bot part width (in)"] * 2.0) / 2.0)

    if "Lat Spac (ft)" in out:
        out["Lat Spac (ft)"] = max(2.0, round(out["Lat Spac (ft)"] * 4.0) / 4.0)  # Snap to 0.25 ft

    if "Harp Pos (ft)" in out:
        out["Harp Pos (ft)"] = max(0.0, min(L_ft, round(out["Harp Pos (ft)"] * 2.0) / 2.0))

    return out

if __name__ == "__main__":
    sample_raw_pred = {
        "Gir Dep (in)": 55.3,
        "Lat Spac (ft)": 6.13,
        "No. of Gir": 7.4,
        "bot flange bot part depth (in)": 8.12,
        "bot flange bot part width (in)": 37.89,
        "Number of strand per girder": 71.3,
        "Harp Pos (ft)": 49.8
    }
    L_ft = 120.0
    processed = enforce_constraints(sample_raw_pred, L_ft)
    print("[Phase 5] Constraint enforcement test:")
    print("Raw Pred:", sample_raw_pred)
    print("Enforced Pred:", processed)
