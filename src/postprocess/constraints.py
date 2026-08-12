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

    # 1. Discrete Parameter: Number of Girders (Ng)
    #    Dataset analysis on averaged optimal designs shows:
    #      Spans 100-160 ft: optimal Ng is 6-8 (mean ~7)
    #      Span 180 ft:      optimal Ng is 8-11 (mean ~10)
    #    Hard bounds [6, 11] — the raw dataset shows max 11, never 12-13 in averaged optima.
    if "No. of Gir" in out:
        ng_val = int(round(out["No. of Gir"]))
        # Span-aware minimum: 180ft span needs at least 8 girders
        ng_min = 8 if L_ft >= 175 else 6
        out["No. of Gir"] = max(ng_min, min(11, ng_val))

    # 2. Discrete Parameter: Number of Strands per Girder (Ns) -> Even integer
    #    Dataset analysis on averaged optimal designs:
    #      100ft: 36-44  |  120ft: 50-60  |  140ft: 66-74
    #      160ft: 86-98  |  180ft: 82-108
    #    Overall range in averaged optima: [36, 108]. Hard clamp to [32, 108].
    if "Number of strand per girder" in out:
        ns_val = out["Number of strand per girder"]
        even_ns = int(round(ns_val / 2.0) * 2)
        out["Number of strand per girder"] = max(32, min(108, even_ns))

    # 3. Girder Depth Physical Bounds:
    #    - Lower floor: 45.0 in (practical minimum for prestressed concrete I-girders in this dataset)
    #    - Upper cap:   72.0 in (AASHTO Type VI standard beam maximum depth — hard limit in the
    #                            optimization that generated this dataset; the formula 0.045*L*12
    #                            was NOT used as the binding constraint here because it produces
    #                            values up to 97.2 in for 180-ft spans which exceed the 72-in
    #                            dataset maximum for all span lengths)
    #    - Snapping: nearest 0.5-in construction increment
    if "Gir Dep (in)" in out:
        snapped = round(out["Gir Dep (in)"] * 2.0) / 2.0
        out["Gir Dep (in)"] = max(45.0, min(72.0, snapped))

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
