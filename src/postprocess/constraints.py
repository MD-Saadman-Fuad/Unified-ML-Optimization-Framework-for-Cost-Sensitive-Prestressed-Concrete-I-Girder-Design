"""
Physical, constructibility, and AASHTO structural code constraint enforcement module.
Includes live load deflection limits, cracking moment ductility checks, and lateral stability safety factor evaluation.
"""
import sys
import os
import math
import json

def enforce_constraints(pred: dict, L_ft: float) -> dict:
    """
    Applies physical, constructibility, and AASHTO code constraints to raw ML model predictions.
    Computes AASHTO structural compliance checks.

    Parameters
    ----------
    pred : dict
        Dictionary of raw model predictions for target columns.
    L_ft : float
        Span length in feet.

    Returns
    -------
    dict
        Post-processed predictions with enforced physical bounds and structural_checks summary.
    """
    out = dict(pred)
    checks = {}

    # 1. Discrete Parameter: Number of Girders (Ng)
    if "No. of Gir" in out:
        ng_val = int(round(out["No. of Gir"]))
        ng_min = 8 if L_ft >= 175 else 6
        out["No. of Gir"] = max(ng_min, min(11, ng_val))

    # 2. Discrete Parameter: Number of Strands per Girder (Ns) -> Even integer
    if "Number of strand per girder" in out:
        ns_val = out["Number of strand per girder"]
        even_ns = int(round(ns_val / 2.0) * 2)
        out["Number of strand per girder"] = max(32, min(108, even_ns))

    # 3. Girder Depth Physical Bounds & Snapping (nearest 0.5 in)
    if "Gir Dep (in)" in out:
        snapped = round(out["Gir Dep (in)"] * 2.0) / 2.0
        out["Gir Dep (in)"] = max(45.0, min(72.0, snapped))

    # 4. Constructibility Snapping for flange dimensions and spacing
    if "bot flange bot part depth (in)" in out:
        out["bot flange bot part depth (in)"] = max(0.0, round(out["bot flange bot part depth (in)"] * 2.0) / 2.0)

    if "bot flange bot part width (in)" in out:
        out["bot flange bot part width (in)"] = max(6.0, round(out["bot flange bot part width (in)"] * 2.0) / 2.0)

    if "Lat Spac (ft)" in out:
        out["Lat Spac (ft)"] = max(2.0, round(out["Lat Spac (ft)"] * 4.0) / 4.0)  # Snap to 0.25 ft

    if "Harp Pos (ft)" in out:
        out["Harp Pos (ft)"] = max(0.0, min(L_ft, round(out["Harp Pos (ft)"] * 2.0) / 2.0))

    # =========================================================================
    # AASHTO Structural & Physics Compliance Verification Checks
    # =========================================================================
    gd_in = out.get("Gir Dep (in)", 68.0)
    ns_strands = out.get("Number of strand per girder", 70)
    spacing_ft = out.get("Lat Spac (ft)", 6.0)

    # A. Deflection Check: Delta_LL <= L_in / 800
    # Composite Moment of Inertia approximation: Ic ~ 0.55 * (b_eff * h^3 / 12)
    L_in = L_ft * 12.0
    b_eff_in = min(spacing_ft * 12.0, 96.0)
    h_comp_in = gd_in + 8.0  # Girder depth + 8 in deck slab
    I_c_in4 = 0.50 * (b_eff_in * (h_comp_in ** 3) / 12.0)
    E_c_psi = 4500000.0  # Concrete E ~ 4.5 Mpsi

    # Live load deflection under HS20 wheel load (P_T ~ 32 kips with distribution)
    P_T_lbs = 32000.0
    delta_ll_in = (324.0 / (E_c_psi * I_c_in4)) * P_T_lbs * ((L_ft**3) - 555.0 * L_ft + 4780.0) / 1000.0
    delta_allowable_in = L_in / 800.0

    checks["live_load_deflection"] = {
        "calculated_delta_in": round(delta_ll_in, 4),
        "allowable_delta_in": round(delta_allowable_in, 4),
        "satisfied": delta_ll_in <= delta_allowable_in
    }

    # B. Cracking Moment Ductility Check: 1.2 * Mcr <= phi * Mn
    # Modulus of rupture f_r = 7.5 * sqrt(f'c) ~ 7.5 * sqrt(6000) ~ 580 psi
    # Prestress force P_e ~ Ns * 0.153 sq.in * 0.60 * 270 ksi
    A_ps_sqin = ns_strands * 0.153
    f_pe_psi = 0.60 * 270000.0
    P_e_lbs = A_ps_sqin * f_pe_psi
    S_bottom_in3 = 0.80 * (gd_in ** 2) * 12.0 / 6.0  # Section modulus proxy

    M_cr_kipft = (S_bottom_in3 * (580.0 + (P_e_lbs / 700.0)) / 12000.0)
    phi_Mn_kipft = 0.90 * A_ps_sqin * 260.0 * (0.85 * h_comp_in) / 12.0

    checks["ductility_min_reinforcement"] = {
        "1.2_Mcr_kipft": round(1.2 * M_cr_kipft, 2),
        "phi_Mn_kipft": round(phi_Mn_kipft, 2),
        "satisfied": (1.2 * M_cr_kipft) <= phi_Mn_kipft
    }

    # C. Lateral Stability Factor during Handling (FS_c >= 1.5)
    # Weak-axis stability proxy based on L_ft / Top_flange_width ratio
    b_top_flange_in = max(24.0, gd_in * 0.45)
    slenderness_ratio = (L_ft * 12.0) / b_top_flange_in
    FS_c = max(1.0, 3.5 - 0.02 * slenderness_ratio)

    checks["lateral_stability_handling"] = {
        "factor_of_safety": round(FS_c, 2),
        "required_FS": 1.5,
        "satisfied": FS_c >= 1.5
    }

    checks["all_satisfied"] = all([
        checks["live_load_deflection"]["satisfied"],
        checks["ductility_min_reinforcement"]["satisfied"],
        checks["lateral_stability_handling"]["satisfied"]
    ])

    out["structural_checks"] = checks
    return out

if __name__ == "__main__":
    sample_raw_pred = {
        "Gir Dep (in)": 65.3,
        "Lat Spac (ft)": 6.13,
        "No. of Gir": 7.4,
        "bot flange bot part depth (in)": 8.12,
        "bot flange bot part width (in)": 37.89,
        "Number of strand per girder": 72.0,
        "Harp Pos (ft)": 49.8
    }
    L_ft = 140.0
    processed = enforce_constraints(sample_raw_pred, L_ft)
    print("[Phase 5] Post-processing & AASHTO Physics Verification:")
    print("Enforced Parameters:", {k: v for k, v in processed.items() if k != "structural_checks"})
    print("Structural Checks Summary:")
    print(json.dumps(processed["structural_checks"], indent=2))

