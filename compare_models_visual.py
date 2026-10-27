# coding: utf-8
"""
Model Comparison: Old RSM (GP.py) vs New XGBoost
4 Shared Parameters: Girder Depth, No. of Strands, Harping Position, Lateral Spacing
3 Figures: (1) Actual vs Predicted, (2) Metrics Bar Chart, (3) Per-sample Errors
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

sys.path.insert(0, ".")
from src.data.load_data import INPUT_COLS, TARGET_COLS
from src.preprocessing.split_and_scale import split_and_scale_data
from src.data.load_data import load_dataset_averaged
from sklearn.model_selection import train_test_split

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.linewidth": 0.8,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "font.size": 10,
})
OLD = "#ff7b72"
NEW = "#3fb950"
DIA = "#ffa657"

# ── Old RSM Coefficients (from previous work / GP.py) ─────────
GD_COEF = {
    "intercept": 136.601234, "L": -0.357148, "C": -0.019188, "G": -10.393466,
    "L2": 0.000015, "LC": 0.000063, "LG": 0.056663,
    "C2": 0.000016, "CG": -0.000976, "G2": 0.041604,
}
NG_COEF = {
    "intercept": 67.451976, "D": -4.230516, "L": 0.538016, "L2": -0.013329,
    "LD": 0.034739, "D2": 0.039728, "L3": 0.000029,
    "L2D": 0.000034, "LD2": -0.000327, "D3": -0.000047,
}
NS_COEF = {
    "intercept": 10.838986, "L": 0.005932, "C": -0.070040, "G": 6.556031, "D": 1.043756,
    "L2": 0.005273, "LC": 0.000049, "LG": 0.010738, "LD": -0.008008,
    "C2": 0.000073, "CG": -0.001015, "CD": 0.000042,
    "G2": -0.382649, "GD": -0.119622, "D2": -0.001395,
}


def rsm_gd(L, C, G):
    b = GD_COEF
    return (b["intercept"] + b["L"]*L + b["C"]*C + b["G"]*G
            + b["L2"]*L**2 + b["LC"]*L*C + b["LG"]*L*G
            + b["C2"]*C**2 + b["CG"]*C*G + b["G2"]*G**2)


def rsm_ng(L, D):
    b = NG_COEF
    return (b["intercept"] + b["D"]*D + b["L"]*L + b["L2"]*L**2
            + b["LD"]*L*D + b["D2"]*D**2 + b["L3"]*L**3
            + b["L2D"]*L**2*D + b["LD2"]*L*D**2 + b["D3"]*D**3)


def rsm_ns_fn(L, C, G, D):
    b = NS_COEF
    return (b["intercept"] + b["L"]*L + b["C"]*C + b["G"]*G + b["D"]*D
            + b["L2"]*L**2 + b["LC"]*L*C + b["LG"]*L*G + b["LD"]*L*D
            + b["C2"]*C**2 + b["CG"]*C*G + b["CD"]*C*D
            + b["G2"]*G**2 + b["GD"]*G*D + b["D2"]*D**2)


def rsm_hp_fn(Sc, D, G, P):
    return (32.129720 - 28.424216*Sc + 0.428984*D + 0.618971*G
            + 0.263758*P + 1.827877*Sc**2
            + 0.332784*Sc*D + 0.768138*Sc*G - 0.091321*Sc*P
            - 0.010078*D**2 + 0.064587*D*G - 0.002618*D*P
            - 0.462759*G**2 + 0.046292*G*P - 0.000003*P**2)


def rsm_iter(L, C, n=20):
    G = max(6, round(L / 20))
    for _ in range(n):
        D = rsm_gd(L, C, G)
        Gn = round(rsm_ng(L, D))
        if Gn == G:
            break
        G = Gn
    D = rsm_gd(L, C, G)
    S = rsm_ns_fn(L, C, G, D)
    return D, G, S


# ── Load Data ─────────────────────────────────────────────────
print("Loading data...")
X_train, X_test, Y_train, Y_test, scaler, fn = split_and_scale_data()
model = joblib.load("models/best_model.pkl")

df = load_dataset_averaged()
_, Xraw, _, _ = train_test_split(
    df[INPUT_COLS], df[TARGET_COLS],
    test_size=0.2, random_state=42, stratify=df["Span_ft"]
)
print("Test samples: " + str(len(Y_test)))

# ── RSM Predictions (Old Model) ───────────────────────────────
r_gd, r_lat, r_ns, r_hp = [], [], [], []
for _, row in Xraw.iterrows():
    L, C, Sc = row["Span_ft"], row["Concrete"], row["Strand"]
    D, G, S = rsm_iter(L, C)
    HP = rsm_hp_fn(Sc, D, G, S)
    r_gd.append(D)
    r_ns.append(S)
    r_hp.append(HP)
    r_lat.append(37.0 / max(G - 1, 1))

r_gd = np.array(r_gd)
r_ns = np.array(r_ns)
r_hp = np.array(r_hp)
r_lat = np.array(r_lat)

# ── Actual Ground Truth ───────────────────────────────────────
a_gd = Y_test["Gir Dep (in)"].values
a_ns = Y_test["Number of strand per girder"].values
a_hp = Y_test["Harp Pos (ft)"].values
a_lat = Y_test["Lat Spac (ft)"].values

# ── New Unified ML Predictions ────────────────────────────────
new_preds = pd.DataFrame(model.predict(X_test), columns=Y_test.columns, index=Y_test.index)
x_gd = new_preds["Gir Dep (in)"].values
x_ns = new_preds["Number of strand per girder"].values
x_hp = new_preds["Harp Pos (ft)"].values
x_lat = new_preds["Lat Spac (ft)"].values


def met(a, p):
    return {
        "MAE": float(mean_absolute_error(a, p)),
        "RMSE": float(np.sqrt(mean_squared_error(a, p))),
        "R2": float(r2_score(a, p)),
    }


PARAMS = [
    ("Girder Depth (in)", a_gd, r_gd, x_gd),
    ("No. of Strands", a_ns, r_ns, x_ns),
    ("Harping Position (ft)", a_hp, r_hp, x_hp),
    ("Lateral Spacing (ft)", a_lat, r_lat, x_lat),
]

# Print metrics table
print()
print("=" * 72)
print("Parameter                    | Model            |     R2 |   RMSE |    MAE")
print("-" * 72)
for nm, a, rp, xp in PARAMS:
    mr = met(a, rp)
    mx = met(a, xp)
    print("{:<28} | {:<16} | {:>6.4f} | {:>6.3f} | {:>6.3f}".format(
        nm, "Old Model (RSM)", mr["R2"], mr["RMSE"], mr["MAE"]))
    print("{:<28} | {:<16} | {:>6.4f} | {:>6.3f} | {:>6.3f}".format(
        "", "New Model (ML)", mx["R2"], mx["RMSE"], mx["MAE"]))
    print("-" * 72)

out_dir = "reports/comparison_plots"
os.makedirs(out_dir, exist_ok=True)
art_dir = r"C:\Users\User\.gemini\antigravity-ide\brain\e3e52262-f46f-4da9-bff3-8d496c0d4e0e"
os.makedirs(art_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# FIGURE 1 - Actual vs Predicted Values
# ══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle(
    "Figure 1: Actual vs. Predicted Values\nOld Model (RSM / GP.py)  vs.  New Unified ML Framework",
    fontsize=14, fontweight="bold", color="#f0f6fc", y=0.99
)
fig1.patch.set_facecolor("#0d1117")

for ax, (nm, a, rp, xp) in zip(axes.flat, PARAMS):
    lo = min(a.min(), rp.min(), xp.min()) * 0.94
    hi = max(a.max(), rp.max(), xp.max()) * 1.06
    ax.plot([lo, hi], [lo, hi], "--", color=DIA, lw=2, alpha=0.8, label="Ideal Fit (y = x)", zorder=1)
    ax.scatter(a, rp, color=OLD, s=65, alpha=0.85, edgecolors="white",
               linewidths=0.5, label="Old Model (RSM)", zorder=3)
    ax.scatter(a, xp, color=NEW, s=65, alpha=0.85, edgecolors="white",
               linewidths=0.5, marker="D", label="New Model (ML)", zorder=4)
    mr = met(a, rp)
    mx = met(a, xp)
    info = "Old RSM: R2={:.3f}, RMSE={:.2f}\nNew ML:  R2={:.3f}, RMSE={:.2f}".format(
        mr["R2"], mr["RMSE"], mx["R2"], mx["RMSE"])
    ax.text(0.04, 0.96, info, transform=ax.transAxes, fontsize=9,
            va="top", color="#e6edf3",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d",
                      edgecolor="#58a6ff", alpha=0.92))
    ax.set_title(nm, fontsize=12, fontweight="bold", color="#f0f6fc", pad=8)
    ax.set_xlabel("Actual Reference Value", fontsize=10)
    ax.set_ylabel("Predicted Value", fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.3)

fig1.tight_layout(pad=2.5)
f1_path = os.path.join(out_dir, "fig1_actual_vs_predicted.png")
fig1.savefig(f1_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
fig1.savefig(os.path.join(art_dir, "fig1_actual_vs_predicted.png"), dpi=200, bbox_inches="tight", facecolor="#0d1117")
plt.close(fig1)
print("Saved fig1_actual_vs_predicted.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 2 - Accuracy Metrics Bar Chart
# ══════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    "Figure 2: Accuracy Metrics Comparison (MAE, RMSE, R²)\nOld Model (RSM)  vs.  New Unified ML Framework",
    fontsize=14, fontweight="bold", color="#f0f6fc", y=1.03
)
fig2.patch.set_facecolor("#0d1117")

pnames = [p[0] for p in PARAMS]
rm_list = [met(p[1], p[2]) for p in PARAMS]
xm_list = [met(p[1], p[3]) for p in PARAMS]
xax = np.arange(len(pnames))
w = 0.36

cfgs = [
    ("MAE", "Mean Absolute Error (MAE)  [Lower is Better]", False),
    ("RMSE", "Root Mean Squared Error (RMSE)  [Lower is Better]", False),
    ("R2", "Coefficient of Determination (R²)  [Higher is Better]", True),
]

for ax2, (mkey, mlabel, hib) in zip(axes2, cfgs):
    rv = [m[mkey] for m in rm_list]
    xv = [m[mkey] for m in xm_list]
    b1 = ax2.bar(xax - w/2, rv, width=w, color=OLD, alpha=0.88,
                 label="Old Model (RSM)", zorder=3, edgecolor="white", linewidth=0.5)
    b2 = ax2.bar(xax + w/2, xv, width=w, color=NEW, alpha=0.88,
                 label="New Model (ML)", zorder=3, edgecolor="white", linewidth=0.5)
    mx_v = max(max(abs(v) for v in rv), max(abs(v) for v in xv))

    for bar, val in zip(b1, rv):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = mx_v * 0.02 if h >= 0 else -mx_v * 0.04
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset,
                 "{:.3f}".format(val), ha="center", va=va,
                 fontsize=8, color=OLD, fontweight="bold")
    for bar, val in zip(b2, xv):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = mx_v * 0.02 if h >= 0 else -mx_v * 0.04
        ax2.text(bar.get_x() + bar.get_width()/2, h + offset,
                 "{:.3f}".format(val), ha="center", va=va,
                 fontsize=8, color=NEW, fontweight="bold")

    for i, (r, x) in enumerate(zip(rv, xv)):
        wb = b2[i] if (x >= r if hib else x <= r) else b1[i]
        wb.set_edgecolor(DIA)
        wb.set_linewidth(2.2)

    ax2.set_title(mlabel, fontsize=11, fontweight="bold", color="#f0f6fc", pad=8)
    ax2.set_xticks(xax)
    ax2.set_xticklabels(pnames, rotation=15, ha="right", fontsize=9.5)
    ax2.set_ylabel(mkey, fontsize=10)
    ax2.axhline(0, color="#484f58", lw=0.8, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.set_axisbelow(True)

fig2.tight_layout(pad=2.2)
f2_path = os.path.join(out_dir, "fig2_metrics_bar_chart.png")
fig2.savefig(f2_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
fig2.savefig(os.path.join(art_dir, "fig2_metrics_bar_chart.png"), dpi=200, bbox_inches="tight", facecolor="#0d1117")
plt.close(fig2)
print("Saved fig2_metrics_bar_chart.png")

# ══════════════════════════════════════════════════════════════
# FIGURE 3 - Per-sample Absolute Prediction Errors
# ══════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 11))
fig3.suptitle(
    "Figure 3: Prediction Absolute Error per Test Sample (|Actual - Predicted|)\nOld Model (RSM)  vs.  New Unified ML Framework",
    fontsize=14, fontweight="bold", color="#f0f6fc", y=0.99
)
fig3.patch.set_facecolor("#0d1117")

for ax3, (nm, a, rp, xp) in zip(axes3.flat, PARAMS):
    n = len(a)
    idx2 = np.arange(1, n + 1)
    er = np.abs(a - rp)
    ex = np.abs(a - xp)
    w2 = 0.38
    ax3.bar(idx2 - w2/2, er, width=w2, color=OLD, alpha=0.8,
            label="Old Model Error", zorder=3, edgecolor="none")
    ax3.bar(idx2 + w2/2, ex, width=w2, color=NEW, alpha=0.8,
            label="New Model Error", zorder=3, edgecolor="none")
    ax3.axhline(er.mean(), color=OLD, lw=2, linestyle="--", alpha=0.9,
                label="Old Mean Error = {:.2f}".format(er.mean()))
    ax3.axhline(ex.mean(), color=NEW, lw=2, linestyle="--", alpha=0.9,
                label="New Mean Error = {:.2f}".format(ex.mean()))
    ax3.set_title(nm, fontsize=12, fontweight="bold", color="#f0f6fc", pad=8)
    ax3.set_xlabel("Test Case Sample Index (1 to 24)", fontsize=9.5)
    ax3.set_ylabel("|Actual - Predicted| Error", fontsize=9.5)
    ax3.set_xticks(idx2)
    ax3.tick_params(axis="x", labelsize=8)
    ax3.legend(fontsize=8, framealpha=0.88, loc="upper right")
    ax3.grid(True, axis="y", alpha=0.3)
    ax3.set_axisbelow(True)

fig3.tight_layout(pad=2.5)
f3_path = os.path.join(out_dir, "fig3_prediction_errors.png")
fig3.savefig(f3_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
fig3.savefig(os.path.join(art_dir, "fig3_prediction_errors.png"), dpi=200, bbox_inches="tight", facecolor="#0d1117")
plt.close(fig3)
print("Saved fig3_prediction_errors.png")

print("\nALL DONE! Check reports/comparison_plots/ and artifact directory")
