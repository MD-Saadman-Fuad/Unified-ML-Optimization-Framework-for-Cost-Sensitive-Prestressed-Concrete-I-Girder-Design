import tkinter as tk
from tkinter import messagebox
import math

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────

# Model 1 — Girder Depth D (in)
GD_COEF = {
    "intercept": 136.601234, "L": -0.357148, "C": -0.019188, "G": -10.393466,
    "L2": 0.000015, "LC": 0.000063, "LG": 0.056663, "C2": 0.000016, "CG": -0.000976, "G2": 0.041604,
}
GD_ACCURACY = 96.41

# Model 2 — No. of Girders G
NG_COEF = {
    "intercept": 67.451976, "D": -4.230516, "L": 0.538016, "L2": -0.013329,
    "LD": 0.034739, "D2": 0.039728, "L3": 0.000029, "L2D": 0.000034, "LD2": -0.000327, "D3": -0.000047,
}
NG_ACCURACY = 90.41

# Model 3 — Strands per Girder S
NS_COEF = {
    "intercept": 10.838986, "L": 0.005932, "C": -0.070040, "G": 6.556031, "D": 1.043756,
    "L2": 0.005273, "LC": 0.000049, "LG": 0.010738, "LD": -0.008008, "C2": 0.000073,
    "CG": -0.001015, "CD": 0.000042, "G2": -0.382649, "GD": -0.119622, "D2": -0.001395,
}
NS_ACCURACY = 95.78

# Model 4 — Harping Position HP (ft)
HP_ACCURACY = 83.22

# ─────────────────────────────────────────────────────────────────────────────
# CALCULATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_girder_depth(L, C, G):
    b = GD_COEF
    return (b["intercept"] + b["L"]*L + b["C"]*C + b["G"]*G + b["L2"]*L**2 + b["LC"]*L*C + b["LG"]*L*G + b["C2"]*C**2 + b["CG"]*C*G + b["G2"]*G**2)

def calculate_num_girders(L, D):
    b = NG_COEF
    val = (b["intercept"] + b["D"]*D + b["L"]*L + b["L2"]*L**2 + b["LD"]*L*D + b["D2"]*D**2 + b["L3"]*L**3 + b["L2D"]*L**2*D + b["LD2"]*L*D**2 + b["D3"]*D**3)
    return round(val)

def calculate_num_strands(L, C, G, D):
    b = NS_COEF
    return (b["intercept"] + b["L"]*L + b["C"]*C + b["G"]*G + b["D"]*D + b["L2"]*L**2 + b["LC"]*L*C + b["LG"]*L*G + b["LD"]*L*D + b["C2"]*C**2 + b["CG"]*C*G + b["CD"]*C*D + b["G2"]*G**2 + b["GD"]*G*D + b["D2"]*D**2)

def calculate_harping_position(S, G, N, P):
    """S = Strand Cost ($/linear ft/strand), G = Girder Depth, N = No. Girders, P = Strands per Girder"""
    return (32.129720 - 28.424216 * S + 0.428984 * G + 0.618971 * N + 0.263758 * P + 1.827877 * (S**2) + 0.332784 * (S * G) + 0.768138 * (S * N) - 0.091321 * (S * P) - 0.010078 * (G**2) + 0.064587 * (G * N) - 0.002618 * (G * P) - 0.462759 * (N**2) + 0.046292 * (N * P) - 0.000003 * (P**2))

def iterative_solve(L, C, max_iter=20):
    G = max(6, round(L / 20))
    for _ in range(max_iter):
        D = calculate_girder_depth(L, C, G)
        G_new = calculate_num_girders(L, D)
        if G_new == G: break
        G = G_new
    D = calculate_girder_depth(L, C, G)
    S = calculate_num_strands(L, C, G, D)
    return D, G, S;

# ─────────────────────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class BridgeApp(tk.Tk):
    DARK_BG, CARD_BG, HDR_BG = "#1a1f2e", "#252b3b", "#0f1320"
    ACCENT, ACCENT2, ACCENT3, ACCENT4 = "#4f8ef7", "#34d399", "#f59e0b", "#c084fc"
    TEXT_PRI, TEXT_SEC, BORDER, DANGER, SUCCESS = "#f0f4ff", "#8b93a8", "#323a50", "#ef4444", "#10b981"

    def __init__(self):
        super().__init__()
        self.title("Optimized Bridge Girder Calculator")
        self.geometry("1000x820")
        self.configure(bg=self.DARK_BG)
        self._build_ui()

    def _card(self, parent, title, accent, builder_fn):
        outer = tk.Frame(parent, bg=self.BORDER)
        outer.pack(fill="x", pady=(0, 12))
        inner = tk.Frame(outer, bg=self.CARD_BG, padx=18, pady=14)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(inner, text=title, bg=self.CARD_BG, fg=accent, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        builder_fn(inner)

    def _sep(self, parent): tk.Frame(parent, bg=self.BORDER, height=1).pack(fill="x", pady=6)

    def _build_ui(self):
        hdr = tk.Frame(self, bg=self.HDR_BG, pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🏗   Optimized Bridge Girder Calculator", bg=self.HDR_BG, fg=self.TEXT_PRI, font=("Segoe UI", 18, "bold")).pack()
        
        main = tk.Frame(self, bg=self.DARK_BG)
        main.pack(fill="both", expand=True, padx=24, pady=16)

        left, right = tk.Frame(main, bg=self.DARK_BG), tk.Frame(main, bg=self.DARK_BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self._card(left, "📐  Input Parameters", self.ACCENT, self._build_inputs)
        self._card(left, "📋  Regression Formulas", self.TEXT_SEC, self._build_formulas)
        self._card(right, "📊  Calculation Results", self.ACCENT, self._build_results)
        self._card(right, "✅  Model Accuracy", self.ACCENT2, self._build_accuracy)

        btn_frame = tk.Frame(self, bg=self.DARK_BG)
        btn_frame.pack(fill="x", padx=24, pady=(0, 16))
        tk.Button(btn_frame, text="▶   CALCULATE", command=self._calculate, bg=self.ACCENT, fg="white", font=("Segoe UI", 13, "bold"), relief="flat", bd=0, padx=32, pady=10).pack(side="left", padx=(0, 8))
        tk.Button(btn_frame, text="✕  Clear", command=self._clear, bg=self.CARD_BG, fg=self.TEXT_SEC, font=("Segoe UI", 11), relief="flat", bd=0, padx=20, pady=10).pack(side="left")
        self.status_lbl = tk.Label(btn_frame, text="", bg=self.DARK_BG, fg=self.TEXT_SEC, font=("Segoe UI", 10))
        self.status_lbl.pack(side="right", padx=8)

    def _build_inputs(self, parent):
        self._span_var = tk.StringVar(value="e.g. 140")
        self._cost_var = tk.StringVar(value="e.g. 505")
        self._strand_cost_var = tk.StringVar(value="e.g. 1.26")

        fields = [("Span Length (ft)", self._span_var, "100 – 180"), ("Concrete Cost ($/yd³)", self._cost_var, "405 – 600"), ("Strand Cost ($/linear ft/strand)", self._strand_cost_var, "1.26 - 2.23")]
        for label, var, hint in fields:
            row = tk.Frame(parent, bg=self.CARD_BG)
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=self.CARD_BG, fg=self.TEXT_PRI, font=("Segoe UI", 10), width=24, anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, bg="#1a1f2e", fg=self.TEXT_PRI, insertbackground=self.TEXT_PRI, font=("Segoe UI", 11), relief="flat", bd=0, highlightthickness=1, highlightbackground=self.BORDER, highlightcolor=self.ACCENT, width=14)
            ent.pack(side="left", padx=8)
            ent.bind("<FocusIn>", lambda e, v=var: self._focus_in(e, v))
            ent.bind("<FocusOut>", lambda e, v=var: self._focus_out(e, v))
            tk.Label(row, text=hint, bg=self.CARD_BG, fg=self.TEXT_SEC, font=("Segoe UI", 9)).pack(side="left", padx=4)

    def _focus_in(self, event, var):
        if var.get().startswith("e.g."): event.widget.delete(0, "end"); event.widget.config(fg=self.TEXT_PRI)
    def _focus_out(self, event, var):
        if not var.get(): event.widget.insert(0, "e.g. ...")

    def _build_formulas(self, parent):
        formulas = [
            ("Model 1 — Girder Depth D (in)", "D = β₀ + β₁·L + β₂·C + β₃·G + β₄·L² + β₅·L·C + β₆·L·G + β₇·C² + β₈·C·G + β₉·G²"),
            ("Model 2 — No. of Girders G", "G = round( β₀ + β₁·D + β₂·L + β₃·L² + β₄·L·D + β₅·D² + β₆·L³ + β₇·L²·D + β₈·L·D² + β₉·D³ )"),
            ("Model 3 — Strands per Girder S", "S = β₀ + β₁·L + β₂·C + β₃·G + β₄·D + β₅·L² + β₆·L·C + β₇·L·G + β₈·L·D + β₉·C² + β₁₀·C·G + β₁₁·C·D + β₁₂·G² + β₁₃·G·D + β₁₄·D²"),
            ("Model 4 — Harping Position HP (ft)", "HP =β₀ + β₁·S + β₂·D + β₃·G + β₄·P + β₅·S2 + β₆·D2 + β₇·G2 + β₈·P2 + β₉·(S*D) + β₁₀·(S*G) + β₁₁·(S*P) + β₁₂·(D*G) + β₁₃·(D*P) + β₁₄·(G*P)")
        ]
        for title, fml in formulas:
            f = tk.Frame(parent, bg="#1a1f2e", highlightbackground=self.TEXT_SEC, highlightthickness=1, padx=10, pady=6)
            f.pack(fill="x", pady=4)
            tk.Label(f, text=title, bg="#1a1f2e", fg=self.TEXT_PRI, font=("Segoe UI", 9, "bold")).pack(anchor="w")
            tk.Label(f, text=fml, bg="#1a1f2e", fg=self.TEXT_SEC, font=("Courier New", 8)).pack(anchor="w")

    def _build_results(self, parent):
        self._result_labels = {}
        for key, label in [("gd", "Girder Depth"), ("ng", "No. of Girders"), ("ns", "No. of Strands"), ("hp", "Harping Position")]:
            row = tk.Frame(parent, bg=self.CARD_BG, pady=5)
            row.pack(fill="x")
            tk.Label(row, text=label, bg=self.CARD_BG, fg=self.TEXT_SEC, width=20, anchor="w").pack(side="left")
            self._result_labels[key] = tk.Label(row, text="—", bg=self.CARD_BG, fg=self.ACCENT, font=("Segoe UI", 16, "bold"))
            self._result_labels[key].pack(side="left")

    def _build_accuracy(self, parent):
        for name, acc in [("Girder Depth", GD_ACCURACY), ("No. of Girders", NG_ACCURACY), ("No. of Strands", NS_ACCURACY), ("Harping Position", HP_ACCURACY)]:
            row = tk.Frame(parent, bg=self.CARD_BG, pady=5)
            row.pack(fill="x")
            tk.Label(row, text=name, bg=self.CARD_BG, fg=self.TEXT_PRI, width=20, anchor="w").pack(side="left")
            tk.Label(row, text=f"{acc}%", bg=self.CARD_BG, fg=self.ACCENT2).pack(side="left")

    def _calculate(self):
        try:
            L, C, S_cost = float(self._span_var.get()), float(self._cost_var.get()), float(self._strand_cost_var.get())
            D, G, S = iterative_solve(L, C)
            HP = calculate_harping_position(S=S_cost, G=D, N=G, P=round(S))
            self._result_labels["gd"].config(text=f"{D:.2f} in")
            self._result_labels["ng"].config(text=str(G))
            self._result_labels["ns"].config(text=str(round(S)))
            self._result_labels["hp"].config(text=f"{HP:.2f} ft")
            self.status_lbl.config(text="✓ Calculation Success", fg=self.SUCCESS)
        except: self.status_lbl.config(text="⚠ Invalid Input", fg=self.DANGER)

    def _clear(self):
        self._span_var.set("e.g. 140"); self._cost_var.set("e.g. 505"); self._strand_cost_var.set("e.g. 1.26")
        for lbl in self._result_labels.values(): lbl.config(text="—")

if __name__ == "__main__":
    app = BridgeApp()
    app.mainloop()