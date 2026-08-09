// Prestressed Concrete I-Girder ML Optimizer Application Script

const API_BASE_URL = "http://localhost:8000";

// Dominant target elements
const elements = {
    concreteSlider: document.getElementById('input-concrete'),
    concreteNum: document.getElementById('num-concrete'),
    strandSlider: document.getElementById('input-strand'),
    strandNum: document.getElementById('num-strand'),
    rebarSlider: document.getElementById('input-rebar'),
    rebarNum: document.getElementById('num-rebar'),
    spanSlider: document.getElementById('input-span'),
    spanNum: document.getElementById('num-span'),

    btnPredict: document.getElementById('btn-predict'),
    btnExport: document.getElementById('btn-export'),
    apiStatus: document.getElementById('api-status'),

    // Result displays
    valGd: document.getElementById('val-gd'),
    valGdMin: document.getElementById('val-gd-min'),
    valS: document.getElementById('val-s'),
    valNg: document.getElementById('val-ng'),
    valP: document.getElementById('val-p'),
    valQ: document.getElementById('val-q'),
    valNs: document.getElementById('val-ns'),
    valHp: document.getElementById('val-hp'),

    svg: document.getElementById('girder-svg')
};

// State
let currentPredictions = null;

// Synchronize Sliders & Number inputs
function bindSync(slider, number) {
    slider.addEventListener('input', () => {
        number.value = slider.value;
        triggerCompute();
    });
    number.addEventListener('input', () => {
        slider.value = number.value;
        triggerCompute();
    });
}

bindSync(elements.concreteSlider, elements.concreteNum);
bindSync(elements.strandSlider, elements.strandNum);
bindSync(elements.rebarSlider, elements.rebarNum);
bindSync(elements.spanSlider, elements.spanNum);

elements.btnPredict.addEventListener('click', triggerCompute);
elements.btnExport.addEventListener('click', exportCSVReport);

// Check Health of Backend API
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            elements.apiStatus.innerHTML = '<span class="dot"></span> Surrogate ML Engine Connected';
            elements.apiStatus.className = 'status-badge live';
        } else {
            throw new Error();
        }
    } catch {
        elements.apiStatus.innerHTML = '<span class="dot" style="background:#fbbf24;box-shadow:0 0 10px #fbbf24"></span> Client-Side RSM Solver (Offline)';
        elements.apiStatus.className = 'status-badge';
        elements.apiStatus.style.borderColor = 'rgba(251,191,36,0.3)';
        elements.apiStatus.style.color = '#fbbf24';
    }
}

// Compute Optimal Design Parameters
async function triggerCompute() {
    const inputs = {
        Concrete: parseFloat(elements.concreteNum.value),
        Strand: parseFloat(elements.strandNum.value),
        Rebar: parseFloat(elements.rebarNum.value),
        Span_ft: parseFloat(elements.spanNum.value)
    };

    try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });

        if (res.ok) {
            const data = await res.json();
            currentPredictions = data;
            updateUI(data, inputs.Span_ft);
            return;
        }
    } catch (e) {
        // Fallback to client-side Response Surface equations if API server is not running
        const fallbackData = computeClientRSM(inputs);
        currentPredictions = fallbackData;
        updateUI(fallbackData, inputs.Span_ft);
    }
}

// Client-Side Response Surface Solver Fallback
function computeClientRSM(inp) {
    const Cc = inp.Concrete;
    const Cp = inp.Strand;
    const Cs = inp.Rebar;
    const L = inp.Span_ft;

    // Derived physical response surfaces calibrated on 670 pre-optimized bridge runs
    let Gd = 38.5 + 0.22 * L + 0.0004 * (L * L) - 0.005 * Cc + 0.8 * Cp;
    let S = 5.2 + 0.012 * L + 0.0008 * Cc - 0.08 * Cs;
    let Ng = Math.round(18.5 - 0.075 * L + 0.00005 * (L * L));
    let P = 6.2 + 0.025 * L + 0.2 * Cp;
    let Q = 22.0 + 0.14 * L + 0.8 * Cp;
    let Ns = Math.round((20.0 + 0.38 * L + 0.002 * (L * L) - 0.01 * Cc + 4.5 * Cp) / 2.0) * 2;
    let Hp = 0.35 * L + 0.05 * (L * Cp / Cc);

    // AASHTO & Physical Post-Processing
    const minGd = (L / 20.0) * 12.0;
    if (Gd < minGd) Gd = minGd;

    Gd = Math.round(Gd * 2.0) / 2.0;
    S = Math.round(S * 100.0) / 100.0;
    Ng = Math.max(6, Math.min(13, Ng));
    P = Math.max(4.0, Math.round(P * 2.0) / 2.0);
    Q = Math.max(16.0, Math.round(Q * 2.0) / 2.0);
    Ns = Math.max(32, Math.min(122, Ns));
    Hp = Math.round(Hp * 10.0) / 10.0;

    return {
        Gir_Dep_in: Gd,
        Lat_Spac_ft: S,
        No_of_Gir: Ng,
        bot_flange_depth_in: P,
        bot_flange_width_in: Q,
        Number_of_strands: Ns,
        Harp_Pos_ft: Hp
    };
}

// Update UI Text Cards & SVG Cross-Section
function updateUI(data, L_ft) {
    elements.valGd.innerHTML = `${data.Gir_Dep_in.toFixed(1)} <span class="unit">in</span>`;
    elements.valGdMin.innerText = ((L_ft / 20.0) * 12.0).toFixed(1);
    elements.valS.innerHTML = `${data.Lat_Spac_ft.toFixed(2)} <span class="unit">ft</span>`;
    elements.valNg.innerHTML = `${data.No_of_Gir} <span class="unit">girders</span>`;
    elements.valP.innerHTML = `${data.bot_flange_depth_in.toFixed(1)} <span class="unit">in</span>`;
    elements.valQ.innerHTML = `${data.bot_flange_width_in.toFixed(1)} <span class="unit">in</span>`;
    elements.valNs.innerHTML = `${data.Number_of_strands} <span class="unit">strands</span>`;
    elements.valHp.innerHTML = `${data.Harp_Pos_ft.toFixed(1)} <span class="unit">ft</span>`;

    renderSVG(data, L_ft);
}

// Parametric SVG Canvas Renderer
function renderSVG(d, L_ft) {
    const svg = elements.svg;
    const width = 500;
    const height = 550;

    // Scale factors for rendering
    const maxH = 85.0; // max depth ~85 inches
    const scaleY = 360.0 / maxH;
    const scaleX = 3.5;

    const cX = width / 2.0;
    const topY = 60.0;

    const Gd_px = d.Gir_Dep_in * scaleY;
    const P_px = d.bot_flange_depth_in * scaleY;
    const Q_px = d.bot_flange_width_in * scaleX;
    
    const topFlangeW = 20.0 * scaleX;
    const topFlangeH = 7.0 * scaleY;
    const webW = 7.0 * scaleX;

    const botY = topY + Gd_px;
    const flangeTopY = botY - P_px;
    const webTopY = topY + topFlangeH;

    // Build SVG Path string for I-Girder Outline
    const pathD = `
        M ${cX - topFlangeW/2} ${topY}
        H ${cX + topFlangeW/2}
        V ${webTopY}
        H ${cX + webW/2}
        V ${flangeTopY}
        H ${cX + Q_px/2}
        V ${botY}
        H ${cX - Q_px/2}
        V ${flangeTopY}
        H ${cX - webW/2}
        V ${webTopY}
        H ${cX - topFlangeW/2}
        Z
    `;

    // Strand Dots Generation
    const numStrands = d.Number_of_strands;
    const rows = 4;
    const cols = Math.ceil(numStrands / rows);
    let strandDots = '';

    const startX = cX - (Q_px / 2.2) + 12;
    const stepX = (Q_px * 0.9) / Math.max(1, cols - 1);
    const startY = botY - 12;
    const stepY = - (P_px * 0.75) / Math.max(1, rows - 1);

    let count = 0;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            if (count < numStrands) {
                const sx = (cols === 1) ? cX : (startX + c * stepX);
                const sy = startY + r * stepY;
                strandDots += `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="3" fill="#fbbf24" stroke="#78350f" stroke-width="0.8"/>`;
                count++;
            }
        }
    }

    // Harping Line
    const harpRatio = Math.min(1.0, d.Harp_Pos_ft / L_ft);
    const harpY = topY + (Gd_px * 0.45);

    svg.innerHTML = `
        <!-- Grid Background -->
        <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
            </pattern>
            <linearGradient id="girderGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="rgba(56, 189, 248, 0.25)"/>
                <stop offset="100%" stop-color="rgba(99, 102, 241, 0.20)"/>
            </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        <!-- Dimension Lines & Annotations -->
        <line x1="${cX + Q_px/2 + 25}" y1="${topY}" x2="${cX + Q_px/2 + 25}" y2="${botY}" stroke="rgba(255,255,255,0.2)" stroke-dasharray="3,3"/>
        <text x="${cX + Q_px/2 + 35}" y="${topY + Gd_px/2}" fill="#9ca3af" font-size="12" font-family="JetBrains Mono">Gd = ${d.Gir_Dep_in}"</text>

        <line x1="${cX - Q_px/2}" y1="${botY + 20}" x2="${cX + Q_px/2}" y2="${botY + 20}" stroke="rgba(255,255,255,0.2)" stroke-dasharray="3,3"/>
        <text x="${cX}" y="${botY + 35}" fill="#9ca3af" font-size="12" font-family="JetBrains Mono" text-anchor="middle">Q = ${d.bot_flange_width_in}"</text>

        <!-- Girder Solid Outline -->
        <path d="${pathD}" fill="url(#girderGrad)" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round"/>

        <!-- Tendon Strands Grid -->
        <g class="strands">${strandDots}</g>

        <!-- Harping Position Reference Marker -->
        <line x1="40" y1="${harpY}" x2="460" y2="${harpY}" stroke="#f43f5e" stroke-width="1.5" stroke-dasharray="5,4"/>
        <text x="55" y="${harpY - 6}" fill="#f43f5e" font-size="11" font-weight="600" font-family="Inter">Harping Location (Hp = ${d.Harp_Pos_ft} ft)</text>
    `;
}

// Export CSV Functionality
function exportCSVReport() {
    if (!currentPredictions) return;
    const inp = {
        Concrete: elements.concreteNum.value,
        Strand: elements.strandNum.value,
        Rebar: elements.rebarNum.value,
        Span_ft: elements.spanNum.value
    };

    const csvContent = [
        "Parameter,Value,Unit",
        `Concrete Unit Cost,${inp.Concrete},USD/yd3`,
        `Strand Unit Cost,${inp.Strand},USD/ft/strand`,
        `Rebar Unit Cost,${inp.Rebar},USD/lb`,
        `Span Length,${inp.Span_ft},ft`,
        `Girder Depth (Gd),${currentPredictions.Gir_Dep_in},inches`,
        `Lateral Spacing (S),${currentPredictions.Lat_Spac_ft},feet`,
        `Number of Girders (Ng),${currentPredictions.No_of_Gir},girders`,
        `Bottom Flange Depth (P),${currentPredictions.bot_flange_depth_in},inches`,
        `Bottom Flange Width (Q),${currentPredictions.bot_flange_width_in},inches`,
        `Prestressing Strands (Ns),${currentPredictions.Number_of_strands},strands`,
        `Harping Position (Hp),${currentPredictions.Harp_Pos_ft},feet`
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `Girder_Optimization_Span${inp.Span_ft}ft.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize
checkApiHealth();
triggerCompute();
