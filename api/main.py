"""
FastAPI Backend Application for Prestressed Concrete I-Girder ML Optimization Framework.
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import build_features
from src.postprocess.constraints import enforce_constraints
from src.data.load_data import TARGET_COLS

app = FastAPI(
    title="Prestressed Concrete I-Girder ML Optimization API",
    description="High-precision ML surrogate inference backend for bridge I-girder design parameter prediction.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static UI Files
if os.path.exists("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

@app.get("/")
def root():
    return RedirectResponse(url="/ui/index.html")


# Global artifacts holders
SCALER = None
MODEL = None

def get_artifacts():
    global SCALER, MODEL
    scaler_path = "models/scaler.pkl"
    model_path = "models/best_model.pkl"

    if SCALER is None and os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
    if MODEL is None and os.path.exists(model_path):
        MODEL = joblib.load(model_path)

    return SCALER, MODEL

class PredictRequest(BaseModel):
    Concrete: float = Field(..., ge=405.0, le=600.0, description="Concrete Unit Cost ($/yd3)")
    Strand: float = Field(..., ge=1.26, le=2.23, description="Prestressing Strand Unit Cost ($/linear ft per strand)")
    Rebar: float = Field(..., ge=2.18, le=3.45, description="Steel Rebar Unit Cost ($/lb)")
    Span_ft: float = Field(..., ge=100.0, le=180.0, description="Span Length (ft)")

    class Config:
        json_schema_extra = {
            "example": {
                "Concrete": 505.0,
                "Strand": 1.73,
                "Rebar": 2.18,
                "Span_ft": 140.0
            }
        }

class PredictResponse(BaseModel):
    Gir_Dep_in: float = Field(..., description="Girder Depth (inches)")
    Lat_Spac_ft: float = Field(..., description="Lateral Spacing Between Girders (feet)")
    No_of_Gir: int = Field(..., description="Number of Girders (count)")
    bot_flange_depth_in: float = Field(..., description="Bottom Flange Bottom Depth (inches)")
    bot_flange_width_in: float = Field(..., description="Bottom Flange Bottom Width (inches)")
    Number_of_strands: int = Field(..., description="Number of Prestressing Strands per Girder (even count)")
    Harp_Pos_ft: float = Field(..., description="Harping Position (feet)")
    raw_predictions: Optional[Dict[str, float]] = None

@app.get("/health")
def health_check():
    scaler, model = get_artifacts()
    return {
        "status": "healthy",
        "scaler_loaded": scaler is not None,
        "model_loaded": model is not None
    }

@app.get("/equations")
def get_rsm_equations():
    eq_path = "reports/equations/rsm_equations.json"
    if not os.path.exists(eq_path):
        raise HTTPException(status_code=404, detail="RSM equations report not found.")
    with open(eq_path, "r") as f:
        data = json.load(f)
    return data

@app.post("/predict", response_model=PredictResponse)
def predict_girder_design(req: PredictRequest):
    scaler, model = get_artifacts()
    if scaler is None or model is None:
        raise HTTPException(
            status_code=500,
            detail="ML artifacts (scaler/best_model.pkl) are missing. Please train the model first."
        )

    try:
        raw_df = pd.DataFrame([{
            "Concrete": req.Concrete,
            "Strand": req.Strand,
            "Rebar": req.Rebar,
            "Span_ft": req.Span_ft
        }])

        # 1. Feature Engineering
        X_feat = build_features(raw_df)

        # 2. Scaling
        X_scaled = scaler.transform(X_feat)

        # 3. Model Inference
        raw_preds_arr = model.predict(X_scaled)[0]
        raw_pred_dict = {col: float(val) for col, val in zip(TARGET_COLS, raw_preds_arr)}

        # 4. Enforce Physical & Code Constraints
        enforced = enforce_constraints(raw_pred_dict, req.Span_ft)

        return PredictResponse(
            Gir_Dep_in=float(enforced["Gir Dep (in)"]),
            Lat_Spac_ft=float(enforced["Lat Spac (ft)"]),
            No_of_Gir=int(enforced["No. of Gir"]),
            bot_flange_depth_in=float(enforced["bot flange bot part depth (in)"]),
            bot_flange_width_in=float(enforced["bot flange bot part width (in)"]),
            Number_of_strands=int(enforced["Number of strand per girder"]),
            Harp_Pos_ft=float(enforced["Harp Pos (ft)"]),
            raw_predictions=raw_pred_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
