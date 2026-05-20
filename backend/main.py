import base64
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from alert_agent import process_alert
from groq_agent import generate_conservation_report
from predict import load_models, models_loaded, predict
from schemas import (
    AlertRequest,
    AlertResponse,
    ConservationReport,
    HealthResponse,
    OceanParams,
    PredictionResponse,
    ReportRequest,
)

load_dotenv()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}
_startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_error
    print("[main] CoralGuard AI starting up — loading ML models...")
    try:
        load_models()
        _startup_error = None
        print("[main] Startup complete — models ready")
    except Exception as exc:
        _startup_error = str(exc)
        print(f"[main] WARNING: Model load failed at startup: {exc}")
    yield
    print("[main] CoralGuard AI shutting down")


app = FastAPI(
    title="CoralGuard AI",
    description="Marine ecosystem monitoring API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    loaded = models_loaded()
    body: dict[str, str | bool] = {"status": "ok", "models_loaded": loaded}
    if not loaded and _startup_error:
        body["load_error"] = _startup_error
    return body


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(
    image: UploadFile = File(...),
    params: str = Form(...),
) -> PredictionResponse:
    if not models_loaded():
        detail = (
            "Models not loaded. Place model files in backend/models/ and restart."
        )
        if _startup_error:
            detail = f"{detail} Startup error: {_startup_error}"
        raise HTTPException(status_code=503, detail=detail)

    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image type. Upload JPG or PNG only.",
        )

    try:
        params_dict = json.loads(params)
        ocean = OceanParams.model_validate(params_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in params field")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        result = predict(image_bytes, ocean.to_feature_dict())
        return PredictionResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"[main] Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/api/report", response_model=ConservationReport)
async def api_report(request: ReportRequest) -> ConservationReport:
    try:
        return generate_conservation_report(request)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        print(f"[main] Report generation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")


@app.post("/api/alert", response_model=AlertResponse)
async def api_alert(request: AlertRequest) -> AlertResponse:
    try:
        return process_alert(request)
    except Exception as exc:
        print(f"[main] Alert processing error: {exc}")
        raise HTTPException(status_code=500, detail=f"Alert processing failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
