"""
CoralGuard AI — FastAPI entrypoint.

Run from project root (`coralguard-api/`):

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.inference import get_engine, get_settings
from app.routers import predict
from app.schemas import HealthResponse
from app.utils import ROOT_DIR, get_logger, setup_logging

setup_logging()
log = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Warm up model weights at startup so the first /predict is fast."""
    log.info("CoralGuard AI starting — loading models…")
    try:
        get_engine()
        log.info("Models ready.")
    except Exception:  # noqa: BLE001
        log.exception("Model warmup failed — some routes may error until files are fixed")
    yield
    log.info("CoralGuard AI shutdown.")


app = FastAPI(
    title="CoralGuard AI",
    description=(
        "Production API for tri-modal marine ecosystem anomaly detection: "
        "EfficientNet-B3 coral imagery, DBSCAN oceanographic regimes, "
        "and a fusion ANN for bleaching extent (Percent_Bleaching), "
        "plus an optional Groq Llama 3.3 marine assessment report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = ROOT_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    log.warning("Static directory missing at %s — thesis figures will not be served", static_dir)

frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.is_dir():

    @app.get("/ui", include_in_schema=False)
    async def _ui_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
else:
    log.warning("Frontend missing at %s — open README for manual static hosting", frontend_dir)

app.include_router(predict.router)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    s = get_settings()
    return HealthResponse(
        service="CoralGuard AI",
        version="1.0.0",
        status="healthy",
        models={
            "cnn": str(s.cnn_path),
            "dbscan": str(s.dbscan_path),
            "fusion": str(s.fusion_path),
        },
        docs="/docs",
        static_figures="/static/thesis/figures/",
        tabular_dir=str(s.tabular_dir),
        frontend="/ui/",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
