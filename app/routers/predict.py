"""Prediction endpoint: image + tabular JSON -> CNN + DBSCAN + fusion + LLM report."""

from __future__ import annotations

import json
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.inference import get_engine
from app.schemas import (
    CNNResult,
    CoralCsvFeatures,
    DBSCANResult,
    FusionResult,
    PredictResponse,
)
from app.utils import ROOT_DIR, ensure_dir, get_logger, safe_unlink

log = get_logger()

router = APIRouter(tags=["predict"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
_EXT_OK = {".jpg", ".jpeg", ".png", ".webp"}

# Swagger UI defaults the `features` text box to the literal word "string" — that is not valid JSON.
FEATURES_JSON_EXAMPLE = (
    '{"latitude_degrees":23.163,"longitude_degrees":-82.526,"depth_m":10,'
    '"turbidity":0.0287,"cyclone_frequency":49.9,"clim_sst":301.61,"ssta":-0.46,'
    '"tsa":-0.8,"percent_cover":0,"date_year":2005}'
)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Tri-modal inference + LLM report",
)
async def predict(
    image: UploadFile = File(..., description="Reef image (JPEG/PNG/WebP)"),
    features: str = Form(
        ...,
        description=(
            "Single JSON object with all CoralCsvFeatures fields (Phase 3 / coral.csv). "
            "Do not leave the default word 'string' — replace with real JSON. Use the example below."
        ),
        openapi_examples={
            "sample_row": {
                "summary": "Valid example (edit values as needed)",
                "description": "Copy this shape; Swagger's default 'string' will always fail.",
                "value": FEATURES_JSON_EXAMPLE,
            }
        },
    ),
) -> PredictResponse:
    upload_dir = ROOT_DIR / "uploads"
    ensure_dir(upload_dir)
    tmp_path = upload_dir / f"{uuid.uuid4().hex}_upload"

    errors: list[str] = []
    cnn_out: CNNResult | None = None
    db_out: DBSCANResult | None = None
    fusion_out: FusionResult | None = None
    llm_text: str | None = None

    try:
        ctype = (image.content_type or "").lower()
        fname = (image.filename or "").lower()
        ext_ok = any(fname.endswith(e) for e in _EXT_OK)
        if ctype not in ALLOWED_IMAGE_TYPES and not (ctype in {"", "application/octet-stream"} and ext_ok):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type {image.content_type!r}. Use JPEG, PNG, or WebP.",
            )

        raw = await image.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="Image exceeds 15 MB limit.")

        feat_raw = (features or "").strip()
        if not feat_raw or feat_raw.lower() == "string":
            raise HTTPException(
                status_code=422,
                detail=(
                    "Field `features` must be valid JSON (object), not the Swagger placeholder 'string'. "
                    f"Example: {FEATURES_JSON_EXAMPLE}"
                ),
            )

        try:
            feats = CoralCsvFeatures.model_validate_json(feat_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid JSON in features: {exc}. "
                    f"Paste one object like: {FEATURES_JSON_EXAMPLE}"
                ),
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Feature validation failed: {exc}") from exc

        tmp_path.write_bytes(raw)

        engine = get_engine()

        try:
            cnn_out = engine.predict_cnn(tmp_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("CNN inference failed")
            errors.append(f"cnn: {exc}")

        try:
            db_out = engine.predict_dbscan(feats)
        except Exception as exc:  # noqa: BLE001
            log.exception("DBSCAN inference failed")
            errors.append(f"dbscan: {exc}")

        cluster_id = db_out.cluster_id if db_out else -1
        try:
            fusion_out = engine.predict_fusion(feats, cluster_id=cluster_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Fusion inference failed")
            errors.append(f"fusion: {exc}")

        try:
            llm_text = engine.generate_llm_report(cnn_out, db_out, fusion_out, feats)
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM report generation failed")
            errors.append(f"llm: {exc}")
            llm_text = f"Report generation failed: {exc}"

        any_model = cnn_out is not None or db_out is not None or fusion_out is not None
        status_val: Literal["ok", "error"] = "ok" if any_model else "error"
        return PredictResponse(
            status=status_val,
            cnn=cnn_out,
            dbscan=db_out,
            fusion=fusion_out,
            llm_report=llm_text,
            meta={
                "image_filename": image.filename,
                "content_type": image.content_type,
                "cluster_id_used_for_fusion": cluster_id,
            },
            errors=errors,
        )
    finally:
        safe_unlink(tmp_path)
