"""Pydantic request/response models for CoralGuard AI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CoralCsvFeatures(BaseModel):
    """
    Base environmental fields from `coral.csv` Phase 3 (`mdm.py`), before RobustScaler.

    Engineered columns (`Thermal_Stress`, `Light_Index`, `SST_Total`) are derived in the
    API exactly as in the notebook; the MI-selected subset and order come from
    `features.pkl` in `CORALGUARD_TABULAR_DIR`.
    """

    latitude_degrees: float = Field(..., description="Latitude_Degrees")
    longitude_degrees: float = Field(..., description="Longitude_Degrees")
    depth_m: float = Field(..., ge=0, description="Depth_m")
    turbidity: float = Field(..., ge=0, description="Turbidity")
    cyclone_frequency: float = Field(..., ge=0, description="Cyclone_Frequency")
    clim_sst: float = Field(..., description="ClimSST")
    ssta: float = Field(..., description="SSTA")
    tsa: float = Field(..., description="TSA")
    percent_cover: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Percent_Cover (0–100)",
    )
    date_year: int = Field(..., ge=1900, le=2100, description="Date_Year")

    model_config = {"extra": "forbid"}


class CNNResult(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float


class DBSCANResult(BaseModel):
    cluster_id: int
    cluster_label: str
    detail: str | None = None


class FusionResult(BaseModel):
    predicted_percent_bleaching: float
    raw_model_output: float | None = None


class PredictResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"
    cnn: CNNResult | None = None
    dbscan: DBSCANResult | None = None
    fusion: FusionResult | None = None
    llm_report: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    service: str
    version: str
    status: str
    models: dict[str, str]
    docs: str
    static_figures: str
    tabular_dir: str = ""
    frontend: str = "/ui/"
