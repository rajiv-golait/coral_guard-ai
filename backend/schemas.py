from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

HealthClass = Literal["Healthy", "Bleached", "Dead"]
RiskLevel = Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]
AlertType = Literal["CRITICAL", "HIGH"]


class OceanParams(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, alias="Latitude_Degrees")
    longitude: float = Field(..., ge=-180, le=180, alias="Longitude_Degrees")
    depth_m: float = Field(..., ge=0, le=50, alias="Depth_m")
    turbidity: float = Field(..., ge=0, le=20, alias="Turbidity")
    cyclone_frequency: float = Field(..., ge=0, le=10, alias="Cyclone_Frequency")
    clim_sst: float = Field(..., ge=20, le=35, alias="ClimSST")
    ssta: float = Field(..., ge=-3, le=6, alias="SSTA")
    tsa: float = Field(..., ge=-2, le=5, alias="TSA")
    percent_cover: float = Field(..., ge=0, le=100, alias="Percent_Cover")
    date_year: int = Field(..., ge=1980, le=2025, alias="Date_Year")

    model_config = {"populate_by_name": True}

    def to_feature_dict(self) -> dict[str, float]:
        return {
            "Latitude_Degrees": self.latitude,
            "Longitude_Degrees": self.longitude,
            "Depth_m": self.depth_m,
            "Turbidity": self.turbidity,
            "Cyclone_Frequency": self.cyclone_frequency,
            "ClimSST": self.clim_sst,
            "SSTA": self.ssta,
            "TSA": self.tsa,
            "Percent_Cover": self.percent_cover,
            "Date_Year": float(self.date_year),
        }


class PredictionResponse(BaseModel):
    health_class: HealthClass
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    cluster_id: int
    cluster_name: str
    is_anomaly: bool
    risk_level: RiskLevel
    thermal_stress: float
    light_index: float
    sst_total: float


class ReportRequest(BaseModel):
    health_class: HealthClass
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    cluster_id: int
    cluster_name: str
    is_anomaly: bool
    risk_level: RiskLevel
    thermal_stress: float
    light_index: float
    sst_total: float
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth_m: float = Field(..., ge=0, le=50)
    turbidity: float = Field(..., ge=0, le=20)
    ssta: float = Field(..., ge=-3, le=6)
    tsa: float = Field(..., ge=-2, le=5)
    site_name: str = "Unknown Reef Site"


class ConservationReport(BaseModel):
    executive_summary: str
    risk_level: RiskLevel
    key_threats: list[str]
    immediate_actions: list[str]
    preventive_measures: list[str]
    monitoring_schedule: str
    recovery_prognosis: str
    scientific_context: str


class AlertRequest(BaseModel):
    health_class: HealthClass
    confidence: float = Field(..., ge=0, le=1)
    cluster_name: str
    is_anomaly: bool
    risk_level: RiskLevel
    site_name: str = "Unknown Reef Site"
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    depth_m: float = Field(..., ge=0, le=50)
    executive_summary: str = ""
    immediate_actions: list[str] = Field(default_factory=list)
    override_email: Optional[str] = None
    image_base64: Optional[str] = None
    image_filename: str = "coral_image.jpg"


class AlertResponse(BaseModel):
    triggered: bool
    alert_type: Optional[AlertType] = None
    email_sent: bool
    sms_sent: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
