from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


RiskLevel = Literal["Low", "Medium", "High"]


class TimelineSegment(BaseModel):
    start: float
    end: float
    label: str
    score: float


class DetectedEvent(BaseModel):
    timestamp: float
    title: str
    detail: str
    score: float



class DomainInsight(BaseModel):
    domain: str
    title: str
    observations: list[str]
    interpretation: list[str]
    microbase_angle: list[str]
    recommended_next_steps: list[str]
    caveat: str


class AnalysisResult(BaseModel):
    analysis_id: str
    video_filename: str
    duration_sec: float
    sampled_frames: int
    model_backend: str
    model_loaded: bool
    risk_level: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    current_state: list[str]
    predicted_near_future_change: list[str]
    reason: str
    timeline: list[TimelineSegment]
    detected_state_change: list[DetectedEvent]
    diagnostics: dict[str, Any]
    domain_insight: Optional[DomainInsight] = None


class HealthResponse(BaseModel):
    ok: bool
    model_backend: str
    model_loaded: bool
    device: str
    model_dir: str
    allow_demo_fallback: bool
    errors: list[str]
    discovered_model_files: list[str] = []
    load_attempted: bool = False
