import json
import os
from typing import Any

from groq import Groq

from schemas import ConservationReport, ReportRequest

# llama-3-8b-8192 was retired on Groq; use llama-3.1-8b-instant (or override via .env)
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _get_groq_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL

SYSTEM_PROMPT = (
    "You are CoralGuard AI, an expert marine biologist AI assistant. "
    "You provide scientifically accurate, actionable conservation assessments. "
    "Always be specific and practical. Respond only in valid JSON format."
)


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured in backend/.env")
    return Groq(api_key=api_key)


def _build_user_prompt(req: ReportRequest) -> str:
    return f"""Analyze this coral reef monitoring assessment and produce a conservation report.

Site: {req.site_name}
Coordinates: {req.latitude}°N, {req.longitude}°E
Depth: {req.depth_m} m
Turbidity: {req.turbidity} NTU
SSTA: {req.ssta} °C
TSA: {req.tsa} °C

AI Classification:
- Health: {req.health_class}
- Confidence: {req.confidence:.1%}
- Probabilities: Healthy {req.probabilities.get('Healthy', 0):.1%}, Bleached {req.probabilities.get('Bleached', 0):.1%}, Dead {req.probabilities.get('Dead', 0):.1%}

Ocean Environment:
- Zone: {req.cluster_name}
- Anomaly detected: {req.is_anomaly}
- Risk level: {req.risk_level}
- Thermal stress index: {req.thermal_stress:.3f}
- Light index: {req.light_index:.3f}
- SST total: {req.sst_total:.2f} °C

Respond with JSON only in this exact structure:
{{
  "executive_summary": "2-3 sentences",
  "risk_level": "CRITICAL|HIGH|MODERATE|LOW",
  "key_threats": ["threat1", "threat2", "threat3"],
  "immediate_actions": ["action1", "action2", "action3", "action4"],
  "preventive_measures": ["measure1", "measure2", "measure3", "measure4"],
  "monitoring_schedule": "specific schedule string",
  "recovery_prognosis": "prognosis string",
  "scientific_context": "brief scientific context"
}}"""


def _parse_report_json(raw: dict[str, Any], fallback_risk: str) -> ConservationReport:
    return ConservationReport(
        executive_summary=str(raw.get("executive_summary", "Assessment complete.")),
        risk_level=raw.get("risk_level", fallback_risk),  # type: ignore[arg-type]
        key_threats=list(raw.get("key_threats", []))[:5] or ["Environmental stress"],
        immediate_actions=list(raw.get("immediate_actions", []))[:5]
        or ["Conduct field survey", "Monitor water temperature", "Document bleaching extent"],
        preventive_measures=list(raw.get("preventive_measures", []))[:5]
        or ["Reduce local pollution", "Establish marine protected area"],
        monitoring_schedule=str(
            raw.get("monitoring_schedule", "Weekly visual surveys for 4 weeks, then bi-weekly.")
        ),
        recovery_prognosis=str(
            raw.get("recovery_prognosis", "Recovery depends on stress reduction and water quality.")
        ),
        scientific_context=str(
            raw.get(
                "scientific_context",
                "Coral bleaching occurs when symbiotic algae are expelled under thermal stress.",
            )
        ),
    )


def generate_conservation_report(req: ReportRequest) -> ConservationReport:
    model = _get_groq_model()
    print(f"[groq_agent] Generating conservation report via Groq ({model})...")
    client = _get_client()
    user_prompt = _build_user_prompt(req)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        raw = json.loads(content)
        report = _parse_report_json(raw, req.risk_level)
        print("[groq_agent] Conservation report generated successfully")
        return report
    except json.JSONDecodeError as exc:
        print(f"[groq_agent] JSON parse error: {exc}")
        raise ValueError("Groq returned invalid JSON for conservation report") from exc
    except Exception as exc:
        print(f"[groq_agent] Groq API error: {exc}")
        raise
