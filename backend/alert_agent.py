import base64
import os
from datetime import datetime, timezone
from typing import Literal, Optional

from schemas import AlertRequest, AlertResponse

AlertType = Literal["CRITICAL", "HIGH"]


def decide_alert(req: AlertRequest) -> tuple[bool, Optional[AlertType], str]:
    """Autonomous agent decision — no human trigger required."""
    hc = req.health_class
    conf = req.confidence
    anomaly = req.is_anomaly

    if hc == "Dead" and conf >= 0.60:
        return True, "CRITICAL", "Dead coral detected with confidence >= 60%"
    if hc == "Bleached" and conf >= 0.75 and anomaly:
        return (
            True,
            "CRITICAL",
            "Bleached coral in anomaly zone with confidence >= 75%",
        )
    if hc == "Bleached" and conf >= 0.75:
        return True, "HIGH", "Bleached coral with confidence >= 75%"
    return False, None, "Conditions do not meet alert thresholds"


def _banner_color(alert_type: AlertType) -> str:
    return "#dc2626" if alert_type == "CRITICAL" else "#ea580c"


def _google_maps_url(latitude: float, longitude: float) -> str:
    return f"https://www.google.com/maps?q={latitude:.6f},{longitude:.6f}"


def _build_html_email(req: AlertRequest, alert_type: AlertType) -> str:
    color = _banner_color(alert_type)
    actions_html = "".join(
        f"<li>{a}</li>" for a in (req.immediate_actions or [])[:4]
    )
    if not actions_html:
        actions_html = "<li>Deploy field team for immediate reef assessment</li>"

    maps_url = _google_maps_url(req.latitude, req.longitude)

    return f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; color: #1e293b; max-width: 640px;">
  <div style="background:{color}; color:white; padding:16px; border-radius:8px 8px 0 0;">
    <h2 style="margin:0;">[{alert_type}] CoralGuard AI Alert</h2>
    <p style="margin:8px 0 0;">{req.health_class} coral detected at {req.site_name}</p>
  </div>
  <div style="padding:20px; border:1px solid #e2e8f0; border-top:none;">
    <h3>Site Information</h3>
    <table style="width:100%; border-collapse:collapse;">
      <tr><td><b>Site</b></td><td>{req.site_name}</td></tr>
      <tr><td><b>Coordinates</b></td><td>{req.latitude:.2f}°, {req.longitude:.2f}°</td></tr>
      <tr><td><b>Map</b></td><td><a href="{maps_url}" style="color:#0d9488;">Open in Google Maps</a></td></tr>
      <tr><td><b>Depth</b></td><td>{req.depth_m} m</td></tr>
      <tr><td><b>Timestamp</b></td><td>{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</td></tr>
    </table>
    <h3>AI Assessment</h3>
    <p><b>Health:</b> {req.health_class} &nbsp;|&nbsp; <b>Confidence:</b> {req.confidence:.1%}<br>
    <b>Risk:</b> {req.risk_level} &nbsp;|&nbsp; <b>Environment:</b> {req.cluster_name}</p>
    <p>{req.executive_summary}</p>
    <h3>Immediate Actions</h3>
    <ol>{actions_html}</ol>
    <hr>
    <p style="font-size:12px; color:#64748b;">CoralGuard AI — Autonomous Marine Monitoring System</p>
  </div>
</body>
</html>"""


def _build_plain_email(req: AlertRequest, alert_type: AlertType) -> str:
    maps_url = _google_maps_url(req.latitude, req.longitude)
    actions = "\n".join(f"  - {a}" for a in (req.immediate_actions or [])[:4])
    return (
        f"CoralGuard AI [{alert_type}] — {req.health_class} at {req.site_name}\n"
        f"Confidence: {req.confidence:.1%} | Risk: {req.risk_level}\n"
        f"Coordinates: {req.latitude:.2f}, {req.longitude:.2f}\n"
        f"Google Maps: {maps_url}\n"
        f"Depth: {req.depth_m} m\n"
        f"{req.executive_summary}\n"
        f"Immediate actions:\n{actions or '  - Deploy field team for reef assessment'}\n"
    )


def _image_mime_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    return "image/jpeg"


def _send_email(req: AlertRequest, alert_type: AlertType) -> bool:
    api_key = os.getenv("SENDGRID_API_KEY", "").strip()
    sender = os.getenv("ALERT_EMAIL_SENDER", "").strip()
    recipient = (req.override_email or os.getenv("ALERT_EMAIL_RECIPIENT", "")).strip()

    if not api_key or not sender or not recipient:
        print(
            "[alert_agent] Email skipped: set SENDGRID_API_KEY, "
            "ALERT_EMAIL_SENDER, and ALERT_EMAIL_RECIPIENT in backend/.env"
        )
        return False

    subject = (
        f"[{alert_type}] CoralGuard AI Alert — {req.health_class} "
        f"Coral at {req.site_name}"
    )

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import (
            Attachment,
            Disposition,
            FileContent,
            FileName,
            FileType,
            Mail,
        )

        message = Mail(
            from_email=sender,
            to_emails=recipient,
            subject=subject,
            html_content=_build_html_email(req, alert_type),
            plain_text_content=_build_plain_email(req, alert_type),
        )

        if req.image_base64:
            try:
                img_data = base64.b64decode(req.image_base64)
                filename = req.image_filename or "coral_image.jpg"
                message.attachment = Attachment(
                    FileContent(base64.b64encode(img_data).decode()),
                    FileName(filename),
                    FileType(_image_mime_type(filename)),
                    Disposition("attachment"),
                )
            except Exception as exc:
                print(f"[alert_agent] Could not attach image: {exc}")

        print(f"[alert_agent] Sending email via SendGrid to {recipient}...")
        client = SendGridAPIClient(api_key)
        response = client.send(message)
        status = response.status_code
        if 200 <= status < 300:
            print(f"[alert_agent] Email sent successfully (HTTP {status})")
            return True
        print(f"[alert_agent] SendGrid returned HTTP {status}: {response.body}")
        return False
    except Exception as exc:
        print(f"[alert_agent] Email failed: {exc}")
        return False


def _send_sms(req: AlertRequest) -> bool:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    to_number = os.getenv("ALERT_SMS_NUMBER", "").strip()

    if not all([account_sid, auth_token, from_number, to_number]):
        print("[alert_agent] SMS skipped: missing Twilio configuration")
        return False

    maps_url = _google_maps_url(req.latitude, req.longitude)
    body = (
        f"CoralGuard AI ALERT — {req.site_name}: {req.health_class} coral "
        f"({req.confidence:.0%} confidence) at {req.latitude:.2f},{req.longitude:.2f}. "
        f"Map: {maps_url} Immediate action required."
    )

    try:
        from twilio.rest import Client

        print(f"[alert_agent] Sending SMS to {to_number}...")
        client = Client(account_sid, auth_token)
        client.messages.create(body=body, from_=from_number, to=to_number)
        print("[alert_agent] SMS sent successfully")
        return True
    except Exception as exc:
        print(f"[alert_agent] SMS failed: {exc}")
        return False


def process_alert(req: AlertRequest) -> AlertResponse:
    print("[alert_agent] Autonomous alert agent evaluating thresholds...")
    triggered, alert_type, reason = decide_alert(req)

    if not triggered or alert_type is None:
        print(f"[alert_agent] No alert: {reason}")
        return AlertResponse(
            triggered=False,
            alert_type=None,
            email_sent=False,
            sms_sent=False,
            message=reason,
        )

    print(f"[alert_agent] Alert TRIGGERED: {alert_type} — {reason}")
    email_sent = _send_email(req, alert_type)
    sms_sent = False
    if alert_type == "CRITICAL":
        sms_sent = _send_sms(req)

    parts = []
    if email_sent:
        parts.append("email dispatched")
    if sms_sent:
        parts.append("SMS dispatched")
    if not parts:
        parts.append("alert triggered but notifications failed — check .env credentials")

    return AlertResponse(
        triggered=True,
        alert_type=alert_type,
        email_sent=email_sent,
        sms_sent=sms_sent,
        message=f"{alert_type} alert: {', '.join(parts)}",
    )
