# CoralGuard AI

Marine ecosystem monitoring web application that classifies coral health from underwater imagery and oceanographic parameters, generates Groq LLaMA-3 conservation reports, and autonomously dispatches email/SMS alerts to marine officials.

## Architecture

```
Frontend (Next.js 14)  →  FastAPI Backend  →  TensorFlow + DBSCAN + Groq + Twilio
```

- **CNN Classification**: EfficientNet-B3 fusion model (image + 13 tabular features)
- **Ocean Environment**: DBSCAN clustering on scaled parameters
- **Generative AI**: Groq `llama-3.1-8b-instant` conservation reports (configurable via `GROQ_MODEL`)
- **Agentic AI**: Autonomous SendGrid email + Twilio SMS alerts when thresholds are met

## Project Structure

```
coralguard/
├── frontend/          Next.js 14 App Router dashboard
├── backend/           FastAPI ML + GenAI + alert agent
│   └── models/        Place trained model files here
└── README.md
```

## Model Files

Download from Google Drive and place in `backend/models/`:

| File | Drive Path |
|------|------------|
| `coralguard_fusion_best.keras` | `/MDM/CoralGuardAI/models/coralguard_fusion_best.keras` |
| `dbscan_model.pkl` | `/MDM/CoralGuardAI/models/dbscan_model.pkl` |
| `scaler.pkl` | `/MDM/CoralGuardAI/outputs/tabular/scaler.pkl` |
| `features.pkl` | `/MDM/CoralGuardAI/outputs/tabular/features.pkl` |

## Environment Setup

### Backend (`backend/.env`)

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
TABULAR_BLEND_ALPHA=0.12
SENDGRID_API_KEY=your_sendgrid_api_key
ALERT_EMAIL_SENDER=verified-sender@yourdomain.com
ALERT_EMAIL_RECIPIENT=marine.official@agency.gov
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
ALERT_SMS_NUMBER=+91XXXXXXXXXX
MODEL_PATH=./models/coralguard_fusion_best.keras
DBSCAN_PATH=./models/dbscan_model.pkl
SCALER_PATH=./models/scaler.pkl
FEATURES_PATH=./models/features.pkl
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Install & Run

### Backend

```powershell
cd coralguard/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```powershell
cd coralguard/frontend
npm install
npm run dev
```

App: http://localhost:3000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Server and model load status |
| `POST` | `/api/predict` | Multipart: `image` + `params` JSON → classification + cluster |
| `POST` | `/api/report` | JSON prediction context → Groq conservation report |
| `POST` | `/api/alert` | JSON results → autonomous email/SMS alert decision |

### Alert Thresholds (Autonomous)

- **CRITICAL** (email + SMS): Dead coral, confidence ≥ 60%
- **CRITICAL** (email + SMS): Bleached + anomaly zone, confidence ≥ 75%
- **HIGH** (email only): Bleached, confidence ≥ 75%
- **No alert**: All other cases

## Pages

| Route | Purpose |
|-------|---------|
| `/` | Landing hero with feature overview |
| `/dashboard` | Image upload, 10 parameter sliders, analyze |
| `/results` | Classification, environment, report, alert status |

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Recharts, Framer Motion
- **Backend**: FastAPI, TensorFlow 2.x, OpenCV, scikit-learn, Groq SDK, Twilio

## License

Academic / research use — CoralGuard AI marine monitoring system.
