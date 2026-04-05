# CoralGuard AI — FastAPI backend & dashboard

Production-oriented API and static web UI for **CoralGuard AI**, a tri-modal marine reef assessment pipeline aligned with the Colab export **`mdm.py`** (thesis / MDM project):

| Module | Role | Artifact |
|--------|------|----------|
| **A** | Coral health CNN | `CoralCNN` → `efficientnet_b3_best.pth` |
| **B** | Oceanographic regimes | `dbscan_model.pkl` + **kNN(1)** on `X_train.npy` |
| **C** | Bleaching regression | `FusionANN` → `ann_fusion_best.pth` |
| **+** | Narrative report | Groq **Llama 3.3** (optional) |

---

## Table of contents

1. [Repository layout](#repository-layout)  
2. [What ships in Git vs what stays local](#what-ships-in-git-vs-what-stays-local)  
3. [Syncing from `CoralGuardAI/` (Colab folder)](#syncing-from-coralguardai-colab-folder)  
4. [Quick start (local)](#quick-start-local)  
5. [Web dashboard](#web-dashboard)  
6. [HTTP API](#http-api)  
7. [Environment variables](#environment-variables)  
8. [Thesis figures (static)](#thesis-figures-static)  
9. [Docker & Compose](#docker--compose)  
10. [Deploying to a server](#deploying-to-a-server)  
11. [Pushing to GitHub](#pushing-to-github)  
12. [Troubleshooting](#troubleshooting)  
13. [Security](#security)  

---

## Repository layout

```
coralguard-api/
├── app/                          # FastAPI application
│   ├── main.py                   # App factory, CORS, /ui mount, /static
│   ├── inference.py              # Settings, engine, Groq report
│   ├── coral_cnn.py              # EfficientNet-B3 + head (matches mdm.py)
│   ├── fusion_ann.py             # Tabular + cluster one-hot → %
│   ├── tabular_mdm.py            # RobustScaler + feature engineering + kNN clusters
│   ├── schemas.py                # Pydantic models
│   ├── utils.py
│   ├── routers/predict.py        # POST /predict
│   └── models/                   # *.pth, *.pkl (see Git section)
├── frontend/                     # Plain HTML/CSS/JS dashboard
├── static/thesis/figures/        # Exported thesis PNGs (served under /static/...)
├── data/tabular/                 # features.pkl, scaler.pkl, X_train.npy, …
├── artifacts/reference/          # image_manifest.csv + notes
├── scripts/
│   └── sync_from_coralguard.ps1  # Copy assets from sibling CoralGuardAI/
├── uploads/                      # Ephemeral uploads (gitignored)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example                  # Template — copy to .env
├── LICENSE
└── README.md
```

---

## What ships in Git vs what stays local

**Intended to be in Git (clone-and-run):**

- All Python and frontend source  
- **Inference artifacts** (so Render / teammates need no manual copy):  
  - `app/models/efficientnet_b3_best.pth`, `dbscan_model.pkl`, `ann_fusion_best.pth`  
  - `data/tabular/features.pkl`, `scaler.pkl`, `X_train.npy`  
- `static/thesis/figures/*.png` — thesis plots  
- `scripts/`, `Dockerfile`, `docker-compose.yml`, docs  

**Never commit:**

| Path | Why |
|------|-----|
| `.env` | Contains `GROQ_API_KEY` and other secrets — use `.env.example` only in Git |

**Optional local-only** (training / eval, not required for `POST /predict`):  
`X_val.npy`, `X_test.npy`, `y_*.npy` — you can add them to Git too if you want; they are small.

**Public GitHub note:** Pushing weights makes the trained CNN/fusion/DBSCAN bundle **public**. For a thesis that is often fine; for proprietary data, use a **private** repo.

---

## Syncing from `CoralGuardAI/` (Colab folder)

If Colab output lives next to this repo, e.g. `S:\MDM\CoralGuardAI\`, run **from `coralguard-api/`**:

```powershell
.\scripts\sync_from_coralguard.ps1
```

Optional custom paths:

```powershell
.\scripts\sync_from_coralguard.ps1 -SourceRoot "D:\MyDrive\MDM\CoralGuardAI" -DestRoot "S:\MDM\coralguard-api"
```

**Copies:**

- `models/efficientnet_b3_best.pth`, `dbscan_model.pkl`, `ann_fusion_best.pth` → `app/models/`  
- `data/tabular/features.pkl`, `scaler.pkl`, `X_train.npy` (+ val/test/y arrays if present) → `data/tabular/`  
- `thesis/figures/*.png` → `static/thesis/figures/`  
- `image_manifest.csv` → `artifacts/reference/`  

---

## Quick start (local)

**Prerequisites:** Python **3.10+**, optional CUDA for GPU.

```powershell
cd S:\MDM\coralguard-api
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
copy .env.example .env
# Edit .env — set GROQ_API_KEY for LLM reports (optional for raw JSON inference)
```

**Run API + UI:**

```powershell
py -3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Open in browser (do not use `0.0.0.0` in the URL bar on Windows):**

- Dashboard: **http://127.0.0.1:8000/ui/**  
- Swagger: **http://127.0.0.1:8000/docs**  
- Health JSON: **http://127.0.0.1:8000/**  

---

## Web dashboard

The UI is **static HTML/CSS/JS** (no build step). It sends **multipart** `POST /predict` with:

- `image` — reef photo (JPEG/PNG/WebP)  
- `features` — JSON string of **`CoralCsvFeatures`** (ten `coral.csv` Phase-3 fields; engineering happens server-side)

If you open `frontend/index.html` from disk (`file://`), set the **API base URL** to `http://127.0.0.1:8000`.

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service name, version, paths to CNN/DBSCAN/fusion, `tabular_dir`, `frontend`, `docs` |
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/ui/` | Browser dashboard |
| `POST` | `/predict` | `multipart/form-data`: `image` (file), `features` (JSON **string**) |
| static | `/static/thesis/figures/<file>.png` | Thesis figures |

**`features` JSON shape** (`CoralCsvFeatures`):

```json
{
  "latitude_degrees": 23.163,
  "longitude_degrees": -82.526,
  "depth_m": 10,
  "turbidity": 0.0287,
  "cyclone_frequency": 49.9,
  "clim_sst": 301.61,
  "ssta": -0.46,
  "tsa": -0.8,
  "percent_cover": 0,
  "date_year": 2005
}
```

**curl (PowerShell):**

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
  -F "image=@S:\path\to\reef.png" `
  -F "features={\"latitude_degrees\":23.163,\"longitude_degrees\":-82.526,\"depth_m\":10,\"turbidity\":0.0287,\"cyclone_frequency\":49.9,\"clim_sst\":301.61,\"ssta\":-0.46,\"tsa\":-0.8,\"percent_cover\":0,\"date_year\":2005}"
```

---

## Environment variables

Defined in **`.env`** (from `.env.example`). Loaded by `pydantic-settings`.

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq API key → LLM report (`llama-3.3-70b-versatile` by default). Empty = skip report, return model JSON only. |
| `GROQ_MODEL` | Override model id. |
| `CORALGUARD_CNN_PATH` | Path to `efficientnet_b3_best.pth` |
| `CORALGUARD_DBSCAN_PATH` | Path to `dbscan_model.pkl` |
| `CORALGUARD_FUSION_PATH` | Path to `ann_fusion_best.pth` (falls back to `fusion_ann.pth` in same folder) |
| `CORALGUARD_TABULAR_DIR` | Directory with `features.pkl`, `scaler.pkl`, `X_train.npy` (default **`data/tabular`** under repo root) |
| `CORALGUARD_DEVICE` | `auto`, `cpu`, or `cuda` |
| `CORALGUARD_CNN_IMAGE_SIZE` | Default **224** (matches `mdm.py`) |
| `CORALGUARD_CLASS_NAMES` | Comma-separated, default `Healthy,Bleached,Dead` |

---

## Thesis figures (static)

After sync, PNGs are under **`static/thesis/figures/`**, including (names from Colab):

- `image_class_distribution.png`, `sample_images_grid.png`, `image_dimensions.png`, `rgb_channel_analysis.png`  
- `target_distribution.png`, `feature_correlation_heatmap.png`, `feature_importance_mi.png`  
- `preprocessing_comparison.png`, `augmentation_samples.png`  
- `confusion_matrix.png`, `roc_curves.png`, `gradcam_visualizations.png`  
- `dbscan_kdistance.png`, `ann_regression_analysis.png`, `ablation_study_comparison.png`  

URL pattern: **`http://127.0.0.1:8000/static/thesis/figures/<filename>.png`**

---

## Docker & Compose

**Build** (from repo root):

```powershell
docker compose build
```

**Run** (expects `.env` plus host folders `app/models` and `data/tabular` populated):

```powershell
docker compose up -d
```

Compose **bind-mounts** `./app/models` and `./data/tabular` read-only so the image stays small; thesis PNGs are **baked in** via `COPY static` if present at build time.

**Single-container run without Compose** (example):

```powershell
docker build -t coralguard-api .
docker run --rm -p 8000:8000 --env-file .env `
  -v "${PWD}/app/models:/app/app/models:ro" `
  -v "${PWD}/data/tabular:/app/data/tabular:ro" `
  coralguard-api
```

---

## Deploying to a server

1. Copy repo + run `sync_from_coralguard.ps1` (or `scp` models/tabular).  
2. Set `GROQ_API_KEY` in environment or `.env` (use secrets manager in production).  
3. Put **HTTPS** reverse proxy (Caddy, nginx, Traefik) in front; restrict CORS in `app/main.py` from `["*"]` to your frontend origin.  
4. Process manager: **systemd**, **Docker**, or **cloud run** with the same env + volumes.  
5. Do **not** expose `.env` or upload directories to the public web root.

### Render ([render.com](https://render.com))

**Ready:** `Dockerfile` uses **`PORT`**, copies **`data/tabular`** into the image, and models ship under **`app/models/`** in Git. `render.yaml` is Docker + health check; add **`GROQ_API_KEY`** in the dashboard.

`render.yaml` uses **`plan: free`** (no paid instance). If PyTorch OOMs or the service crashes, switch to **`plan: starter`** in `render.yaml` (paid).

Full steps: **[docs/DEPLOY_RENDER.md](docs/DEPLOY_RENDER.md)**.

---

## Pushing to GitHub

**One-time (if not already a repo):**

```powershell
cd S:\MDM\coralguard-api
git init
git add .
git status   # confirm no .env and no *.pth / *.pkl / *.npy listed
git commit -m "Initial commit: CoralGuard AI API, UI, thesis figures"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

**Before every push:**

- `git status` — ensure `.env` is **not** staged (artifacts **may** be staged — they are no longer gitignored).  
- If you added real keys to `.env`, **rotate** them at the provider (see Security).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Tabular bundle not loaded` | Ensure `data/tabular` has `features.pkl`, `scaler.pkl`, `X_train.npy`; check `CORALGUARD_TABULAR_DIR`. |
| `DBSCAN labels length != X_train rows` | Re-export `dbscan_model.pkl` and `X_train.npy` from the **same** Colab run. |
| `422` on `/predict` / `input_value='string'` | Swagger default placeholder — paste real JSON in `features` (see dashboard or README example). |
| Browser cannot open `http://0.0.0.0:8000` | Use **`127.0.0.1`** or **`localhost`**. |
| CNN load errors | Confirm `efficientnet_b3_best.pth` matches `CoralCNN` in `coral_cnn.py` (mdm.py Phase 4). |
| No LLM report | Set `GROQ_API_KEY`; check Groq quota / model name. |

---

## Security

- **Never commit `.env`.** It is listed in `.gitignore`.  
- If an API key was ever pasted into a tracked file or shared, **revoke and create a new key** in the [Groq console](https://console.groq.com/).  
- For production, use least-privilege secrets, HTTPS, and tight CORS.

---

## License

See [LICENSE](LICENSE) (MIT).

---

## Citation / project context

CoralGuard AI — marine ecosystem anomaly detection for thesis / SDG 14 alignment. Training and figure generation follow the Colab notebook exported as **`mdm.py`** in the parent MDM project.
