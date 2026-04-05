# CoralGuard AI — FastAPI backend & dashboard

API and static web UI for **CoralGuard AI**, a tri-modal marine reef assessment pipeline aligned with the Colab export **`mdm.py`** (thesis / MDM project):

| Module | Role | Artifact |
|--------|------|----------|
| **A** | Coral health CNN | `CoralCNN` → `efficientnet_b3_best.pth` |
| **B** | Oceanographic regimes | `dbscan_model.pkl` + **kNN(1)** on `X_train.npy` |
| **C** | Bleaching regression | `FusionANN` → `ann_fusion_best.pth` |
| **+** | Narrative report | Groq **Llama 3.3** (optional) |

**This repo is maintained for local development and GitHub backup.** Run the stack on your machine (see [Quick start](#quick-start-local)); there are no cloud deployment configs in-tree.

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
9. [Pushing to GitHub](#pushing-to-github)  
10. [Troubleshooting](#troubleshooting)  
11. [Security](#security)  

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
│   ├── schemas.py
│   ├── utils.py
│   ├── routers/predict.py        # POST /predict
│   └── models/                   # *.pth, *.pkl (trained weights — in Git)
├── frontend/                     # Plain HTML/CSS/JS dashboard
├── static/thesis/figures/        # Thesis PNGs (served under /static/...)
├── data/tabular/                 # features.pkl, scaler.pkl, X_train.npy, …
├── artifacts/reference/          # image_manifest.csv + notes
├── scripts/
│   └── sync_from_coralguard.ps1  # Copy assets from sibling CoralGuardAI/
├── uploads/                      # Ephemeral uploads (gitignored)
├── requirements.txt
├── .env.example                  # Template — copy to .env
├── LICENSE
└── README.md
```

---

## What ships in Git vs what stays local

**In Git (clone + local run):**

- Application source (`app/`, `frontend/`)  
- **Inference artifacts:** `app/models/*.pth`, `*.pkl` and `data/tabular/*` (scaler, features list, `X_train.npy`, etc.)  
- `static/thesis/figures/*.png`  
- `scripts/sync_from_coralguard.ps1`, `artifacts/reference/`  

**Never commit:**

| Path | Why |
|------|-----|
| `.env` | Secrets (`GROQ_API_KEY`) — use `.env.example` only in Git |

**Optional:** `X_val.npy`, `X_test.npy`, `y_*.npy` (eval only; may already be tracked).

**Public repo:** Weights are world-readable. Use a **private** repo if that matters.

---

## Syncing from `CoralGuardAI/` (Colab folder)

From `coralguard-api/`:

```powershell
.\scripts\sync_from_coralguard.ps1
```

Custom paths:

```powershell
.\scripts\sync_from_coralguard.ps1 -SourceRoot "D:\MyDrive\MDM\CoralGuardAI" -DestRoot "S:\MDM\coralguard-api"
```

---

## Quick start (local)

**Prerequisites:** Python **3.10+**, optional CUDA.

```powershell
cd path\to\coralguard-api
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
copy .env.example .env
# Edit .env — set GROQ_API_KEY for LLM reports (optional)
```

**Run:**

```powershell
py -3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Open in the browser** (use `127.0.0.1`, not `0.0.0.0` on Windows):

- Dashboard: **http://127.0.0.1:8000/ui/**  
- Swagger: **http://127.0.0.1:8000/docs**  
- Health: **http://127.0.0.1:8000/**  

---

## Web dashboard

Static **HTML/CSS/JS**. Sends `POST /predict` with an image + JSON **`CoralCsvFeatures`**.

If you open `frontend/index.html` via `file://`, set the **API base URL** to `http://127.0.0.1:8000`.

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info, model paths, `tabular_dir`, `frontend`, `docs` |
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/ui/` | Dashboard |
| `POST` | `/predict` | `multipart`: `image`, `features` (JSON string) |
| static | `/static/thesis/figures/<file>.png` | Thesis figures |

**Example `features` JSON:**

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

---

## Environment variables

Set in **`.env`** (see `.env.example`).

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq → LLM report (optional) |
| `GROQ_MODEL` | Default `llama-3.3-70b-versatile` |
| `CORALGUARD_CNN_PATH` | CNN weights path |
| `CORALGUARD_DBSCAN_PATH` | DBSCAN pickle path |
| `CORALGUARD_FUSION_PATH` | Fusion checkpoint path |
| `CORALGUARD_TABULAR_DIR` | Default `data/tabular` |
| `CORALGUARD_DEVICE` | `auto`, `cpu`, `cuda` |
| `CORALGUARD_CNN_IMAGE_SIZE` | Default **224** |
| `CORALGUARD_CLASS_NAMES` | `Healthy,Bleached,Dead` |

---

## Thesis figures (static)

Under **`static/thesis/figures/`** — e.g. EDA, confusion matrix, ROC, Grad-CAM, DBSCAN, ANN, ablation.

**Local URL:** `http://127.0.0.1:8000/static/thesis/figures/<filename>.png`

---

## Pushing to GitHub

```powershell
cd path\to\coralguard-api
git status          # ensure .env is not staged
git add -A
git commit -m "Your message"
git push origin main
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Tabular bundle not loaded | Check `data/tabular` and `CORALGUARD_TABULAR_DIR` |
| DBSCAN / X_train mismatch | Re-export from the same Colab run |
| `422` / `string` in features | Paste real JSON, not Swagger placeholder |
| `0.0.0.0` won’t load in browser | Use **127.0.0.1** or **localhost** |
| No LLM report | Set `GROQ_API_KEY` |

---

## Security

- Never commit **`.env`**.  
- Rotate keys if they were ever exposed.

---

## License

[LICENSE](LICENSE) (MIT).

---

## Project context

CoralGuard AI — marine ecosystem anomaly detection / SDG 14. Training follows **`mdm.py`** in the parent MDM project.
