# Deploy CoralGuard AI on [Render](https://render.com)

## Is it “ready”?

| Piece | Status |
|-------|--------|
| FastAPI + health route `/health` | Yes |
| `Dockerfile` + dynamic **`PORT`** | Yes (required by Render) |
| `render.yaml` Blueprint | Yes (optional one-click shape) |
| **Weights + tabular files** in Git | **No** (`.gitignore`) — you **must** supply them on the server |
| **RAM** | PyTorch + CNN is heavy — **avoid Free** if you see OOM; use **Starter** or higher |

So: **the app is ready to deploy as a service**, but **you still need to copy `app/models/*` and `data/tabular/*` onto Render** (persistent disk or another method) before `/predict` works end-to-end.

---

## Option A — Blueprint (Docker + disk)

1. Push this repo to GitHub (already done for [coral_guard-ai](https://github.com/rajiv-golait/coral_guard-ai)).
2. Render Dashboard → **New** → **Blueprint**.
3. Select the repo; Render reads `render.yaml`.
4. In the dashboard, create an **environment variable** `GROQ_API_KEY` (mark secret) if you want LLM reports.
5. After the first deploy, open **Shell** (paid instances) or use **SSH** if enabled, and create directories on the mounted disk:

   ```bash
   mkdir -p /data/models /data/tabular
   ```

6. Upload files from your laptop (same layout as local project):

   - `/data/models/efficientnet_b3_best.pth`
   - `/data/models/dbscan_model.pkl`
   - `/data/models/ann_fusion_best.pth`
   - `/data/tabular/features.pkl`
   - `/data/tabular/scaler.pkl`
   - `/data/tabular/X_train.npy`

   Use Render’s docs for your workflow (shell + `curl` from a private URL, `scp`, CI artifact, etc.). **Do not** commit secrets or large blobs to a public repo.

7. **Redeploy** or restart the service so the process picks up files (usually not required if files appear before requests).

**URLs after deploy**

- API docs: `https://<service-name>.onrender.com/docs`
- UI: `https://<service-name>.onrender.com/ui/`
- Figures: `https://<service-name>.onrender.com/static/thesis/figures/`

---

## Option B — Manual Web Service (no Blueprint)

1. **New** → **Web Service** → connect repo.
2. **Runtime:** Docker (use root `Dockerfile`) *or* Native Python:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Add **environment variables** (same names as `.env.example`). Set paths to wherever you mount or copy artifacts (`/data/...` if using a disk).
4. Attach a **persistent disk** and upload artifacts as in Option A.

---

## Free tier warning

- **512 MB RAM** is often **not enough** for `torch` + EfficientNet-B3 + requests.
- If the service **crashes or 502s** during import or first inference, upgrade to **Starter** or higher.

---

## CORS

`main.py` currently allows `allow_origins=["*"]`. For a separate frontend domain, narrow this to your exact origin in production.

---

## Cold starts

On free/starter Docker services, the first request after idle can be slow while the container starts. Consider **Render paid** plans or a **health cron** ping if you need warmer behavior.
