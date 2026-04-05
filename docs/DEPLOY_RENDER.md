# Deploy CoralGuard AI on [Render](https://render.com)

## Is it “ready”?

| Piece | Status |
|-------|--------|
| FastAPI + health route `/health` | Yes |
| `Dockerfile` + dynamic **`PORT`** | Yes (required by Render) |
| `render.yaml` Blueprint | Yes (optional one-click shape) |
| **Weights + tabular files** in Git | **Yes** — `efficientnet_b3_best.pth`, `dbscan_model.pkl`, `ann_fusion_best.pth`, `features.pkl`, `scaler.pkl`, `X_train.npy` are tracked so the image build includes them (no manual disk upload unless you choose to strip them again). |
| **RAM** | **Free** works for many stacks; this app uses PyTorch + CNN — if the service **crashes or 502s**, upgrade to **Starter** (paid). |

If you **removed** artifacts from Git for a slimmer clone, use a **persistent disk** (as below) or restore files on the host before serving traffic.

---

## Option A — Blueprint (Docker)

1. Push this repo to GitHub (weights and `data/tabular` are **in Git** so the image builds complete).
2. Render Dashboard → **New** → **Blueprint** → select the repo.
3. Add secret **`GROQ_API_KEY`** when prompted (optional for JSON-only inference).

**URLs after deploy**

- API docs: `https://<service-name>.onrender.com/docs`
- UI: `https://<service-name>.onrender.com/ui/`
- Figures: `https://<service-name>.onrender.com/static/thesis/figures/`

---

## Option B — Manual Web Service (no Blueprint)

1. **New** → **Web Service** → connect repo.
2. **Runtime:** Docker → `Dockerfile` at repo root.
3. Add **`GROQ_API_KEY`** (secret) if you want LLM reports.
4. Default paths in `.env.example` (`app/models/...`, `data/tabular`) work inside the container.

---

## Free tier (default in `render.yaml`)

- **No paid instance** — Blueprint should not ask for a card for `plan: free` (workspace policies may vary; see [Render free docs](https://docs.render.com/docs/free)).
- **512 MB RAM** may be tight for `torch` + EfficientNet-B3; if builds fail or the app OOMs, switch `plan:` to `starter` in `render.yaml` (requires payment method on file).
- **Cold starts** after ~15 min idle are normal on free web services.

---

## CORS

`main.py` currently allows `allow_origins=["*"]`. For a separate frontend domain, narrow this to your exact origin in production.

---

## Cold starts

On free/starter Docker services, the first request after idle can be slow while the container starts. Consider **Render paid** plans or a **health cron** ping if you need warmer behavior.
