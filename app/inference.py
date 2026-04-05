"""
Model loading and inference aligned with Colab `mdm.py` (CoralGuard AI).

- **Module A**: `CoralCNN` — EfficientNet-B3 + custom head, 224×224, weights `efficientnet_b3_best.pth`
- **Module B**: DBSCAN artifact + **kNN(1)** on training `X_train.npy` with saved `labels` (Colab has no `predict`)
- **Module C**: `FusionANN(tab, clust)` — `ann_fusion_best.pth` (fallback: `fusion_ann.pth`)

Tabular inputs follow **Phase 3** `coral.csv` features + engineering; scaling via `scaler.pkl` and
column order via `features.pkl` under `CORALGUARD_TABULAR_DIR`.
"""

from __future__ import annotations

import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from groq import Groq
from PIL import Image
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.coral_cnn import CoralCNN
from app.fusion_ann import FusionANN
from app.schemas import CNNResult, CoralCsvFeatures, DBSCANResult, FusionResult
from app.tabular_mdm import MdmTabularArtifacts, load_mdm_tabular_bundle
from app.utils import ROOT_DIR, get_logger

log = get_logger()

APP_DIR = Path(__file__).resolve().parent


def _default_tabular_dir() -> Path:
    """Bundled tabular artifacts under repo `data/tabular` (deployment default)."""
    return (ROOT_DIR / "data" / "tabular").resolve()


class Settings(BaseSettings):
    """Environment-driven configuration (.env in project root)."""

    groq_api_key: str = Field(default="", validation_alias=AliasChoices("GROQ_API_KEY"))
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("GROQ_MODEL"),
    )

    cnn_path: Path = Field(
        default=APP_DIR / "models" / "efficientnet_b3_best.pth",
        validation_alias=AliasChoices("CORALGUARD_CNN_PATH"),
    )
    dbscan_path: Path = Field(
        default=APP_DIR / "models" / "dbscan_model.pkl",
        validation_alias=AliasChoices("CORALGUARD_DBSCAN_PATH"),
    )
    fusion_path: Path = Field(
        default=APP_DIR / "models" / "ann_fusion_best.pth",
        validation_alias=AliasChoices("CORALGUARD_FUSION_PATH"),
    )

    tabular_dir: Path = Field(
        default_factory=_default_tabular_dir,
        validation_alias=AliasChoices("CORALGUARD_TABULAR_DIR"),
    )

    device: str = Field(default="auto", validation_alias=AliasChoices("CORALGUARD_DEVICE"))
    cnn_image_size: int = Field(default=224, validation_alias=AliasChoices("CORALGUARD_CNN_IMAGE_SIZE"))
    cnn_class_names: str = Field(
        default="Healthy,Bleached,Dead",
        validation_alias=AliasChoices("CORALGUARD_CLASS_NAMES"),
    )

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for name in ("cnn_path", "dbscan_path", "fusion_path", "tabular_dir"):
        p = getattr(s, name)
        if not p.is_absolute():
            setattr(s, name, (ROOT_DIR / p).resolve())
    return s


def _pick_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def _parse_class_names(raw: str) -> list[str]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts if parts else ["Healthy", "Bleached", "Dead"]


def _build_cnn_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _strip_prefix_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


class CoralGuardEngine:
    """Loads Colab-exported artifacts and runs Module A/B/C."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = _pick_device(self.settings.device)
        self.class_names = _parse_class_names(self.settings.cnn_class_names)
        self._transform = _build_cnn_transform(self.settings.cnn_image_size)

        self._cnn: torch.nn.Module | None = None
        self._fusion: FusionANN | None = None
        self._tabular: MdmTabularArtifacts | None = None
        self._dbscan_info: dict[str, Any] = {}

    def load_all(self) -> None:
        self._load_tabular_and_dbscan()
        self._load_cnn()
        self._load_fusion()

    def _load_tabular_and_dbscan(self) -> None:
        self._tabular = load_mdm_tabular_bundle(
            self.settings.tabular_dir,
            self.settings.dbscan_path,
        )
        if self._tabular is None:
            log.warning(
                "Tabular/DBSCAN bundle not loaded — copy `features.pkl`, `scaler.pkl`, "
                "`X_train.npy` into %s and ensure dbscan_model.pkl matches Colab export.",
                self.settings.tabular_dir,
            )
            return

        try:
            with open(self.settings.dbscan_path, "rb") as fp:
                raw = pickle.load(fp)
            if isinstance(raw, dict) and "params" in raw:
                self._dbscan_info = {"params": raw.get("params"), "n_clusters": raw.get("n_clusters")}
        except Exception:  # noqa: BLE001
            self._dbscan_info = {}

    def _load_cnn(self) -> None:
        path = self.settings.cnn_path
        if not path.is_file():
            log.warning("CNN weights not found at %s — Module A disabled", path)
            self._cnn = None
            return

        n_cls = len(self.class_names)
        model = CoralCNN(num_classes=n_cls, dropout=0.3).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict") or ckpt.get("model_state_dict")
            if state is None:
                state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            state = ckpt  # type: ignore[assignment]

        cleaned = _strip_prefix_state_dict(state)  # type: ignore[arg-type]
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            log.info("CoralCNN load_state_dict: missing=%s unexpected=%s", missing, unexpected)

        model.eval()
        self._cnn = model
        log.info("Loaded CoralCNN from %s", path)

    def _resolve_fusion_path(self) -> Path | None:
        p = self.settings.fusion_path
        if p.is_file():
            return p
        alt = p.parent / "fusion_ann.pth"
        if alt.is_file():
            log.info("Using fallback fusion weights at %s", alt)
            return alt
        log.warning("Fusion weights not found at %s or fusion_ann.pth", p)
        return None

    def _load_fusion(self) -> None:
        path = self._resolve_fusion_path()
        if path is None or self._tabular is None:
            self._fusion = None
            if self._tabular is None:
                log.warning("Fusion ANN skipped — tabular bundle required for architecture dims")
            return

        n_tab = self._tabular.n_tab
        n_clust = self._tabular.n_clust_cols
        model = FusionANN(n_tab=n_tab, n_clust=n_clust, dropout=0.3).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            log.error("Unexpected fusion checkpoint format")
            self._fusion = None
            return

        cleaned = _strip_prefix_state_dict(state)  # type: ignore[arg-type]
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            log.info("FusionANN load_state_dict: missing=%s unexpected=%s", missing, unexpected)

        model.eval()
        self._fusion = model
        log.info("Loaded FusionANN from %s (n_tab=%s n_clust=%s)", path, n_tab, n_clust)

    def predict_cnn(self, image_path: Path) -> CNNResult:
        if self._cnn is None:
            raise RuntimeError("CNN weights not loaded")

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        tensor = self._transform(image=arr)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._cnn(tensor)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)

        idx = int(np.argmax(probs))
        pred = self.class_names[idx] if idx < len(self.class_names) else str(idx)
        prob_dict = {
            self.class_names[i]: float(probs[i]) for i in range(min(len(self.class_names), len(probs)))
        }
        return CNNResult(predicted_class=pred, probabilities=prob_dict, confidence=float(probs[idx]))

    def predict_dbscan(self, features: CoralCsvFeatures) -> DBSCANResult:
        if self._tabular is None:
            raise RuntimeError("Tabular bundle not loaded (features.pkl, scaler.pkl, X_train.npy)")

        x_s = self._tabular.transform(features)
        cid = self._tabular.predict_cluster(x_s)
        detail = None
        if self._dbscan_info.get("params"):
            pr = self._dbscan_info["params"]
            detail = f"Training search params: eps={pr.get('eps'):.4f}, min_samples={pr.get('min_samples')}"
        return DBSCANResult(cluster_id=cid, cluster_label=f"cluster_{cid}", detail=detail)

    def predict_fusion(self, features: CoralCsvFeatures, cluster_id: int) -> FusionResult:
        if self._fusion is None or self._tabular is None:
            raise RuntimeError("Fusion ANN or tabular bundle not loaded")

        x_s = self._tabular.transform(features).astype(np.float32)
        tab_t = torch.from_numpy(x_s).to(self.device)
        oh = self._tabular.one_hot_for_cluster(cluster_id)
        cl_t = torch.from_numpy(oh).to(self.device)

        with torch.no_grad():
            pred = self._fusion(tab_t, cl_t).cpu().numpy().reshape(-1)

        raw = float(pred[0])
        clipped = float(np.clip(raw, 0.0, 100.0))
        return FusionResult(predicted_percent_bleaching=clipped, raw_model_output=raw)

    def generate_llm_report(
        self,
        cnn: CNNResult | None,
        dbscan: DBSCANResult | None,
        fusion: FusionResult | None,
        features: CoralCsvFeatures,
    ) -> str:
        key = self.settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if not key.strip():
            return (
                "LLM report skipped: set GROQ_API_KEY in `.env`. "
                "Raw model outputs are still available in the JSON fields."
            )

        client = Groq(api_key=key)
        payload = {
            "coral_csv_style_features": features.model_dump(),
            "cnn": cnn.model_dump() if cnn else None,
            "dbscan": dbscan.model_dump() if dbscan else None,
            "fusion": fusion.model_dump() if fusion else None,
        }

        system = (
            "You are a senior marine ecologist and coral-reef physiologist writing for "
            "managers and scientists. Be precise, cautious, and evidence-aligned. "
            "Do not invent site-specific facts not implied by the inputs."
        )
        user = f"""Using the structured model outputs below, write a concise professional assessment for CoralGuard AI.

Structure (use markdown headings):
## Risk level
Assign Low / Moderate / High / Severe with one-sentence justification.

## Scientific interpretation
Explain how the image CNN class, the unsupervised oceanographic regime (DBSCAN cluster), and the fused bleaching-risk estimate relate — and where uncertainty remains.

## Conservation & management recommendations
3–5 bullet points actionable for reef managers.

## SDG 14 (Life Below Water)
One short paragraph tying the assessment to UN Sustainable Development Goal 14 and concrete stewardship actions.

Data (JSON):
{json.dumps(payload, indent=2)}
"""

        chat = client.chat.completions.create(
            model=self.settings.groq_model,
            temperature=0.35,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = chat.choices[0].message.content or ""
        return text.strip()


_engine: CoralGuardEngine | None = None


def get_engine() -> CoralGuardEngine:
    global _engine
    if _engine is None:
        _engine = CoralGuardEngine()
        _engine.load_all()
    return _engine


def reset_engine_cache() -> None:
    global _engine
    _engine = None
    get_settings.cache_clear()
