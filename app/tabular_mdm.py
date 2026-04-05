"""
Tabular preprocessing aligned with Phase 3 / 5 / 6 of `mdm.py`:
  - Base `coral.csv` columns + engineered Thermal_Stress, Light_Index, SST_Total
  - Subset to MI-selected columns from `features.pkl`
  - `RobustScaler` from `scaler.pkl`
  - DBSCAN train labels + `X_train.npy` for kNN assignment of new points
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from app.schemas import CoralCsvFeatures
from app.utils import get_logger

log = get_logger()


def make_onehot(labels: np.ndarray, n_cols: int) -> np.ndarray:
    """Same logic as `make_onehot` in `mdm.py` Phase 6."""
    oh = np.zeros((len(labels), n_cols), dtype=np.float32)
    for i, lab in enumerate(labels):
        l = int(lab)
        if l == -1:
            oh[i, -1] = 1.0
        elif l < n_cols - 1:
            oh[i, l] = 1.0
    return oh


def _feature_bag(f: CoralCsvFeatures) -> dict[str, float]:
    depth_clip = max(float(f.depth_m), 0.1)
    thermal_stress = float(f.ssta) * float(f.tsa)
    light_index = 1.0 / (1.0 + float(f.turbidity) * depth_clip)
    sst_total = float(f.clim_sst) + float(f.ssta)
    return {
        "Latitude_Degrees": float(f.latitude_degrees),
        "Longitude_Degrees": float(f.longitude_degrees),
        "Depth_m": float(f.depth_m),
        "Turbidity": float(f.turbidity),
        "Cyclone_Frequency": float(f.cyclone_frequency),
        "ClimSST": float(f.clim_sst),
        "SSTA": float(f.ssta),
        "TSA": float(f.tsa),
        "Percent_Cover": float(f.percent_cover),
        "Date_Year": float(f.date_year),
        "Thermal_Stress": float(thermal_stress),
        "Light_Index": float(light_index),
        "SST_Total": float(sst_total),
    }


def features_to_selected_matrix(f: CoralCsvFeatures, selected_columns: list[str]) -> np.ndarray:
    bag = _feature_bag(f)
    missing = [c for c in selected_columns if c not in bag]
    if missing:
        raise ValueError(f"Selected feature(s) not in engineered bag: {missing}")
    return np.array([[bag[c] for c in selected_columns]], dtype=np.float64)


@dataclass
class MdmTabularArtifacts:
    scaler: Any
    selected_columns: list[str]
    x_train: np.ndarray
    train_cluster_labels: np.ndarray
    knn: KNeighborsClassifier
    n_tab: int
    n_clust_cols: int

    def transform(self, f: CoralCsvFeatures) -> np.ndarray:
        x = features_to_selected_matrix(f, self.selected_columns)
        return self.scaler.transform(x)

    def predict_cluster(self, x_scaled: np.ndarray) -> int:
        return int(self.knn.predict(x_scaled)[0])

    def one_hot_for_cluster(self, cluster_id: int) -> np.ndarray:
        return make_onehot(np.array([cluster_id]), self.n_clust_cols)


def load_mdm_tabular_bundle(
    tabular_dir: Path,
    dbscan_pickle_path: Path,
) -> MdmTabularArtifacts | None:
    """
    Load scaler.pkl, features.pkl, X_train.npy and align with dbscan_model.pkl['labels'].
    Returns None if required files are missing.
    """
    tabular_dir = Path(tabular_dir)
    feat_path = tabular_dir / "features.pkl"
    sc_path = tabular_dir / "scaler.pkl"
    x_path = tabular_dir / "X_train.npy"

    for p in (feat_path, sc_path, x_path, dbscan_pickle_path):
        if not p.is_file():
            log.warning("MDM tabular/DBSCAN artifact missing: %s", p)
            return None

    with open(feat_path, "rb") as fp:
        selected = pickle.load(fp)
    if not isinstance(selected, list) or not selected:
        log.error("features.pkl must contain a non-empty list of column names")
        return None

    with open(sc_path, "rb") as fp:
        scaler = pickle.load(fp)

    x_train = np.load(x_path)
    if x_train.ndim != 2:
        log.error("X_train.npy must be 2-D")
        return None

    n_features = int(getattr(scaler, "n_features_in_", x_train.shape[1]))
    if x_train.shape[1] != n_features:
        log.warning(
            "X_train.npy columns (%s) != scaler.n_features_in_ (%s)",
            x_train.shape[1],
            n_features,
        )

    with open(dbscan_pickle_path, "rb") as fp:
        db = pickle.load(fp)

    if not isinstance(db, dict) or "labels" not in db or "n_clusters" not in db:
        log.error("dbscan_model.pkl must be a dict with 'labels' and 'n_clusters' (Colab export).")
        return None

    labels = np.asarray(db["labels"])
    if labels.shape[0] != x_train.shape[0]:
        log.error(
            "DBSCAN labels length %s != X_train rows %s — cannot assign clusters.",
            labels.shape[0],
            x_train.shape[0],
        )
        return None

    n_clusters = int(db["n_clusters"])
    n_clust_cols = n_clusters + 1  # +1 noise / anomaly column (mdm.py)

    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    knn.fit(x_train, labels)

    log.info(
        "MDM tabular bundle ready: %s features, %s train rows, %s cluster one-hot cols",
        len(selected),
        x_train.shape[0],
        n_clust_cols,
    )

    return MdmTabularArtifacts(
        scaler=scaler,
        selected_columns=list(selected),
        x_train=x_train,
        train_cluster_labels=labels,
        knn=knn,
        n_tab=x_train.shape[1],
        n_clust_cols=n_clust_cols,
    )
