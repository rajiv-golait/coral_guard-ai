import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.preprocessing import RobustScaler

CLUSTER_NAMES: dict[int, str] = {
    -1: "Extreme Anomaly Zone",
    0: "Normal Ocean Baseline",
    1: "High Thermal Stress Zone",
    2: "Nutrient-Excess / Low-O2 Zone",
    3: "Turbid Coastal Zone",
}

CLASS_LABELS = ["Healthy", "Bleached", "Dead"]

# Raw slider features in training order for the 10-dim Keras tabular branch (RobustScaler)
KERAS_TABULAR_FEATURES = [
    "Latitude_Degrees",
    "Longitude_Degrees",
    "Depth_m",
    "Turbidity",
    "Cyclone_Frequency",
    "ClimSST",
    "SSTA",
    "TSA",
    "Percent_Cover",
    "Date_Year",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Module-level model state (loaded once at startup)
keras_model: Any = None
dbscan_model: Any = None
scaler: RobustScaler | None = None
feature_names: list[str] | None = None
neutral_tabular: np.ndarray | None = None
TABULAR_BLEND_ALPHA = float(os.getenv("TABULAR_BLEND_ALPHA", "0.12"))


def _resolve_path(env_key: str, default: str) -> Path:
    base = Path(__file__).resolve().parent
    raw = os.getenv(env_key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path


def load_models() -> None:
    """Load all ML artifacts once at application startup."""
    global keras_model, dbscan_model, scaler, feature_names, neutral_tabular

    import tensorflow as tf

    from custom_layers import get_custom_objects

    model_path = _resolve_path("MODEL_PATH", "./models/coralguard_fusion_best.keras")
    dbscan_path = _resolve_path("DBSCAN_PATH", "./models/dbscan_model.pkl")
    scaler_path = _resolve_path("SCALER_PATH", "./models/scaler.pkl")
    features_path = _resolve_path("FEATURES_PATH", "./models/features.pkl")

    for path, label in [
        (model_path, "coralguard_fusion_best.keras"),
        (dbscan_path, "dbscan_model.pkl"),
        (scaler_path, "scaler.pkl"),
        (features_path, "features.pkl"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {label} at {path}. Download model files into backend/models/"
            )

    print(f"[predict] Loading Keras model from {model_path}")
    keras_model = tf.keras.models.load_model(
        str(model_path),
        custom_objects=get_custom_objects(),
        compile=False,
        safe_mode=False,
    )

    print(f"[predict] Loading DBSCAN from {dbscan_path}")
    with open(dbscan_path, "rb") as f:
        dbscan_obj = pickle.load(f)
    if isinstance(dbscan_obj, dict):
        dbscan_model = dbscan_obj.get("model", dbscan_obj)
    else:
        dbscan_model = dbscan_obj

    print(f"[predict] Loading scaler from {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print(f"[predict] Loading feature names from {features_path}")
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    # Neutral tabular = training median in scaled space (lets image branch drive class)
    neutral_tabular = scaler.transform(
        scaler.center_.reshape(1, -1).astype(np.float32)
    ).astype(np.float32)
    print(f"[predict] Keras tabular features: {KERAS_TABULAR_FEATURES}")
    print(f"[predict] Tabular blend alpha (slider influence): {TABULAR_BLEND_ALPHA}")

    print("[predict] All models loaded successfully")


def models_loaded() -> bool:
    return (
        keras_model is not None
        and dbscan_model is not None
        and scaler is not None
        and feature_names is not None
        and neutral_tabular is not None
    )


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    result[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    result[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    result[:, :, 2] *= avg_gray / (avg_r + 1e-6)
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode and preprocess coral image per spec. Returns (1, 224, 224, 3) float32."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid or corrupt image file. Please upload a valid JPG or PNG.")

    img = gray_world_white_balance(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def _engineered_features(raw: dict[str, float]) -> dict[str, float]:
    thermal_stress = raw["SSTA"] * raw["TSA"]
    light_index = 1.0 / (1.0 + raw["Turbidity"] * max(raw["Depth_m"], 0.1))
    sst_total = raw["ClimSST"] + raw["SSTA"]
    return {
        **raw,
        "Thermal_Stress": thermal_stress,
        "Light_Index": light_index,
        "SST_Total": sst_total,
    }


def build_keras_tabular(raw: dict[str, float]) -> np.ndarray:
    """
    Build 10-dim scaled tabular input for the fusion model.

    Wrong feature order previously saturated softmax to 'Healthy'. We use the
    training median (neutral) as baseline and blend a small amount of scaled
    user sliders so the image branch dominates classification.
    """
    if scaler is None or neutral_tabular is None:
        raise RuntimeError("Models not loaded")

    user_arr = np.array(
        [[raw[name] for name in KERAS_TABULAR_FEATURES]], dtype=np.float32
    )
    user_scaled = scaler.transform(user_arr).astype(np.float32)
    user_scaled = np.clip(user_scaled, -2.5, 2.5)

    alpha = max(0.0, min(1.0, TABULAR_BLEND_ALPHA))
    if alpha <= 0.0:
        return neutral_tabular.copy()

    blended = (1.0 - alpha) * neutral_tabular + alpha * user_scaled
    return blended.astype(np.float32)


def build_tabular_features(
    raw: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return keras tabular (1,10), cluster vector (1,18), and engineered metrics."""
    engineered = _engineered_features(raw)
    thermal_stress = engineered["Thermal_Stress"]
    light_index = engineered["Light_Index"]
    sst_total = engineered["SST_Total"]

    if feature_names is None or scaler is None:
        raise RuntimeError("Models not loaded")

    keras_scaled = build_keras_tabular(raw)

    # 18-dim vector for DBSCAN nearest-core assignment (all 13 engineered + extras)
    keras_n = len(KERAS_TABULAR_FEATURES)
    remaining = [engineered[name] for name in feature_names[keras_n:]]
    extras = [
        raw["Depth_m"],
        raw["Percent_Cover"],
        raw["Cyclone_Frequency"],
        raw["SSTA"],
        raw["TSA"],
    ]
    cluster_vec = np.hstack([keras_scaled, np.array([remaining + extras], dtype=np.float32)])
    cluster_vec = cluster_vec.astype(np.float64)

    return keras_scaled, cluster_vec, thermal_stress, light_index, sst_total


def _heuristic_cluster(engineered: dict[str, float]) -> int:
    if engineered["Turbidity"] >= 12:
        return 3
    if engineered["Thermal_Stress"] >= 2.5 or engineered["SSTA"] >= 2.0:
        return 1
    if engineered["Turbidity"] >= 8 and engineered["Depth_m"] <= 5:
        return 2
    if abs(engineered["SSTA"]) >= 4 or engineered["Thermal_Stress"] <= -2:
        return -1
    return 0


def run_dbscan_cluster(
    cluster_vec: np.ndarray, engineered: dict[str, float]
) -> tuple[int, str, bool]:
    if dbscan_model is None:
        raise RuntimeError("DBSCAN model not loaded")

    from sklearn.metrics import pairwise_distances

    cluster_id: int

    if isinstance(dbscan_model, dict):
        bundle = dbscan_model
        model = bundle.get("model")
        labels = bundle.get("labels")
        if "X_train" in bundle and labels is not None:
            dists = pairwise_distances(cluster_vec, bundle["X_train"]).flatten()
            cluster_id = int(labels[int(np.argmin(dists))])
        elif model is not None and hasattr(model, "fit_predict"):
            cluster_id = int(model.fit_predict(cluster_vec)[0])
        else:
            cluster_id = _heuristic_cluster(engineered)
    elif hasattr(dbscan_model, "components_") and dbscan_model.components_.shape[0] > 0:
        dists = pairwise_distances(cluster_vec, dbscan_model.components_).flatten()
        idx = int(np.argmin(dists))
        min_dist = float(dists[idx])
        if hasattr(dbscan_model, "core_sample_indices_"):
            label_idx = int(dbscan_model.core_sample_indices_[idx])
            cluster_id = int(dbscan_model.labels_[label_idx])
        else:
            cluster_id = int(dbscan_model.labels_[idx])
        # If far from training manifold, use rule-based zone mapping
        if min_dist > 50.0:
            cluster_id = _heuristic_cluster(engineered)
    else:
        cluster_id = _heuristic_cluster(engineered)

    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    is_anomaly = cluster_id == -1
    return cluster_id, cluster_name, is_anomaly


def compute_risk_level(
    health_class: str, confidence: float, is_anomaly: bool
) -> str:
    if health_class == "Dead" and confidence > 0.60:
        return "CRITICAL"
    if health_class == "Bleached" and is_anomaly and confidence > 0.75:
        return "CRITICAL"
    if health_class == "Bleached" and confidence > 0.75:
        return "HIGH"
    if health_class == "Bleached" and confidence < 0.75:
        return "MODERATE"
    return "LOW"


def predict(image_bytes: bytes, params: dict[str, float]) -> dict[str, Any]:
    if not models_loaded():
        raise RuntimeError("Models not loaded. Server may still be starting.")

    print("[predict] Preprocessing image...")
    image_tensor = preprocess_image(image_bytes)

    print("[predict] Building tabular features...")
    tabular, cluster_vec, thermal_stress, light_index, sst_total = build_tabular_features(
        params
    )
    engineered = _engineered_features(params)

    print("[predict] Running fusion model inference...")
    probs = keras_model.predict([image_tensor, tabular], verbose=0)[0]
    probs = probs.tolist()

    pred_idx = int(np.argmax(probs))
    health_class = CLASS_LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    print("[predict] Running DBSCAN clustering...")
    cluster_id, cluster_name, is_anomaly = run_dbscan_cluster(cluster_vec, engineered)

    risk_level = compute_risk_level(health_class, confidence, is_anomaly)

    result = {
        "health_class": health_class,
        "confidence": confidence,
        "probabilities": {
            "Healthy": float(probs[0]),
            "Bleached": float(probs[1]),
            "Dead": float(probs[2]),
        },
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "is_anomaly": is_anomaly,
        "risk_level": risk_level,
        "thermal_stress": float(thermal_stress),
        "light_index": float(light_index),
        "sst_total": float(sst_total),
    }
    print(
        f"[predict] Result: {health_class} ({confidence:.2%}), risk={risk_level}, "
        f"probs={[round(p, 4) for p in probs]}"
    )
    return result
