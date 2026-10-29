"""
Train and Export Final Production Web Model.
Protocol:
- Features: eGeMAPSv02 (88 dims) per-utterance functionals
- Augmentation: Web-microphone simulation (gain, colored noise, early reverb, bandpass EQ, soft clipping)
- Scaler: StandardScaler fitted on pooled training data
- Classifier: LogisticRegression(C=0.1, class_weight='balanced')
- Calibrator: Post-hoc Temperature Scaling
- Artifact saved: week6_webmodel/models/final_web_model.joblib
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
MODELS_DIR = WEEK6_DIR / "models"
CACHE_DIR = WEEK6_DIR / "cache"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEEK6_DIR))

from data_manager import get_unified_metadata, get_p1_folds, get_split_indices_for_p1
from inference import EMOTION_LABELS


class TemperatureCalibrator:
    def __init__(self, temperature=1.0):
        self.temperature = float(temperature)
        
    def calibrate(self, logits):
        scaled = logits / max(self.temperature, 1e-3)
        exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=-1, keepdims=True)


def train_and_export():
    df = get_unified_metadata()
    # Use clean pooled evaluation corpora: EmoDB, CAFE, AESDD
    pool_mask = df["corpus"].isin(["emodb", "cafe", "aesdd"])
    pool_indices = np.where(pool_mask)[0]
    
    y_pool = df.iloc[pool_indices]["target_label"].values
    
    X_clean = np.load(CACHE_DIR / "egemaps_88_features.npy")[pool_indices]
    X_micsim = np.load(CACHE_DIR / "egemaps_88_micsim.npy")[pool_indices]
    
    # Augmented dataset for web-microphone invariance: clean + mic-simulated audio
    X_train_aug = np.vstack([X_clean, X_micsim])
    y_train_aug = np.concatenate([y_pool, y_pool])
    
    print(f"[Export] Fitting StandardScaler on {len(X_train_aug)} training clips (88 dims)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    
    print(f"[Export] Training LogisticRegression (C=0.1, balanced) on pooled corpora...")
    clf = LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train_scaled, y_train_aug)
    
    # Compute temperature from C5 calibration results (mean T = 1.25)
    # Mean temperature from validation across folds was 1.247
    calibrator = TemperatureCalibrator(temperature=1.247)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = MODELS_DIR / "final_web_model.joblib"
    
    artifact = {
        "model_name": "EmotionSER_eGeMAPS_v1",
        "feature_set": "eGeMAPSv02_88",
        "sample_rate": 16000,
        "classes": EMOTION_LABELS,
        "scaler": scaler,
        "classifier": clf,
        "temperature": calibrator.temperature,
        "training_n": len(X_train_aug),
        "robustness": "web-microphone augmented"
    }
    
    joblib.dump(artifact, export_path)
    print(f"[Export] Saved final web model artifact to {export_path}")
    print(f"Artifact size: {export_path.stat().st_size / 1024:.2f} KB")
    
    # Verify inference using inference.py
    from inference import predict_emotion
    synth_wave = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000, dtype=np.float32))
    pred = predict_emotion(synth_wave)
    print("\n[Export Verification] Test prediction on synthetic waveform:")
    print(pred)
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-4, "Probabilities do not sum to 1.0!"
    print("Verification PASSED: Probabilities sum to 1.0, latency:", pred["inference_latency_ms"], "ms")


if __name__ == "__main__":
    train_and_export()
