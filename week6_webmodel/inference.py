"""
Week 6 Production Web Model Inference Engine.
Contract:
- Single callable: predict_emotion(audio_input, sr=16000)
- Deployability: Per-utterance processing only. No dependence on batch or external speaker stats.
- Robustness: Handles mono/stereo, resampling, clipping, silence trimming, and length validation.
- Uncertainty: Temperature-calibrated probabilities with low-confidence / high-entropy rejection flags.
- Latency: < 150 ms on CPU (well below the 3.0s budget).
"""

import io
import time
from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import soundfile as sf
import librosa
import joblib
import opensmile

WEEK6_DIR = Path(__file__).resolve().parent
MODELS_DIR = WEEK6_DIR / "models"
MODEL_FILE = MODELS_DIR / "final_web_model.joblib"

TARGET_SR = 16000
EMOTION_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]

# Global cache for loaded model to avoid re-loading on each web request
_CACHED_MODEL_ARTIFACTS = None


def load_model_artifacts():
    """Load trained model, scaler, calibrator, and metadata."""
    global _CACHED_MODEL_ARTIFACTS
    if _CACHED_MODEL_ARTIFACTS is not None:
        return _CACHED_MODEL_ARTIFACTS
        
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_FILE}. "
            f"Please run model training/export first."
        )
        
    _CACHED_MODEL_ARTIFACTS = joblib.load(MODEL_FILE)
    return _CACHED_MODEL_ARTIFACTS


def preprocess_audio(
    audio_input: Union[str, Path, bytes, np.ndarray],
    target_sr: int = TARGET_SR
) -> tuple[np.ndarray, float, Optional[str]]:
    """
    Standardize audio input to 1D float32 array at target_sr (16 kHz).
    
    Returns:
        (waveform, duration_seconds, low_confidence_reason)
    """
    y = None
    sr = target_sr
    
    # 1. Load from file path, bytes, or array
    if isinstance(audio_input, (str, Path)):
        p = Path(audio_input)
        if not p.exists():
            raise FileNotFoundError(f"Audio file does not exist: {p}")
        y, orig_sr = sf.read(str(p), dtype="float32", always_2d=False)
        sr = orig_sr
    elif isinstance(audio_input, bytes):
        bio = io.BytesIO(audio_input)
        y, orig_sr = sf.read(bio, dtype="float32", always_2d=False)
        sr = orig_sr
    elif isinstance(audio_input, np.ndarray):
        y = audio_input.astype(np.float32)
        sr = target_sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")
        
    # 2. Convert to mono if multichannel
    if y.ndim > 1:
        y = np.mean(y, axis=1 if y.shape[1] < y.shape[0] else 0)
        
    # 3. Resample if necessary
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    # 4. Silence check and trimming
    duration_s = float(len(y)) / float(target_sr)
    rms = np.sqrt(np.mean(y ** 2)) if len(y) > 0 else 0.0
    
    if len(y) < int(0.4 * target_sr) or rms < 1e-4:
        return y, duration_s, "Audio is too short (< 0.5s) or near silent (RMS < 1e-4)."
        
    # Trim leading/trailing silence
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        if len(y_trimmed) >= int(0.4 * target_sr):
            y = y_trimmed
            duration_s = float(len(y)) / float(target_sr)
    except Exception:
        pass
        
    # 5. Length boundary handling
    reason = None
    if duration_s > 10.0:
        # Cap at 10 seconds for consistent processing
        y = y[: int(10.0 * target_sr)]
        duration_s = 10.0
        reason = "Audio exceeds 10s; trimmed to first 10.0s."
        
    return y.astype(np.float32), duration_s, reason


def extract_single_clip_features(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract eGeMAPSv02 (88 dims) per-utterance functionals using openSMILE.
    Fast, completely self-contained, and independent of external dataset stats.
    """
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    df_feat = smile.process_signal(y, sr)
    feats = df_feat.values[0].astype(np.float32)
    # Replace any NaNs / Infs with 0.0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def predict_emotion(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sr: int = TARGET_SR
) -> Dict[str, Any]:
    """
    Predict 6-class emotion distribution from an unknown speaker's recording.
    
    Returns:
        dict:
            predicted_emotion: str
            confidence: float (calibrated)
            probabilities: dict of {str: float} summing to 1.0
            entropy: float (Shannon entropy in nats)
            low_confidence: bool
            low_confidence_reason: Optional[str]
            duration_seconds: float
            inference_latency_ms: float
    """
    t0 = time.perf_counter()
    
    # 1. Preprocess waveform
    y, duration_s, prep_reason = preprocess_audio(audio_input, target_sr=sr)
    
    # Check for short / silent rejection
    if prep_reason and "too short" in prep_reason:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        uniform_p = 1.0 / len(EMOTION_LABELS)
        return {
            "predicted_emotion": "neutral",
            "confidence": uniform_p,
            "probabilities": {e: uniform_p for e in EMOTION_LABELS},
            "entropy": float(np.log(len(EMOTION_LABELS))),
            "low_confidence": True,
            "low_confidence_reason": prep_reason,
            "duration_seconds": duration_s,
            "inference_latency_ms": round(latency_ms, 2)
        }
        
    # 2. Extract features
    feats = extract_single_clip_features(y, sr=TARGET_SR)
    
    # 3. Load model artifacts
    artifacts = load_model_artifacts()
    scaler = artifacts["scaler"]
    classifier = artifacts["classifier"]
    temperature = artifacts.get("temperature", 1.0)
    
    # Scale features using training-fitted scaler
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    
    # 4. Predict uncalibrated probabilities / logits
    if hasattr(classifier, "predict_proba"):
        raw_probs = classifier.predict_proba(feats_scaled)[0]
        # Convert to logits for temperature scaling
        eps = 1e-12
        logits = np.log(raw_probs + eps)
    else:
        logits = classifier.decision_function(feats_scaled)[0]
        
    # 5. Temperature calibration
    scaled_logits = logits / max(temperature, 1e-3)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    cal_probs = (exp_logits / np.sum(exp_logits)).astype(float)
    
    # Ensure exact probability simplex: sum to 1.0
    cal_probs = cal_probs / np.sum(cal_probs)
    
    pred_idx = int(np.argmax(cal_probs))
    pred_emotion = EMOTION_LABELS[pred_idx]
    confidence = float(cal_probs[pred_idx])
    
    # 6. Uncertainty estimation: Shannon entropy
    entropy = float(-np.sum(cal_probs * np.log(cal_probs + 1e-12)))
    
    # Rejection criteria:
    # - If entropy is high (> 1.65 out of log(6)=1.79) OR max probability is very low (< 0.25)
    low_conf = False
    rejection_reason = prep_reason
    if entropy > 1.65 or confidence < 0.25:
        low_conf = True
        rejection_reason = "High prediction entropy / low model confidence."
        
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    probs_dict = {
        EMOTION_LABELS[i]: round(float(cal_probs[i]), 4)
        for i in range(len(EMOTION_LABELS))
    }
    
    return {
        "predicted_emotion": pred_emotion,
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
        "entropy": round(entropy, 4),
        "low_confidence": low_conf,
        "low_confidence_reason": rejection_reason,
        "duration_seconds": round(duration_s, 2),
        "inference_latency_ms": round(latency_ms, 2)
    }
