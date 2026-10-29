"""
Week 6 Test Suite: Verification of Zero Leakage, Inference Engine, and API.
Tests:
1. Zero speaker leakage across P1 folds and P2 LOCO splits.
2. Training-only constraint for MESD (zero MESD in val/test).
3. Inference engine contract: probability simplex, determinism, low-confidence handling.
4. FastAPI endpoint health and prediction schemas.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEEK6_DIR))

from data_manager import (
    get_unified_metadata, get_p1_folds, get_p2_folds,
    get_split_indices_for_p1, get_split_indices_for_p2
)


def test_unified_metadata_integrity():
    """Verify unified metadata table properties."""
    df = get_unified_metadata()
    assert len(df) == 2608, f"Expected 2608 clips across 4 corpora, got {len(df)}"
    assert set(df["corpus"].unique()) == {"emodb", "cafe", "aesdd", "mesd"}
    
    # Check 6 pooled target classes
    expected_emotions = {"anger", "disgust", "fear", "happiness", "neutral", "sadness"}
    assert set(df["emotion"].unique()) == expected_emotions
    
    # Check class indices 0..5
    assert set(df["target_label"].unique()) == {0, 1, 2, 3, 4, 5}


def test_p1_zero_speaker_leakage():
    """Verify strict speaker disjointness in Protocol P1 folds."""
    df = get_unified_metadata()
    folds = get_p1_folds()
    assert len(folds) == 5, f"Expected 5 folds for P1, got {len(folds)}"
    
    for fold in folds:
        f_idx = fold["fold"]
        tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=False)
        
        tr_spks = set(df.iloc[tr_idx]["full_speaker_id"].unique())
        va_spks = set(df.iloc[va_idx]["full_speaker_id"].unique())
        te_spks = set(df.iloc[te_idx]["full_speaker_id"].unique())
        
        # Pairwise speaker disjointness
        assert len(tr_spks & va_spks) == 0, f"Fold {f_idx}: train/val speaker overlap: {tr_spks & va_spks}"
        assert len(tr_spks & te_spks) == 0, f"Fold {f_idx}: train/test speaker overlap: {tr_spks & te_spks}"
        assert len(va_spks & te_spks) == 0, f"Fold {f_idx}: val/test speaker overlap: {va_spks & te_spks}"
        
        # Index disjointness
        assert len(set(tr_idx) & set(va_idx)) == 0
        assert len(set(tr_idx) & set(te_idx)) == 0
        assert len(set(va_idx) & set(te_idx)) == 0


def test_p2_leave_one_corpus_out_integrity():
    """Verify Protocol P2 Leave-One-Corpus-Out disjointness."""
    df = get_unified_metadata()
    p2_folds = get_p2_folds()
    assert len(p2_folds) == 3, f"Expected 3 LOCO folds, got {len(p2_folds)}"
    
    for loco in p2_folds:
        f_name = loco["fold_name"]
        test_corpus = loco["test_corpus"]
        tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=False)
        
        # Test corpus must be 100% held-out
        assert (df.iloc[te_idx]["corpus"] == test_corpus).all()
        assert (df.iloc[tr_idx]["corpus"] != test_corpus).all()
        assert (df.iloc[va_idx]["corpus"] != test_corpus).all()
        
        # Speaker disjointness between train and val
        tr_spks = set(df.iloc[tr_idx]["full_speaker_id"].unique())
        va_spks = set(df.iloc[va_idx]["full_speaker_id"].unique())
        assert len(tr_spks & va_spks) == 0, f"LOCO {f_name}: train/val speaker overlap: {tr_spks & va_spks}"


def test_mesd_never_in_val_or_test():
    """Verify MESD is strictly restricted to training sets when enabled."""
    df = get_unified_metadata()
    p1_folds = get_p1_folds()
    p2_folds = get_p2_folds()
    
    for fold in p1_folds:
        tr_idx, va_idx, te_idx = get_split_indices_for_p1(df, fold, include_mesd_in_train=True)
        assert not (df.iloc[va_idx]["corpus"] == "mesd").any(), "MESD leaked into P1 validation set!"
        assert not (df.iloc[te_idx]["corpus"] == "mesd").any(), "MESD leaked into P1 test set!"
        assert (df.iloc[tr_idx]["corpus"] == "mesd").sum() == 862, "All MESD clips should be in train!"
        
    for loco in p2_folds:
        tr_idx, va_idx, te_idx = get_split_indices_for_p2(df, loco, include_mesd_in_train=True)
        assert not (df.iloc[va_idx]["corpus"] == "mesd").any(), "MESD leaked into P2 validation set!"
        assert not (df.iloc[te_idx]["corpus"] == "mesd").any(), "MESD leaked into P2 test set!"
        assert (df.iloc[tr_idx]["corpus"] == "mesd").sum() == 862, "All MESD clips should be in train!"


def test_mic_simulation_properties():
    """Verify microphone simulation produces valid perturbed audio signals."""
    from mic_simulation import simulate_web_microphone
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    y_clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    y_sim = simulate_web_microphone(y_clean, sr=16000)
    assert len(y_sim) == len(y_clean)
    assert np.isfinite(y_sim).all()
    assert not np.allclose(y_sim, y_clean), "Simulated audio should differ from clean audio"


def test_preprocess_short_audio_rejection():
    """Verify audio < 0.5s or near silent is flagged as low confidence."""
    from inference import preprocess_audio
    # 0.2 second chirp (too short)
    y_short = np.zeros(int(0.2 * 16000), dtype=np.float32)
    _, dur, reason = preprocess_audio(y_short, target_sr=16000)
    assert reason is not None
    assert "too short" in reason


def test_egemaps_feature_shape():
    """Verify single-clip eGeMAPS functional extraction outputs 88 valid features."""
    from inference import extract_single_clip_features
    t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
    y_tone = 0.4 * np.sin(2 * np.pi * 300 * t)
    feats = extract_single_clip_features(y_tone, sr=16000)
    assert feats.shape == (88,), f"Expected 88 dims, got {feats.shape}"
    assert np.isfinite(feats).all()


def test_end_to_end_inference():
    """Verify predict_emotion returns valid schema, probability simplex, and determinism."""
    from inference import predict_emotion, EMOTION_LABELS
    t = np.linspace(0, 2.0, 32000, endpoint=False, dtype=np.float32)
    y_test = 0.5 * np.sin(2 * np.pi * 500 * t)
    
    res1 = predict_emotion(y_test)
    res2 = predict_emotion(y_test)
    
    assert res1["predicted_emotion"] in EMOTION_LABELS
    assert 0.0 <= res1["confidence"] <= 1.0
    assert abs(sum(res1["probabilities"].values()) - 1.0) < 1e-4
    assert res1["inference_latency_ms"] < 3000.0  # Deployability constraint (<3s)
    
    # Determinism check
    assert res1["predicted_emotion"] == res2["predicted_emotion"]
    assert np.isclose(res1["confidence"], res2["confidence"], atol=1e-5)


def test_fastapi_endpoints():
    """Verify FastAPI service health and prediction endpoints."""
    from fastapi.testclient import TestClient
    from app import app
    import io
    import soundfile as sf
    
    client = TestClient(app)
    
    # 1. Health check
    r_health = client.get("/health")
    assert r_health.status_code == 200
    data_health = r_health.json()
    assert data_health["status"] == "ok"
    assert "classes" in data_health
    
    # 2. Prediction endpoint with synthetic wav
    t = np.linspace(0, 1.5, 24000, endpoint=False, dtype=np.float32)
    y_wave = 0.4 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, y_wave, 16000, format="WAV")
    buf.seek(0)
    
    r_pred = client.post(
        "/predict",
        files={"file": ("test.wav", buf, "audio/wav")}
    )
    assert r_pred.status_code == 200
    data_pred = r_pred.json()
    assert "predicted_emotion" in data_pred
    assert "probabilities" in data_pred
    assert abs(sum(data_pred["probabilities"].values()) - 1.0) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
