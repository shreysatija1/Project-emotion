"""
C2: Pretrained Speech Embedding Extractor.
Supports frozen transformer encoders from HuggingFace:
- facebook/wav2vec2-base
- facebook/hubert-base-ls960
- microsoft/wavlm-base-plus

Extracts multi-layer hidden states (layers 3, 6, 9, 12):
- Mean pooling (768 dims per layer)
- Mean + Std pooling (1,536 dims per layer)
Accelerated via PyTorch CUDA (NVIDIA RTX 3050 Laptop GPU) in half-precision (fp16).
Caches all extracted embeddings to disk (.npz) so extraction is performed ONCE.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModel

ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK6_DIR = ROOT_DIR / "week6_webmodel"
CACHE_DIR = WEEK6_DIR / "cache"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(WEEK6_DIR))

from data_manager import get_unified_metadata, load_clip, SR

EXTRACT_LAYERS = [3, 6, 9, 12]  # Multi-layer representation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_embeddings_for_dataset(model_name, batch_size=16):
    """
    Extract multi-layer embeddings for all clips in the unified dataset
    using a frozen pretrained speech encoder.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. 'facebook/wav2vec2-base').
    batch_size : int
        Inference batch size (default 16).
        
    Returns
    -------
    dict:
        {
            f"layer_{l}_mean": np.ndarray (N, 768),
            f"layer_{l}_mean_std": np.ndarray (N, 1536)
        }
    """
    model_slug = model_name.replace("/", "_").replace("-", "_")
    cache_path = CACHE_DIR / f"embeddings_{model_slug}.npz"
    
    if cache_path.exists():
        print(f"[C2] Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}
        
    print(f"\n[C2] Extracting embeddings using {model_name} on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    df = get_unified_metadata()
    n_clips = len(df)
    
    # Pre-allocate containers
    layer_means = {l: [] for l in EXTRACT_LAYERS}
    layer_mean_stds = {l: [] for l in EXTRACT_LAYERS}
    
    # Process in batches
    for i in range(0, n_clips, batch_size):
        batch_rows = df.iloc[i : i + batch_size]
        waveforms = [load_clip(r["audio_path"], target_sr=SR, max_duration=10.0) for _, r in batch_rows.iterrows()]
        
        inputs = processor(
            waveforms,
            sampling_rate=SR,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            if "attention_mask" in inputs:
                input_lengths = inputs["attention_mask"].sum(dim=1)
                try:
                    valid_lengths = model._get_feat_extract_output_lengths(input_lengths)
                except Exception:
                    valid_lengths = [outputs.hidden_states[0].shape[1]] * len(input_lengths)
            else:
                valid_lengths = [outputs.hidden_states[0].shape[1]] * len(waveforms)
                
            for l in EXTRACT_LAYERS:
                hidden = outputs.hidden_states[l]  # (B, T, D)
                b_means = []
                b_mean_stds = []
                for b_idx in range(hidden.shape[0]):
                    v_len = max(int(valid_lengths[b_idx]), 1)
                    v_hidden = hidden[b_idx, :v_len, :]
                    m_p = v_hidden.mean(dim=0)
                    s_p = v_hidden.std(dim=0)
                    b_means.append(m_p)
                    b_mean_stds.append(torch.cat([m_p, s_p], dim=-1))
                
                b_means_t = torch.stack(b_means)
                b_mean_stds_t = torch.stack(b_mean_stds)
                
                layer_means[l].append(b_means_t.cpu().float().numpy().astype(np.float32))
                layer_mean_stds[l].append(b_mean_stds_t.cpu().float().numpy().astype(np.float32))
                
        if (i + batch_size) % 200 < batch_size or (i + batch_size) >= n_clips:
            print(f"  Processed {min(i + batch_size, n_clips)}/{n_clips} clips...")
            
    out_dict = {}
    for l in EXTRACT_LAYERS:
        out_dict[f"layer_{l}_mean"] = np.vstack(layer_means[l])
        out_dict[f"layer_{l}_mean_std"] = np.vstack(layer_mean_stds[l])
        
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out_dict)
    print(f"[C2] Saved cached embeddings to {cache_path}")
    return out_dict


if __name__ == "__main__":
    for m in ["facebook/wav2vec2-base", "facebook/hubert-base-ls960", "microsoft/wavlm-base-plus"]:
        try:
            feats = extract_embeddings_for_dataset(m, batch_size=16)
            print(f"Success with {m}! Keys: {list(feats.keys())}")
        except Exception as e:
            print(f"Failed extracting {m}: {e}")
