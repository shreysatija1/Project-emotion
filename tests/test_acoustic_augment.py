import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from acoustic_augment import augment_pitch_shift, augment_time_stretch, extract_spec_augment_features

def test_acoustic_augmentations():
    sr = 16000
    y = np.random.randn(sr * 2).astype(np.float32)
    
    # Pitch shift
    p_copies = augment_pitch_shift(y, sr)
    assert len(p_copies) == 5, f"Expected 5 copies, got {len(p_copies)}"
    for idx, c in enumerate(p_copies):
        assert not np.isnan(c).any(), f"NaN in pitch shift copy {idx}"
        assert c.dtype == np.float32, f"Pitch shift dtype: {c.dtype}"
    print("test_pitch_shift: PASSED")
    
    # Time stretch
    t_copies = augment_time_stretch(y)
    assert len(t_copies) == 5, f"Expected 5 copies, got {len(t_copies)}"
    for idx, c in enumerate(t_copies):
        assert not np.isnan(c).any(), f"NaN in time stretch copy {idx}"
        assert c.dtype == np.float32, f"Time stretch dtype: {c.dtype}"
    print("test_time_stretch: PASSED")
    
    # SpecAugment
    rng = np.random.default_rng(42)
    for feat_kind in ["mfcc", "mfmc"]:
        for n_coeff in [24, 30]:
            s_copies = extract_spec_augment_features(y, sr, feat_kind, n_coeff, rng, n_copies=5)
            assert len(s_copies) == 5, f"Expected 5 copies, got {len(s_copies)}"
            for idx, c in enumerate(s_copies):
                assert c.shape == (2 * n_coeff,), f"Expected shape ({2 * n_coeff},), got {c.shape}"
                assert not np.isnan(c).any(), f"NaN in spec augment {feat_kind} {n_coeff} copy {idx}"
                assert not np.isinf(c).any(), f"Inf in spec augment {feat_kind} {n_coeff} copy {idx}"
    print("test_spec_augment: PASSED")
    print("ALL ACOUSTIC AUGMENTATION TESTS PASSED!")

if __name__ == "__main__":
    test_acoustic_augmentations()
