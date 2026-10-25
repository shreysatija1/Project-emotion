"""
Unit and integration tests for src/augment.py.
Can be executed via:
    python src/test_augment.py
or:
    python -m unittest discover -s src -p "test_*.py"
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root and src dir to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import COEFF_LIST, SR, FEAT_DIR, PROC_DIR
from augment import (
    extract_features,
    build_augmented_sets,
    load_saved_augmented_data,
    test_no_source_clip_crosses_splits_version_a,
    test_no_nans,
    test_correct_shapes,
    test_set2_features_equal_set1_features,
)


class TestAugmentationUnit(unittest.TestCase):
    """Unit tests for waveform augmentation and feature extraction."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        # 1-second synthetic audio waveform at 16 kHz
        self.wave = self.rng.standard_normal(SR).astype(np.float32)

    def test_build_augmented_sets_structure_and_equality(self):
        """Test the 5 sets generated per clip."""
        sets = build_augmented_sets(self.wave, rng=np.random.default_rng(42))
        self.assertEqual(len(sets), 5)

        names = [s["set_name"] for s in sets]
        self.assertEqual(names, ["set1", "set2", "set3", "set4", "set5"])

        levels = [s["noise_level"] for s in sets]
        self.assertEqual(levels, [0.0, 0.0, 0.01, 0.025, 0.05])

        s1 = sets[0]["wave"]
        s2 = sets[1]["wave"]
        s3 = sets[2]["wave"]
        s4 = sets[3]["wave"]
        s5 = sets[4]["wave"]

        # set2 must be an exact zero-noise copy of set1
        self.assertTrue(np.array_equal(s1, s2), "Set2 must be bitwise identical to Set1")
        self.assertTrue(np.array_equal(s1, self.wave), "Set1 must match input waveform")

        # Check noise variance roughly matches level^2
        noise3 = s3 - s2
        noise4 = s4 - s2
        noise5 = s5 - s1

        self.assertAlmostEqual(np.std(noise3), 0.01, delta=0.003)
        self.assertAlmostEqual(np.std(noise4), 0.025, delta=0.005)
        self.assertAlmostEqual(np.std(noise5), 0.05, delta=0.01)

    def test_extract_features_shapes_and_deterministic_equality(self):
        """Test feature extraction on clean and copy waveforms."""
        s1 = self.wave.copy()
        s2 = self.wave.copy()

        for n in COEFF_LIST:
            mfcc_1 = extract_features(s1, sr=SR, feature_type="mfcc", n_coeff=n)
            mfcc_2 = extract_features(s2, sr=SR, feature_type="mfcc", n_coeff=n)
            self.assertEqual(mfcc_1.shape, (2 * n,))
            self.assertTrue(np.allclose(mfcc_1, mfcc_2), f"Set1 and Set2 MFCC mismatch at n={n}")
            self.assertTrue(np.isfinite(mfcc_1).all())

            mfmc_1 = extract_features(s1, sr=SR, feature_type="mfmc", n_coeff=n)
            mfmc_2 = extract_features(s2, sr=SR, feature_type="mfmc", n_coeff=n)
            self.assertEqual(mfmc_1.shape, (2 * n,))
            self.assertTrue(np.allclose(mfmc_1, mfmc_2), f"Set1 and Set2 MFMC mismatch at n={n}")
            self.assertTrue(np.isfinite(mfmc_1).all())


class TestSavedArtifacts(unittest.TestCase):
    """Integration tests verifying generated .npy, .csv, and split constraints."""

    @classmethod
    def setUpClass(cls):
        cls.feats_a, cls.labels_a, cls.meta_a = load_saved_augmented_data("version_a")
        cls.feats_b, cls.labels_b, cls.meta_b = load_saved_augmented_data("version_b")

    def test_version_a_no_source_clip_crosses_splits(self):
        """Ensure no source clip in Version A crosses splits."""
        test_no_source_clip_crosses_splits_version_a(self.meta_a)

    def test_no_nans_in_saved_artifacts(self):
        """Ensure no NaN or infinite values exist in any arrays or tables."""
        test_no_nans(self.feats_a, self.labels_a, self.meta_a,
                     self.feats_b, self.labels_b, self.meta_b)

    def test_correct_shapes(self):
        """Ensure all array dimensions and metadata rows match specification."""
        test_correct_shapes(self.meta_a, self.feats_a, self.labels_a,
                            self.meta_b, self.feats_b, self.labels_b)

    def test_set2_features_equal_set1_features_version_a(self):
        """Ensure set2 features equal set1 features across Version A."""
        test_set2_features_equal_set1_features(self.meta_a, self.feats_a, name="Version A")

    def test_set2_features_equal_set1_features_version_b(self):
        """Ensure set2 features equal set1 features across Version B."""
        test_set2_features_equal_set1_features(self.meta_b, self.feats_b, name="Version B")

    def test_version_b_stratified_split_ratios(self):
        """Verify that Version B maintains a 60/20/20 split across emotions."""
        counts = self.meta_b["split"].value_counts()
        self.assertEqual(counts["train"], 1605)
        self.assertEqual(counts["val"], 535)
        self.assertEqual(counts["test"], 535)

        # Check label stratification
        for split in ["train", "val", "test"]:
            sub = self.meta_b[self.meta_b["split"] == split]
            self.assertEqual(len(sub["label"].unique()), 7, "All 7 emotions must be present in each split")


if __name__ == "__main__":
    unittest.main()
