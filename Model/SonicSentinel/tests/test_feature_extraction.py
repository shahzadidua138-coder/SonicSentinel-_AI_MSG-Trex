# ============================================================
# SonicSentinel AI - Feature Extraction Unit Tests
# ============================================================
import unittest
import numpy as np
from feature_extraction.extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()
        # Create 1s synthetic noise signal
        self.sr = 22050
        self.audio = np.random.normal(0, 0.1, self.sr)

    def test_extract_all_features(self):
        feats = self.extractor.extract_all_features(self.audio, self.sr)

        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(len(feats), 373)

    def test_feature_vector_dimension(self):
        vector = self.extractor.extract_all_features(self.audio, self.sr)

        self.assertEqual(len(vector), 373)
        self.assertIsInstance(vector, np.ndarray)


if __name__ == "__main__":
    unittest.main()
