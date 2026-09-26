# ============================================================
# SonicSentinel AI - Audio Preprocessing Unit Tests
# ============================================================
import os
import unittest
import numpy as np
import soundfile as sf
from audio_preprocessing.preprocessor import AudioPreprocessor


class TestAudioPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = AudioPreprocessor()
        self.test_filename = "temp_test_audio.wav"
        # Generate 1 second synthetic sine wave at 44100Hz stereo
        sr = 44100
        t = np.linspace(0, 1, sr, False)
        sig = 0.5 * np.sin(2 * np.pi * 440 * t)
        stereo_sig = np.vstack((sig, sig)).T
        sf.write(self.test_filename, stereo_sig, sr)

    def tearDown(self):
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_preprocess_pipeline(self):
        res = self.preprocessor.preprocess(self.test_filename)

        self.assertTrue("preprocessed_audio" in res or "processed_audio" in res)
        self.assertEqual(res["sr"], 22050)
        self.assertIn("validation", res)


if __name__ == "__main__":
    unittest.main()
