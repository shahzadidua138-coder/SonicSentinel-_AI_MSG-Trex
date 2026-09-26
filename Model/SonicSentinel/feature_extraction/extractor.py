# ============================================================
# SonicSentinel AI - Acoustic Feature Extraction
# ============================================================
"""
Extracts meaningful acoustic features as per SRS Step 6:
  - Mel-frequency cepstral coefficients (MFCC)
  - Mel spectrogram
  - Chroma features
  - Zero-crossing rate
  - Root mean square energy
  - Spectral centroid
  - Spectral bandwidth
  - Spectral roll-off
  - Onset strength
  - Tempo
"""
import numpy as np
import librosa


class FeatureExtractor:
    """Extracts acoustic features from preprocessed audio segments."""

    def __init__(
        self,
        sr: int = 22050,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_chroma: int = 12,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    # ------------------------------------------------------------------
    # Individual Feature Extractors
    # ------------------------------------------------------------------
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC features (mean + std across time frames)."""
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram features (mean + std)."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        mel_std = np.std(mel_db, axis=1)
        return np.concatenate([mel_mean, mel_std])

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Extract Chroma features (mean + std)."""
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, n_chroma=self.n_chroma,
            n_fft=self.n_fft, hop_length=self.hop_length,
        )
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])

    def extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Extract Zero-Crossing Rate (mean + std)."""
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_rms(self, y: np.ndarray) -> np.ndarray:
        """Extract Root Mean Square Energy (mean + std)."""
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        return np.array([np.mean(rms), np.std(rms)])

    def extract_spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """Extract Spectral Centroid (mean + std)."""
        sc = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
        )
        return np.array([np.mean(sc), np.std(sc)])

    def extract_spectral_bandwidth(self, y: np.ndarray) -> np.ndarray:
        """Extract Spectral Bandwidth (mean + std)."""
        sb = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
        )
        return np.array([np.mean(sb), np.std(sb)])

    def extract_spectral_rolloff(self, y: np.ndarray) -> np.ndarray:
        """Extract Spectral Roll-off (mean + std)."""
        sro = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
        )
        return np.array([np.mean(sro), np.std(sro)])

    def extract_onset_strength(self, y: np.ndarray) -> np.ndarray:
        """Extract Onset Strength (mean + std)."""
        onset = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length,
        )
        return np.array([np.mean(onset), np.std(onset)])

    def extract_tempo(self, y: np.ndarray) -> np.ndarray:
        """Extract Tempo estimation."""
        tempo = librosa.beat.tempo(y=y, sr=self.sr, hop_length=self.hop_length)
        return np.array([tempo[0]])

    # ------------------------------------------------------------------
    # Combined Feature Vector
    # ------------------------------------------------------------------
    def extract_all_features(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        if sr is not None:
            self.sr = sr
        """
        Extract ALL acoustic features and concatenate into a single vector.

        Feature vector composition:
          - MFCC:               40 mean + 40 std = 80
          - Mel spectrogram:    128 mean + 128 std = 256
          - Chroma:             12 mean + 12 std = 24
          - ZCR:                1 mean + 1 std = 2
          - RMS:                1 mean + 1 std = 2
          - Spectral centroid:  1 mean + 1 std = 2
          - Spectral bandwidth: 1 mean + 1 std = 2
          - Spectral rolloff:   1 mean + 1 std = 2
          - Onset strength:     1 mean + 1 std = 2
          - Tempo:              1 value = 1
          ─────────────────────────────────
          Total:                373 features
        """
        features = []

        try:
            features.append(self.extract_mfcc(y))
        except Exception:
            features.append(np.zeros(self.n_mfcc * 2))

        try:
            features.append(self.extract_mel_spectrogram(y))
        except Exception:
            features.append(np.zeros(self.n_mels * 2))

        try:
            features.append(self.extract_chroma(y))
        except Exception:
            features.append(np.zeros(self.n_chroma * 2))

        try:
            features.append(self.extract_zcr(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_rms(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_spectral_centroid(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_spectral_bandwidth(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_spectral_rolloff(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_onset_strength(y))
        except Exception:
            features.append(np.zeros(2))

        try:
            features.append(self.extract_tempo(y))
        except Exception:
            features.append(np.zeros(1))

        return np.concatenate(features)

    def get_feature_names(self) -> list:
        """Return descriptive names for each feature dimension."""
        names = []
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_{i + 1}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mels):
                names.append(f"mel_{i + 1}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_chroma):
                names.append(f"chroma_{i + 1}_{stat}")
        for feat in ["zcr", "rms", "spectral_centroid", "spectral_bandwidth",
                      "spectral_rolloff", "onset_strength"]:
            names.extend([f"{feat}_mean", f"{feat}_std"])
        names.append("tempo")
        return names

    def extract_features(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        """Alias for extract_all_features."""
        return self.extract_all_features(y, sr)

    def get_feature_vector(self, features=None) -> np.ndarray:
        """Returns 1D feature vector."""
        if isinstance(features, np.ndarray):
            return features
        return self.extract_all_features(features)
