"""
SonicSentinel AI - Acoustic Feature Extraction
Extracts MFCC, Mel Spectrogram, Chroma, ZCR, RMSE, Spectral features, etc.
"""
import numpy as np
import librosa
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SAMPLE_RATE, N_MFCC, N_MELS, HOP_LENGTH, N_FFT


class FeatureExtractor:
    """
    Extracts a comprehensive set of acoustic features from audio segments.
    Features: MFCC, Delta-MFCC, Mel Spectrogram, Chroma, ZCR, RMS,
              Spectral Centroid, Bandwidth, Roll-off, Onset Strength, Tempo.
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    # ─────────────────────────────────────────────
    # Individual Feature Methods
    # ─────────────────────────────────────────────
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """Extract MFCC coefficients: mean + std of each coefficient."""
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                                     n_fft=self.n_fft, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        feats = []
        for m in [mfcc, mfcc_delta, mfcc_delta2]:
            feats.extend(np.mean(m, axis=1))
            feats.extend(np.std(m, axis=1))
        return np.array(feats)  # shape: (n_mfcc * 3 * 2,) = 240

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram statistics."""
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels,
                                              n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return np.concatenate([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])  # 256

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft,
                                              hop_length=self.hop_length)
        return np.concatenate([np.mean(chroma, axis=1), np.std(chroma, axis=1)])  # 24

    def extract_chroma_cqt(self, y: np.ndarray) -> np.ndarray:
        """Extract CQT-based chroma."""
        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)
            return np.concatenate([np.mean(chroma_cqt, axis=1), np.std(chroma_cqt, axis=1)])
        except Exception:
            return np.zeros(24)

    def extract_zcr(self, y: np.ndarray) -> np.ndarray:
        """Extract zero-crossing rate."""
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length)
        return np.array([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])

    def extract_rms(self, y: np.ndarray) -> np.ndarray:
        """Extract root mean square energy."""
        rms = librosa.feature.rms(y=y, frame_length=self.n_fft, hop_length=self.hop_length)
        return np.array([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

    def extract_spectral_centroid(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral centroid."""
        sc = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.n_fft,
                                                hop_length=self.hop_length)
        return np.array([np.mean(sc), np.std(sc), np.max(sc)])

    def extract_spectral_bandwidth(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral bandwidth."""
        bw = librosa.feature.spectral_bandwidth(y=y, sr=self.sr, n_fft=self.n_fft,
                                                  hop_length=self.hop_length)
        return np.array([np.mean(bw), np.std(bw), np.max(bw)])

    def extract_spectral_rolloff(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral roll-off (at 85% and 95%)."""
        ro85 = librosa.feature.spectral_rolloff(y=y, sr=self.sr, roll_percent=0.85,
                                                  n_fft=self.n_fft, hop_length=self.hop_length)
        ro95 = librosa.feature.spectral_rolloff(y=y, sr=self.sr, roll_percent=0.95,
                                                  n_fft=self.n_fft, hop_length=self.hop_length)
        return np.array([np.mean(ro85), np.std(ro85), np.mean(ro95), np.std(ro95)])

    def extract_spectral_contrast(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral contrast."""
        try:
            sc = librosa.feature.spectral_contrast(y=y, sr=self.sr, n_fft=self.n_fft,
                                                    hop_length=self.hop_length)
            return np.concatenate([np.mean(sc, axis=1), np.std(sc, axis=1)])
        except Exception:
            return np.zeros(14)

    def extract_spectral_flatness(self, y: np.ndarray) -> np.ndarray:
        """Extract spectral flatness."""
        sf = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)
        return np.array([np.mean(sf), np.std(sf)])

    def extract_onset_strength(self, y: np.ndarray) -> np.ndarray:
        """Extract onset strength envelope statistics."""
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        return np.array([
            np.mean(onset_env), np.std(onset_env),
            np.max(onset_env), np.sum(onset_env > np.mean(onset_env))
        ])

    def extract_tempo(self, y: np.ndarray) -> np.ndarray:
        """Extract estimated tempo."""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
            return np.array([float(tempo)])
        except Exception:
            return np.array([0.0])

    def extract_tonnetz(self, y: np.ndarray) -> np.ndarray:
        """Extract tonal centroid features (tonnetz)."""
        try:
            y_harm = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harm, sr=self.sr)
            return np.concatenate([np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)])
        except Exception:
            return np.zeros(12)

    def extract_poly_features(self, y: np.ndarray) -> np.ndarray:
        """Extract polynomial features from STFT."""
        try:
            poly = librosa.feature.poly_features(y=y, sr=self.sr, order=2,
                                                   n_fft=self.n_fft, hop_length=self.hop_length)
            return np.concatenate([np.mean(poly, axis=1), np.std(poly, axis=1)])
        except Exception:
            return np.zeros(6)

    # ─────────────────────────────────────────────
    # Main Extraction Method
    # ─────────────────────────────────────────────
    def extract_all(self, y: np.ndarray) -> np.ndarray:
        """
        Extract complete feature vector from a single audio segment.
        Returns flat numpy array of features.
        """
        if len(y) == 0:
            raise ValueError("Empty audio signal provided")

        # Ensure minimum length for STFT
        min_samples = self.n_fft
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)), mode="constant")

        feature_parts = []

        try:
            feature_parts.append(self.extract_mfcc(y))           # 240
        except Exception:
            feature_parts.append(np.zeros(self.n_mfcc * 6))

        try:
            feature_parts.append(self.extract_chroma(y))         # 24
        except Exception:
            feature_parts.append(np.zeros(24))

        try:
            feature_parts.append(self.extract_chroma_cqt(y))     # 24
        except Exception:
            feature_parts.append(np.zeros(24))

        try:
            feature_parts.append(self.extract_zcr(y))            # 4
        except Exception:
            feature_parts.append(np.zeros(4))

        try:
            feature_parts.append(self.extract_rms(y))            # 4
        except Exception:
            feature_parts.append(np.zeros(4))

        try:
            feature_parts.append(self.extract_spectral_centroid(y))   # 3
        except Exception:
            feature_parts.append(np.zeros(3))

        try:
            feature_parts.append(self.extract_spectral_bandwidth(y))  # 3
        except Exception:
            feature_parts.append(np.zeros(3))

        try:
            feature_parts.append(self.extract_spectral_rolloff(y))    # 4
        except Exception:
            feature_parts.append(np.zeros(4))

        try:
            feature_parts.append(self.extract_spectral_contrast(y))   # 14
        except Exception:
            feature_parts.append(np.zeros(14))

        try:
            feature_parts.append(self.extract_spectral_flatness(y))   # 2
        except Exception:
            feature_parts.append(np.zeros(2))

        try:
            feature_parts.append(self.extract_onset_strength(y))      # 4
        except Exception:
            feature_parts.append(np.zeros(4))

        try:
            feature_parts.append(self.extract_tempo(y))               # 1
        except Exception:
            feature_parts.append(np.zeros(1))

        try:
            feature_parts.append(self.extract_tonnetz(y))             # 12
        except Exception:
            feature_parts.append(np.zeros(12))

        try:
            feature_parts.append(self.extract_poly_features(y))       # 6
        except Exception:
            feature_parts.append(np.zeros(6))

        features = np.concatenate(feature_parts)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.astype(np.float32)

    def extract_mel_spectrogram_2d(self, y: np.ndarray) -> np.ndarray:
        """
        Extract 2D Mel spectrogram for CNN input.
        Returns shape: (n_mels, time_frames)
        """
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels,
                                              n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names corresponding to extract_all output."""
        names = []
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_{i+1}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_delta_{i+1}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_delta2_{i+1}_{stat}")
        for i in range(12):
            names.append(f"chroma_{i+1}_mean")
        for i in range(12):
            names.append(f"chroma_{i+1}_std")
        for i in range(12):
            names.append(f"chroma_cqt_{i+1}_mean")
        for i in range(12):
            names.append(f"chroma_cqt_{i+1}_std")
        names += ["zcr_mean", "zcr_std", "zcr_max", "zcr_min"]
        names += ["rms_mean", "rms_std", "rms_max", "rms_min"]
        names += ["spec_centroid_mean", "spec_centroid_std", "spec_centroid_max"]
        names += ["spec_bandwidth_mean", "spec_bandwidth_std", "spec_bandwidth_max"]
        names += ["spec_rolloff85_mean", "spec_rolloff85_std", "spec_rolloff95_mean", "spec_rolloff95_std"]
        for i in range(7):
            names.append(f"spec_contrast_{i+1}_mean")
        for i in range(7):
            names.append(f"spec_contrast_{i+1}_std")
        names += ["spec_flatness_mean", "spec_flatness_std"]
        names += ["onset_mean", "onset_std", "onset_max", "onset_count"]
        names += ["tempo"]
        for i in range(6):
            names.append(f"tonnetz_{i+1}_mean")
        for i in range(6):
            names.append(f"tonnetz_{i+1}_std")
        for i in range(3):
            names.append(f"poly_{i+1}_mean")
        for i in range(3):
            names.append(f"poly_{i+1}_std")
        return names
