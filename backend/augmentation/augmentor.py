"""
SonicSentinel AI - Data Augmentation Pipeline
"""
import numpy as np
import librosa
import random
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SAMPLE_RATE


class AudioAugmentor:
    """
    Training data augmentation for SonicSentinel AI.
    Implements: noise addition, time shift, pitch shift, time stretch,
    volume variation, reverberation, distance simulation.
    """

    def __init__(self, sr: int = SAMPLE_RATE, seed: int = 42):
        self.sr = sr
        random.seed(seed)
        np.random.seed(seed)

    def add_background_noise(self, y: np.ndarray, noise_factor: float = None) -> np.ndarray:
        """Add random Gaussian background noise."""
        if noise_factor is None:
            noise_factor = random.uniform(0.003, 0.02)
        noise = np.random.randn(len(y)) * noise_factor
        return (y + noise).astype(np.float32)

    def time_shift(self, y: np.ndarray, shift_factor: float = None) -> np.ndarray:
        """Randomly shift audio in time."""
        if shift_factor is None:
            shift_factor = random.uniform(-0.3, 0.3)
        shift_samples = int(len(y) * shift_factor)
        if shift_samples > 0:
            return np.pad(y[:-shift_samples], (shift_samples, 0), mode="constant")
        elif shift_samples < 0:
            return np.pad(y[-shift_samples:], (0, -shift_samples), mode="constant")
        return y

    def pitch_shift(self, y: np.ndarray, n_steps: float = None) -> np.ndarray:
        """Shift pitch by n semitones."""
        if n_steps is None:
            n_steps = random.uniform(-3.0, 3.0)
        try:
            return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        except Exception:
            return y

    def time_stretch(self, y: np.ndarray, rate: float = None) -> np.ndarray:
        """Stretch or compress time without changing pitch."""
        if rate is None:
            rate = random.uniform(0.8, 1.25)
        try:
            stretched = librosa.effects.time_stretch(y, rate=rate)
            # Restore original length
            if len(stretched) < len(y):
                stretched = np.pad(stretched, (0, len(y) - len(stretched)), mode="constant")
            else:
                stretched = stretched[:len(y)]
            return stretched
        except Exception:
            return y

    def volume_variation(self, y: np.ndarray, factor: float = None) -> np.ndarray:
        """Scale audio volume."""
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        y_aug = y * factor
        # Clip to [-1, 1]
        return np.clip(y_aug, -1.0, 1.0).astype(np.float32)

    def add_reverberation(self, y: np.ndarray, reverb_amount: float = None) -> np.ndarray:
        """Simulate simple reverb using a short IR convolution."""
        if reverb_amount is None:
            reverb_amount = random.uniform(0.1, 0.4)
        ir_length = int(0.1 * self.sr)  # 100ms IR
        decay = np.exp(-5 * np.linspace(0, 1, ir_length))
        ir = decay * np.random.randn(ir_length) * reverb_amount
        ir[0] = 1.0  # Direct sound
        try:
            y_reverb = np.convolve(y, ir, mode="full")[:len(y)]
            # Normalize
            max_val = np.max(np.abs(y_reverb))
            if max_val > 0:
                y_reverb = y_reverb / max_val * np.max(np.abs(y))
            return y_reverb.astype(np.float32)
        except Exception:
            return y

    def simulate_distance(self, y: np.ndarray, distance_factor: float = None) -> np.ndarray:
        """
        Simulate distant recording: reduce volume + add low-pass filter effect.
        """
        if distance_factor is None:
            distance_factor = random.uniform(0.3, 0.8)

        # Reduce volume
        y = y * distance_factor

        # Simple low-pass via moving average (simulates air absorption)
        window = max(1, int((1 - distance_factor) * 5))
        if window > 1:
            kernel = np.ones(window) / window
            y = np.convolve(y, kernel, mode="same")

        return y.astype(np.float32)

    def simulate_recording_device(self, y: np.ndarray) -> np.ndarray:
        """Simulate different microphone characteristics via frequency shaping."""
        # Random frequency response variation
        freq_noise = np.random.uniform(0.9, 1.1, size=len(y))
        y_aug = y * freq_noise
        return np.clip(y_aug, -1.0, 1.0).astype(np.float32)

    def augment(
        self,
        y: np.ndarray,
        augmentations: List[str] = None,
        p: float = 0.5,
    ) -> np.ndarray:
        """
        Apply a random subset of augmentations.
        
        Args:
            y: Input audio array
            augmentations: List of augmentation names to apply (None = all)
            p: Probability of applying each augmentation
        """
        all_augmentations = {
            "noise": self.add_background_noise,
            "time_shift": self.time_shift,
            "pitch_shift": self.pitch_shift,
            "time_stretch": self.time_stretch,
            "volume": self.volume_variation,
            "reverb": self.add_reverberation,
            "distance": self.simulate_distance,
            "device": self.simulate_recording_device,
        }

        if augmentations is None:
            augmentations = list(all_augmentations.keys())

        y_aug = y.copy().astype(np.float32)
        for aug_name in augmentations:
            if aug_name in all_augmentations and random.random() < p:
                try:
                    y_aug = all_augmentations[aug_name](y_aug)
                except Exception:
                    pass  # Skip failed augmentation

        return y_aug

    def augment_dataset(
        self,
        X: np.ndarray,
        y_labels: np.ndarray,
        augmentations_per_sample: int = 2,
        p: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment an entire dataset.
        Returns augmented X and corresponding labels.
        Note: Augmented samples are ADDITIONS (original + augmented).
        """
        X_aug_list = []
        y_aug_list = []

        for i in range(len(X)):
            for _ in range(augmentations_per_sample):
                aug = self.augment(X[i], p=p)
                X_aug_list.append(aug)
                y_aug_list.append(y_labels[i])

        X_aug = np.array(X_aug_list, dtype=np.float32)
        y_aug = np.array(y_aug_list)

        return X_aug, y_aug
