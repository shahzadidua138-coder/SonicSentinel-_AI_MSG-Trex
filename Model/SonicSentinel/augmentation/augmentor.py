# ============================================================
# SonicSentinel AI - Audio Augmentation
# ============================================================
"""
Training-data augmentation techniques as per SRS Step 5 / xix:
  - Background noise addition
  - Time shifting
  - Pitch shifting
  - Time stretching
  - Volume adjustment
  - Reverberation simulation
  - Distance simulation
  - Recording-device simulation
"""
import numpy as np
import librosa
from scipy.signal import fftconvolve


class AudioAugmentor:
    """Provides controlled audio augmentation for training data."""

    def __init__(self, sr: int = 22050):
        self.sr = sr

    # ------------------------------------------------------------------
    def add_noise(self, y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add Gaussian white noise."""
        noise = np.random.randn(len(y)) * noise_factor
        return (y + noise).astype(np.float32)

    def add_background_noise(
        self, y: np.ndarray, noise: np.ndarray, snr_db: float = 10.0
    ) -> np.ndarray:
        """Mix audio with a real background noise clip at a target SNR."""
        if len(noise) < len(y):
            noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
        noise = noise[: len(y)]

        signal_power = np.mean(y ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return y

        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        return (y + scale * noise).astype(np.float32)

    # ------------------------------------------------------------------
    def time_shift(self, y: np.ndarray, max_shift_fraction: float = 0.2) -> np.ndarray:
        """Randomly shift audio in time."""
        max_shift = int(len(y) * max_shift_fraction)
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(y, shift)

    # ------------------------------------------------------------------
    def pitch_shift(self, y: np.ndarray, n_steps: float = None) -> np.ndarray:
        """Shift pitch by n_steps semitones (random if not specified)."""
        if n_steps is None:
            n_steps = np.random.uniform(-3, 3)
        return librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=n_steps)

    # ------------------------------------------------------------------
    def time_stretch(self, y: np.ndarray, rate: float = None) -> np.ndarray:
        """Time-stretch without changing pitch (random if rate not specified)."""
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        stretched = librosa.effects.time_stretch(y=y, rate=rate)
        # Ensure same length
        if len(stretched) > len(y):
            stretched = stretched[: len(y)]
        elif len(stretched) < len(y):
            stretched = np.pad(stretched, (0, len(y) - len(stretched)))
        return stretched

    # ------------------------------------------------------------------
    def volume_adjustment(self, y: np.ndarray, gain_db: float = None) -> np.ndarray:
        """Adjust volume by a gain factor in dB."""
        if gain_db is None:
            gain_db = np.random.uniform(-6, 6)
        gain = 10 ** (gain_db / 20)
        return np.clip(y * gain, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def add_reverb(
        self, y: np.ndarray, decay: float = 0.3, delay_ms: float = 30.0
    ) -> np.ndarray:
        """Simulate limited reverberation using a simple impulse response."""
        delay_samples = int(self.sr * delay_ms / 1000)
        ir_length = int(self.sr * 0.5)
        ir = np.zeros(ir_length)
        ir[0] = 1.0
        for i in range(1, 6):
            idx = i * delay_samples
            if idx < ir_length:
                ir[idx] = decay ** i

        convolved = fftconvolve(y, ir, mode="full")[: len(y)]
        max_val = np.max(np.abs(convolved))
        if max_val > 0:
            convolved = convolved / max_val
        return convolved.astype(np.float32)

    # ------------------------------------------------------------------
    def simulate_distance(
        self, y: np.ndarray, distance_factor: float = None
    ) -> np.ndarray:
        """Simulate recording at a farther distance (attenuate + add noise)."""
        if distance_factor is None:
            distance_factor = np.random.uniform(1.5, 5.0)
        attenuated = y / distance_factor
        noise = np.random.randn(len(y)) * 0.002 * distance_factor
        return np.clip(attenuated + noise, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def simulate_device(self, y: np.ndarray, device_type: str = "phone") -> np.ndarray:
        """
        Simulate different recording devices via frequency filtering.
        Supported: 'phone', 'cctv', 'handheld'
        """
        from scipy.signal import butter, filtfilt

        nyquist = self.sr / 2
        if device_type == "phone":
            low, high = 300 / nyquist, min(8000, nyquist - 1) / nyquist
        elif device_type == "cctv":
            low, high = 200 / nyquist, min(6000, nyquist - 1) / nyquist
        else:  # handheld
            low, high = 100 / nyquist, min(10000, nyquist - 1) / nyquist

        high = min(high, 0.99)
        low = max(low, 0.01)
        if low >= high:
            return y

        b, a = butter(4, [low, high], btype="band")
        filtered = filtfilt(b, a, y).astype(np.float32)
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            filtered = filtered / max_val
        return filtered

    # ------------------------------------------------------------------
    def augment_random(self, y: np.ndarray, n_augmentations: int = 2) -> list:
        """
        Apply a random selection of augmentations.
        Returns a list of augmented audio arrays.
        """
        augmentation_fns = [
            lambda x: self.add_noise(x, noise_factor=np.random.uniform(0.002, 0.01)),
            lambda x: self.time_shift(x),
            lambda x: self.pitch_shift(x),
            lambda x: self.time_stretch(x),
            lambda x: self.volume_adjustment(x),
            lambda x: self.add_reverb(x),
            lambda x: self.simulate_distance(x),
        ]

        results = []
        for _ in range(n_augmentations):
            aug_fn = np.random.choice(augmentation_fns)
            try:
                augmented = aug_fn(y.copy())
                results.append(augmented)
            except Exception:
                results.append(y.copy())

        return results
