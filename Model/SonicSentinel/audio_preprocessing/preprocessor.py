# ============================================================
# SonicSentinel AI - Audio Preprocessing Service
# ============================================================
"""
Handles all audio preprocessing steps as per SRS Step 4:
  - Resampling to target sample rate (22050 Hz)
  - Stereo-to-mono conversion
  - Amplitude normalization
  - Silence trimming
  - Noise reduction
  - Fixed-duration segmentation
  - Padding or truncation
  - Audio format conversion
"""
import os
import hashlib
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


class AudioPreprocessor:
    """Full audio preprocessing pipeline for SonicSentinel AI."""

    def __init__(
        self,
        target_sr: int = 22050,
        target_channels: int = 1,
        segment_duration: float = 3.0,
        silence_threshold_db: float = -40.0,
        clipping_threshold: float = 0.99,
    ):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.segment_duration = segment_duration
        self.silence_threshold_db = silence_threshold_db
        self.clipping_threshold = clipping_threshold

    # ------------------------------------------------------------------
    # 1. Load Audio
    # ------------------------------------------------------------------
    def load_audio(self, file_path: str) -> tuple:
        """
        Load an audio file and return (audio_array, sample_rate).
        Supports WAV, MP3, FLAC, OGG, M4A via librosa/soundfile.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        y, sr = librosa.load(file_path, sr=None, mono=False)
        return y, sr

    # ------------------------------------------------------------------
    # 2. Validate Audio
    # ------------------------------------------------------------------
    def validate_audio(
        self, y: np.ndarray, sr: int, min_duration: float = 0.5, max_duration: float = 300.0
    ) -> dict:
        """
        Validate audio file properties: format, duration, silence, clipping.
        Returns a dict with validation results and extracted metadata.
        """
        duration = librosa.get_duration(y=y, sr=sr)

        result = {
            "is_valid": True,
            "duration": round(duration, 2),
            "sample_rate": sr,
            "channels": 1 if y.ndim == 1 else y.shape[0],
            "is_silent": False,
            "has_clipping": False,
            "errors": [],
        }

        # Duration check
        if duration < min_duration:
            result["is_valid"] = False
            result["errors"].append(
                f"Audio too short ({duration:.2f}s). Minimum is {min_duration}s."
            )
        if duration > max_duration:
            result["is_valid"] = False
            result["errors"].append(
                f"Audio too long ({duration:.2f}s). Maximum is {max_duration}s."
            )

        # Silence check
        mono = y if y.ndim == 1 else librosa.to_mono(y)
        rms = np.sqrt(np.mean(mono ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        if rms_db < self.silence_threshold_db:
            result["is_silent"] = True
            result["is_valid"] = False
            result["errors"].append("Audio appears to be silent or near-silent.")

        # Clipping check
        max_amp = np.max(np.abs(mono))
        if max_amp >= self.clipping_threshold:
            result["has_clipping"] = True

        return result

    # ------------------------------------------------------------------
    # 3. Resample
    # ------------------------------------------------------------------
    def resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == self.target_sr:
            return y
        return librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)

    # ------------------------------------------------------------------
    # 4. Convert to Mono
    # ------------------------------------------------------------------
    def to_mono(self, y: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono."""
        if y.ndim == 1:
            return y
        return librosa.to_mono(y)

    # ------------------------------------------------------------------
    # 5. Normalize Amplitude
    # ------------------------------------------------------------------
    def normalize(self, y: np.ndarray) -> np.ndarray:
        """Peak-normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y

    # ------------------------------------------------------------------
    # 6. Trim Silence
    # ------------------------------------------------------------------
    def trim_silence(self, y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
        """Remove leading and trailing silence using librosa."""
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return trimmed

    # ------------------------------------------------------------------
    # 7. Reduce Noise
    # ------------------------------------------------------------------
    def reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction using noisereduce library.
        Falls back to simple high-pass filter if noisereduce is unavailable.
        """
        if HAS_NOISEREDUCE:
            return nr.reduce_noise(y=y, sr=sr, prop_decrease=0.6)

        # Fallback: simple high-pass Butterworth filter at 80 Hz
        nyquist = sr / 2
        cutoff = 80 / nyquist
        if cutoff < 1.0:
            b, a = butter(4, cutoff, btype="high")
            return filtfilt(b, a, y).astype(np.float32)
        return y

    # ------------------------------------------------------------------
    # 8. Segment Audio
    # ------------------------------------------------------------------
    def segment_audio(self, y: np.ndarray, sr: int) -> list:
        """
        Divide audio into fixed-duration segments.
        Each segment is padded or truncated to exactly segment_duration seconds.
        Returns a list of (segment_array, start_time, end_time) tuples.
        """
        segment_samples = int(self.segment_duration * sr)
        total_samples = len(y)
        segments = []

        for start in range(0, total_samples, segment_samples):
            end = start + segment_samples
            segment = y[start:end]

            # Pad with zeros if the last segment is shorter
            if len(segment) < segment_samples:
                segment = np.pad(segment, (0, segment_samples - len(segment)))

            start_time = start / sr
            end_time = min(end, total_samples) / sr
            segments.append((segment, round(start_time, 3), round(end_time, 3)))

        return segments

    # ------------------------------------------------------------------
    # 9. Full Pipeline
    # ------------------------------------------------------------------
    def preprocess(self, file_path: str) -> dict:
        """
        Run the complete preprocessing pipeline on a single audio file.

        Returns:
            dict with keys: segments, metadata, validation, sr
        """
        # Load
        y, sr = self.load_audio(file_path)

        # Validate
        validation = self.validate_audio(y, sr)
        if not validation["is_valid"]:
            return {"segments": [], "metadata": validation, "validation": validation, "sr": sr}

        # To mono
        y = self.to_mono(y)

        # Resample
        y = self.resample(y, sr)
        sr = self.target_sr

        # Normalize
        y = self.normalize(y)

        # Trim silence
        y = self.trim_silence(y)

        # Noise reduction
        y = self.reduce_noise(y, sr)

        # Re-normalize after noise reduction
        y = self.normalize(y)

        # Segment
        segments = self.segment_audio(y, sr)

        return {
            "segments": segments,
            "metadata": validation,
            "validation": validation,
            "sr": sr,
            "preprocessed_audio": y,
        }

    # ------------------------------------------------------------------
    # 10. Compute SHA-256 Hash (Duplicate Detection)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_hash(file_path: str) -> str:
        """Compute SHA-256 hash of audio file for duplicate detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ------------------------------------------------------------------
    # 11. Audio Quality Assessment
    # ------------------------------------------------------------------
    def assess_quality(self, y: np.ndarray, sr: int) -> dict:
        """
        Classify audio quality as Good / Acceptable / Poor / Unusable.
        Checks: silence, clipping, noise level (SNR estimate), duration.
        """
        mono = y if y.ndim == 1 else librosa.to_mono(y)
        duration = len(mono) / sr

        # RMS energy
        rms = np.sqrt(np.mean(mono ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Peak amplitude
        peak = np.max(np.abs(mono))

        # Estimate SNR (signal vs noise floor)
        frame_length = min(2048, len(mono))
        rms_frames = librosa.feature.rms(y=mono, frame_length=frame_length)[0]
        noise_floor = np.percentile(rms_frames, 10)
        signal_peak = np.percentile(rms_frames, 90)
        snr_estimate = 20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10))

        # Clipping ratio
        clipping_samples = np.sum(np.abs(mono) >= self.clipping_threshold)
        clipping_ratio = clipping_samples / len(mono)

        # Determine quality
        is_silent = rms_db < self.silence_threshold_db
        has_severe_clipping = clipping_ratio > 0.01
        has_mild_clipping = clipping_ratio > 0.001

        if is_silent or duration < 0.3:
            quality = "Unusable"
        elif has_severe_clipping or snr_estimate < 5:
            quality = "Poor"
        elif has_mild_clipping or snr_estimate < 10:
            quality = "Acceptable"
        else:
            quality = "Good"

        return {
            "quality": quality,
            "rms_db": round(float(rms_db), 2),
            "peak_amplitude": round(float(peak), 4),
            "snr_estimate_db": round(float(snr_estimate), 2),
            "clipping_ratio": round(float(clipping_ratio), 6),
            "has_clipping": has_mild_clipping or has_severe_clipping,
            "is_silent": is_silent,
            "duration": round(duration, 2),
        }

    # ------------------------------------------------------------------
    # 12. Save Preprocessed Audio
    # ------------------------------------------------------------------
    def save_audio(self, y: np.ndarray, sr: int, output_path: str) -> str:
        """Save preprocessed audio as WAV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y, sr)
        return output_path
