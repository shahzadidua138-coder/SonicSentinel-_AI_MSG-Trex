"""
SonicSentinel AI - Audio Preprocessing Pipeline
"""
import numpy as np
import librosa
import soundfile as sf
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    SAMPLE_RATE, MAX_AUDIO_DURATION, MIN_AUDIO_DURATION,
    SEGMENT_DURATION, SILENCE_THRESHOLD, CLIPPING_THRESHOLD,
    SUPPORTED_FORMATS, QUALITY_THRESHOLDS
)


class AudioPreprocessor:
    """
    Complete audio preprocessing pipeline for SonicSentinel AI.
    Handles: validation, loading, resampling, mono conversion,
    normalization, silence trimming, noise reduction, segmentation.
    """

    def __init__(
        self,
        target_sr: int = SAMPLE_RATE,
        segment_duration: float = SEGMENT_DURATION,
        max_duration: float = MAX_AUDIO_DURATION,
        min_duration: float = MIN_AUDIO_DURATION,
    ):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.segment_samples = int(segment_duration * target_sr)

    # ─────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate audio file before processing."""
        path = Path(file_path)
        errors = []
        warnings_list = []

        if not path.exists():
            errors.append(f"File not found: '{file_path}'")
            return {"valid": False, "errors": errors, "warnings": warnings_list, "metadata": {}}

        # Format check
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            errors.append(f"Unsupported format '{path.suffix}'. Supported: {SUPPORTED_FORMATS}")
            return {"valid": False, "errors": errors, "warnings": warnings_list}

        # File size check (50 MB limit)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 50:
            errors.append(f"File too large: {file_size_mb:.1f} MB. Max 50 MB.")

        # Try loading metadata
        try:
            info = sf.info(str(path))
            duration = info.duration
            sample_rate = info.samplerate

            if duration < self.min_duration:
                errors.append(f"Audio too short: {duration:.2f}s. Minimum: {self.min_duration}s")
            if duration > self.max_duration * 2:  # Allow some extra
                warnings_list.append(f"Audio is very long: {duration:.1f}s. Will be segmented.")

            metadata = {
                "filename": path.name,
                "format": path.suffix.lower(),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "num_channels": info.channels,
                "file_size_bytes": path.stat().st_size,
                "subtype": getattr(info, "subtype", "unknown"),
            }
        except Exception as e:
            errors.append(f"Cannot read audio file: {str(e)}")
            return {"valid": False, "errors": errors, "warnings": warnings_list, "metadata": {}}

        # Check for audio signal presence
        try:
            y, sr = librosa.load(str(path), sr=None, mono=True, duration=3.0)
            rms = float(np.sqrt(np.mean(y**2)))
            if rms < SILENCE_THRESHOLD:
                errors.append("Audio appears to be silent. No usable signal detected.")
        except Exception as e:
            warnings_list.append(f"Quick signal check failed: {e}")

        valid = len(errors) == 0
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings_list,
            "metadata": metadata if valid or metadata else {},
        }

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash for duplicate detection."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ─────────────────────────────────────────────
    # Loading & Basic Preprocessing
    # ─────────────────────────────────────────────
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to mono at target sample rate."""
        y, sr = librosa.load(str(file_path), sr=self.target_sr, mono=True)
        return y, sr

    def resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)
        return y

    def normalize_amplitude(self, y: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude to [-1, 1] range."""
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val
        return y

    def trim_silence(self, y: np.ndarray, top_db: int = 30) -> np.ndarray:
        """Trim leading and trailing silence."""
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed if len(y_trimmed) > 0 else y

    def reduce_noise(self, y: np.ndarray) -> np.ndarray:
        """
        Simple spectral gating noise reduction.
        Estimates noise from first 0.5s of audio.
        """
        noise_samples = min(int(0.5 * self.target_sr), len(y) // 4)
        if noise_samples < 100:
            return y

        # Compute STFT
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise profile from beginning
        noise_mag = np.mean(magnitude[:, :max(1, noise_samples // 512)], axis=1, keepdims=True)

        # Spectral gating
        mask = magnitude > (noise_mag * 2.0)
        clean_mag = magnitude * mask

        # Reconstruct
        clean_stft = clean_mag * np.exp(1j * phase)
        y_clean = librosa.istft(clean_stft, hop_length=512, length=len(y))
        return y_clean

    def pad_or_truncate(self, y: np.ndarray, target_length: int) -> np.ndarray:
        """Pad with zeros or truncate to target length."""
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]
        return y

    # ─────────────────────────────────────────────
    # Segmentation
    # ─────────────────────────────────────────────
    def segment_audio(
        self, y: np.ndarray, overlap: float = 0.5
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into fixed-duration segments with optional overlap.
        Returns list of (segment_array, start_time, end_time).
        """
        hop = int(self.segment_samples * (1 - overlap))
        segments = []
        start = 0

        while start + self.segment_samples <= len(y):
            seg = y[start : start + self.segment_samples]
            start_time = start / self.target_sr
            end_time = (start + self.segment_samples) / self.target_sr
            segments.append((seg, start_time, end_time))
            start += hop

        # Handle remaining audio (pad last segment)
        if start < len(y) and len(y) - start >= self.segment_samples // 2:
            seg = self.pad_or_truncate(y[start:], self.segment_samples)
            start_time = start / self.target_sr
            end_time = start_time + self.segment_duration
            segments.append((seg, start_time, end_time))

        # If no segments created (very short), pad the whole thing
        if not segments:
            seg = self.pad_or_truncate(y, self.segment_samples)
            segments.append((seg, 0.0, self.segment_duration))

        return segments

    # ─────────────────────────────────────────────
    # Audio Quality Assessment
    # ─────────────────────────────────────────────
    def assess_quality(self, y: np.ndarray) -> Dict[str, Any]:
        """Assess audio quality: silence, clipping, SNR, noise."""
        if len(y) == 0:
            return {"quality": "unusable", "rms": 0.0, "clip_ratio": 0.0, "snr_db": -np.inf, "silence_ratio": 1.0}

        # RMS energy
        rms = float(np.sqrt(np.mean(y**2)))

        # Clipping ratio
        clip_ratio = float(np.mean(np.abs(y) >= CLIPPING_THRESHOLD))

        # Silence ratio
        frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        silence_ratio = float(np.mean(frame_rms < SILENCE_THRESHOLD))

        # SNR estimation
        noise_floor = np.percentile(np.abs(y), 10)
        signal_peak = np.percentile(np.abs(y), 90)
        snr_db = float(20 * np.log10((signal_peak + 1e-9) / (noise_floor + 1e-9)))

        # Quality classification
        quality = "unusable"
        if rms > QUALITY_THRESHOLDS["good"]["min_rms"] and \
           clip_ratio < QUALITY_THRESHOLDS["good"]["max_clip_ratio"] and \
           snr_db > QUALITY_THRESHOLDS["good"]["min_snr"]:
            quality = "good"
        elif rms > QUALITY_THRESHOLDS["acceptable"]["min_rms"] and \
             clip_ratio < QUALITY_THRESHOLDS["acceptable"]["max_clip_ratio"] and \
             snr_db > QUALITY_THRESHOLDS["acceptable"]["min_snr"]:
            quality = "acceptable"
        elif rms > QUALITY_THRESHOLDS["poor"]["min_rms"] and \
             clip_ratio < QUALITY_THRESHOLDS["poor"]["max_clip_ratio"] and \
             snr_db > QUALITY_THRESHOLDS["poor"]["min_snr"]:
            quality = "poor"

        notes = []
        if rms < SILENCE_THRESHOLD:
            notes.append("Near-silent audio")
        if clip_ratio > 0.01:
            notes.append(f"Clipping detected ({clip_ratio*100:.1f}% of samples)")
        if silence_ratio > 0.5:
            notes.append(f"High silence ratio ({silence_ratio*100:.1f}%)")
        if snr_db < 10:
            notes.append(f"Low SNR ({snr_db:.1f} dB)")

        return {
            "quality": quality,
            "rms": rms,
            "clip_ratio": clip_ratio,
            "snr_db": snr_db,
            "silence_ratio": silence_ratio,
            "notes": "; ".join(notes) if notes else "No issues detected",
        }

    # ─────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────
    def preprocess(
        self,
        file_path: str,
        apply_noise_reduction: bool = True,
        apply_silence_trim: bool = True,
    ) -> Dict[str, Any]:
        """
        Full preprocessing pipeline.
        Returns: {audio, sample_rate, segments, quality, metadata}
        """
        # Load
        y, sr = self.load_audio(file_path)

        # Quality on raw audio
        quality_info = self.assess_quality(y)

        # Trim silence
        if apply_silence_trim:
            y = self.trim_silence(y)

        # Normalize
        y = self.normalize_amplitude(y)

        # Noise reduction (skip if unusable)
        if apply_noise_reduction and quality_info["quality"] != "unusable":
            y = self.reduce_noise(y)
            y = self.normalize_amplitude(y)  # Re-normalize after noise reduction

        # Segmentation
        segments_raw = self.segment_audio(y)
        segments = []
        for seg_y, start, end in segments_raw:
            seg_y = self.normalize_amplitude(seg_y)
            seg_quality = self.assess_quality(seg_y)
            segments.append({
                "audio": seg_y,
                "start_time": start,
                "end_time": end,
                "quality": seg_quality["quality"],
            })

        return {
            "audio": y,
            "sample_rate": sr,
            "segments": segments,
            "quality": quality_info,
            "duration_seconds": len(y) / sr,
        }

    def preprocess_live_window(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Lightweight preprocessing for live microphone windows."""
        y = self.normalize_amplitude(y)
        y = self.pad_or_truncate(y, self.segment_samples)
        quality = self.assess_quality(y)
        return y, quality
