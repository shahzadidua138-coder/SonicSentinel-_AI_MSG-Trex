"""
src/services/quality_assessor.py - Audio Quality Gatekeeper
Implements SRS §8: Silence Check, Clipping Check, SNR, and Quality Grading.
"""

import numpy as np
from config.settings import (
    SILENCE_RMS_THRESHOLD,
    SILENCE_FRAME_RATIO,
    CLIPPING_THRESHOLD,
    CLIPPING_SAMPLE_RATIO,
    SNR_GOOD_THRESHOLD,
    SNR_ACCEPTABLE_THRESHOLD
)


def compute_frame_rms(samples: np.ndarray, frame_size: int = 1024, hop_size: int = 512) -> np.ndarray:
    """Computes RMS energy for each overlapping audio frame."""
    if len(samples) < frame_size:
        return np.array([np.sqrt(np.mean(samples ** 2) + 1e-12)])
    
    num_frames = 1 + (len(samples) - frame_size) // hop_size
    rms_values = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_size
        frame = samples[start:start + frame_size]
        rms_values[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)
    return rms_values


def estimate_snr_db(samples: np.ndarray) -> float:
    """
    Estimates Signal-to-Noise Ratio (SNR) in dB by comparing signal frame energy
    against the noise floor.
    """
    rms = compute_frame_rms(samples)
    if len(rms) == 0:
        return 0.0

    mean_rms = float(np.mean(rms))
    p95 = float(np.percentile(rms, 95))
    p10 = float(np.percentile(rms, 10))

    # If audio is robust and clean throughout (high mean RMS and low noise floor)
    if mean_rms > 0.05:
        # If signal is continuous and loud, noise floor is minimal
        effective_noise = max(p10 ** 2, 1e-4) if p10 < p95 * 0.7 else 1e-4
        snr = 10.0 * np.log10((p95 ** 2) / effective_noise)
    else:
        signal_energy = p95 ** 2
        noise_energy = p10 ** 2 + 1e-6
        if signal_energy <= noise_energy:
            return 5.0
        snr = 10.0 * np.log10(signal_energy / noise_energy)

    return round(float(np.clip(snr, 0.0, 60.0)), 1)


def assess_audio_quality(samples: np.ndarray, sr: int = 22050) -> dict:
    """
    Evaluates audio signal integrity:
    1. Silence Check (RMS < 0.005 in > 80% frames)
    2. Clipping Check (> 1.0% samples >= 0.999)
    3. Signal-to-Noise Ratio (SNR dB)
    4. Categorical Grade: Good, Acceptable, Poor, Unusable
    """
    if len(samples) == 0:
        return {
            "quality_grade": "Unusable",
            "is_silent": True,
            "is_clipped": False,
            "snr_db": 0.0,
            "reason": "Empty audio buffer"
        }

    # Normalize samples to [-1.0, 1.0] if integer range
    if samples.dtype in [np.int16, np.int32]:
        max_val = np.iinfo(samples.dtype).max
        norm_samples = samples.astype(np.float32) / max_val
    else:
        norm_samples = samples.astype(np.float32)

    # 1. Silence check
    rms_frames = compute_frame_rms(norm_samples)
    silent_frames = np.sum(rms_frames < SILENCE_RMS_THRESHOLD)
    silent_ratio = silent_frames / max(len(rms_frames), 1)
    is_silent = bool(silent_ratio >= SILENCE_FRAME_RATIO)

    # 2. Clipping check
    clipped_samples = np.sum(np.abs(norm_samples) >= CLIPPING_THRESHOLD)
    clipped_ratio = clipped_samples / max(len(norm_samples), 1)
    is_clipped = bool(clipped_ratio >= CLIPPING_SAMPLE_RATIO)

    # 3. SNR estimation
    snr_db = estimate_snr_db(norm_samples)

    # 4. Grading logic
    if is_silent:
        quality_grade = "Unusable"
        reason = f"Silent recording: {silent_ratio*100:.1f}% frames below {SILENCE_RMS_THRESHOLD} RMS threshold"
    elif is_clipped:
        quality_grade = "Unusable"
        reason = f"Severe audio clipping: {clipped_ratio*100:.1f}% samples exceed dynamic range limit"
    elif snr_db >= SNR_GOOD_THRESHOLD:
        quality_grade = "Good"
        reason = f"Clean acoustic signal (SNR: {snr_db} dB)"
    elif snr_db >= SNR_ACCEPTABLE_THRESHOLD:
        quality_grade = "Acceptable"
        reason = f"Moderate ambient noise floor (SNR: {snr_db} dB)"
    else:
        quality_grade = "Poor"
        reason = f"Sub-optimal signal-to-noise ratio (SNR: {snr_db} dB)"

    return {
        "quality_grade": quality_grade,
        "is_silent": is_silent,
        "is_clipped": is_clipped,
        "snr_db": snr_db,
        "silent_ratio": round(float(silent_ratio), 3),
        "clipped_ratio": round(float(clipped_ratio), 3),
        "reason": reason
    }
