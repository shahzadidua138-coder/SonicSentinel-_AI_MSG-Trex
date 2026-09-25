"""
create_sample_audio.py - Generates realistic synthetic benchmark audio samples for the 10 categories
and saves them to sample_audio/ for live judging walkthroughs.
"""

import os
import wave
import struct
import numpy as np

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_audio')
os.makedirs(SAMPLE_DIR, exist_ok=True)
SR = 22050


def save_wav(filename: str, samples: np.ndarray):
    path = os.path.join(SAMPLE_DIR, filename)
    norm = np.clip(samples, -1.0, 1.0)
    int_samples = (norm * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int_samples.tobytes())
    print(f"Generated sample: {path} ({len(samples)/SR:.2f}s)")


def generate_all_samples():
    # 1. Gunshot: sharp supersonic impulse blast with rapid exponential decay
    dur_gun = 0.8
    t_gun = np.linspace(0, dur_gun, int(SR * dur_gun), endpoint=False)
    noise = np.random.normal(0, 1, len(t_gun))
    gunshot = (noise * np.exp(-t_gun * 25.0) * 1.8 + np.sin(2 * np.pi * 90 * t_gun) * np.exp(-t_gun * 12.0) * 0.9)
    save_wav("test_gunshot.wav", gunshot)

    # 2. Glass Breaking: high-frequency brittle shatter impulses
    dur_glass = 1.2
    t_glass = np.linspace(0, dur_glass, int(SR * dur_glass), endpoint=False)
    glass = (
        (np.random.normal(0, 0.8, len(t_glass))) * np.exp(-t_glass * 16.0) +
        np.sin(2 * np.pi * 4800 * t_glass) * np.exp(-((t_glass - 0.05) % 0.15) * 20.0) * 0.4 +
        np.sin(2 * np.pi * 7200 * t_glass) * np.exp(-((t_glass - 0.12) % 0.20) * 25.0) * 0.3
    ) * np.exp(-t_glass * 1.8)
    save_wav("test_glass_break.wav", glass)

    # 3. Panic Scream: vocal formant sweep > 1200 Hz with pitch jitter
    dur_scream = 1.5
    t_scream = np.linspace(0, dur_scream, int(SR * dur_scream), endpoint=False)
    pitch = 1200.0 + 350.0 * np.sin(2 * np.pi * 6.5 * t_scream) + np.random.normal(0, 20, len(t_scream))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    scream = (0.7 * np.sin(phase) + 0.3 * np.sin(2 * phase)) * np.exp(-t_scream * 0.4)
    save_wav("test_scream.wav", scream)

    # 4. Severe Audio Clipping: heavily over-amplified signal hitting limits
    dur_clip = 1.5
    t_clip = np.linspace(0, dur_clip, int(SR * dur_clip), endpoint=False)
    clipped = 5.0 * np.sin(2 * np.pi * 350 * t_clip)
    # Clip > 5% of samples
    clipped = np.clip(clipped, -0.9999, 0.9999)
    save_wav("test_clipping_audio.wav", clipped)

    # 5. Silent Recording: ambient room noise with RMS < 0.002
    dur_silent = 3.0
    silent = np.random.normal(0, 0.001, int(SR * dur_silent))
    save_wav("test_silent_recording.wav", silent)

    # 6. Dog Bark: harmonic burst with fast repetition
    dur_dog = 1.0
    t_dog = np.linspace(0, dur_dog, int(SR * dur_dog), endpoint=False)
    bark = (np.sin(2 * np.pi * 420 * t_dog) + 0.5 * np.sin(2 * np.pi * 840 * t_dog)) * np.exp(-((t_dog % 0.3) * 18))
    save_wav("test_dog_bark.wav", bark)


if __name__ == '__main__':
    generate_all_samples()
