"""
SonicSentinel AI - Unit Tests for Audio Preprocessor
"""
import pytest
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
import sys

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from audio_preprocessing.preprocessor import AudioPreprocessor
from config.settings import SAMPLE_RATE, SEGMENT_DURATION


@pytest.fixture
def preprocessor():
    return AudioPreprocessor(target_sr=SAMPLE_RATE)


@pytest.fixture
def sample_wav_file():
    """Create a temporary synthetic 3-second 440Hz sine wave."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    sf.write(temp_path, audio, sr)
    
    yield temp_path
    
    # Cleanup
    try:
        Path(temp_path).unlink()
    except Exception:
        pass


def test_validate_file(preprocessor, sample_wav_file):
    res = preprocessor.validate_file(sample_wav_file)
    assert res["valid"] is True
    assert "metadata" in res
    assert res["metadata"]["sample_rate"] == SAMPLE_RATE


def test_validate_nonexistent_file(preprocessor):
    res = preprocessor.validate_file("nonexistent_audio_file.wav")
    assert res["valid"] is False
    assert len(res["errors"]) > 0


def test_load_and_resample(preprocessor, sample_wav_file):
    audio, sr = preprocessor.load_audio(sample_wav_file)
    assert sr == SAMPLE_RATE
    assert len(audio) > 0
    assert isinstance(audio, np.ndarray)


def test_normalize_amplitude(preprocessor):
    raw = np.array([0.1, -0.4, 0.8, -0.2], dtype=np.float32)
    normalized = preprocessor.normalize_amplitude(raw)
    assert np.max(np.abs(normalized)) <= 1.0
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


def test_pad_or_truncate(preprocessor):
    sr = 22050
    target_len = int(SEGMENT_DURATION * sr)
    
    # Short audio -> padded
    short_audio = np.ones(1000, dtype=np.float32)
    padded = preprocessor.pad_or_truncate(short_audio, target_len)
    assert len(padded) == target_len
    
    # Long audio -> truncated
    long_audio = np.ones(target_len + 5000, dtype=np.float32)
    truncated = preprocessor.pad_or_truncate(long_audio, target_len)
    assert len(truncated) == target_len


def test_assess_quality(preprocessor, sample_wav_file):
    audio, sr = preprocessor.load_audio(sample_wav_file)
    quality = preprocessor.assess_quality(audio)
    assert "quality" in quality
    assert "rms" in quality
    assert "clip_ratio" in quality
    assert quality["clip_ratio"] == 0.0
