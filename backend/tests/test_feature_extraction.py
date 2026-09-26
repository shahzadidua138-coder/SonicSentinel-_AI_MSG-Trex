"""
SonicSentinel AI - Unit Tests for Feature Extractor
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from feature_extraction.feature_extractor import FeatureExtractor
from config.settings import SAMPLE_RATE, N_MFCC, N_MELS


@pytest.fixture
def extractor():
    return FeatureExtractor(sr=SAMPLE_RATE)


@pytest.fixture
def synthetic_audio():
    """Create a 3-second synthetic audio signal with harmonics."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    noise = np.random.normal(0, 0.05, len(t))
    return (signal + noise).astype(np.float32)


def test_extract_mfcc(extractor, synthetic_audio):
    mfcc = extractor.extract_mfcc(synthetic_audio)
    assert mfcc is not None
    assert isinstance(mfcc, np.ndarray)
    # 40 coefficients * 3 (raw, delta, delta2) * 2 (mean, std) = 240
    assert len(mfcc) == N_MFCC * 3 * 2


def test_extract_mel_spectrogram(extractor, synthetic_audio):
    mel_spec = extractor.extract_mel_spectrogram(synthetic_audio)
    assert mel_spec is not None
    assert isinstance(mel_spec, np.ndarray)
    # 128 bands * 2 (mean, std) = 256
    assert len(mel_spec) == N_MELS * 2


def test_extract_chroma(extractor, synthetic_audio):
    chroma = extractor.extract_chroma(synthetic_audio)
    assert chroma is not None
    # 12 pitch classes (mean + std = 24)
    assert len(chroma) == 24


def test_extract_spectral_features(extractor, synthetic_audio):
    sc = extractor.extract_spectral_centroid(synthetic_audio)
    sr = extractor.extract_spectral_rolloff(synthetic_audio)
    con = extractor.extract_spectral_contrast(synthetic_audio)
    
    assert isinstance(sc, np.ndarray) and len(sc) == 3
    assert isinstance(sr, np.ndarray) and len(sr) == 4
    assert isinstance(con, np.ndarray) and len(con) > 0


def test_extract_all_vector(extractor, synthetic_audio):
    vector = extractor.extract_all(synthetic_audio)
    assert isinstance(vector, np.ndarray)
    assert len(vector) == 345  # Comprehensive multi-domain feature vector (345 acoustic features)
    assert not np.isnan(vector).any()
    assert not np.isinf(vector).any()
