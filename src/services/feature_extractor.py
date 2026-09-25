"""
src/services/feature_extractor.py - Acoustic Feature Extraction Engine
Extracts 40 MFCCs, Spectral Centroid, Bandwidth, Roll-off, Zero-Crossing Rate (ZCR), and RMS.
"""

import numpy as np


def extract_spectral_features(samples: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop_length: int = 512) -> dict:
    """
    Extracts acoustic features from normalized 1D audio sample array:
    - Zero Crossing Rate (ZCR)
    - RMS Energy (mean & std)
    - Spectral Centroid (mean & std)
    - Spectral Spread / Bandwidth (mean & std)
    - Spectral Roll-off (85% energy frequency)
    - 40 Mel-Frequency Filterbank Energy coefficients (MFCCs proxy)
    """
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)), mode='constant')

    # 1. Zero Crossing Rate
    signs = np.sign(samples)
    signs[signs == 0] = 1
    zcr = np.mean(np.abs(np.diff(signs))) / 2.0

    # 2. RMS Energy
    frame_rms = []
    for i in range(0, len(samples) - n_fft + 1, hop_length):
        frame = samples[i:i + n_fft]
        frame_rms.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    
    frame_rms = np.array(frame_rms) if frame_rms else np.array([np.sqrt(np.mean(samples ** 2) + 1e-12)])
    rms_mean = float(np.mean(frame_rms))
    rms_std = float(np.std(frame_rms))

    # 3. Short-Time Fourier Transform (STFT)
    window = np.hanning(n_fft)
    stft_frames = []
    for i in range(0, len(samples) - n_fft + 1, hop_length):
        frame = samples[i:i + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        stft_frames.append(spectrum)
    
    stft_matrix = np.array(stft_frames) if stft_frames else np.array([np.abs(np.fft.rfft(samples[:n_fft] * window))])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # 4. Spectral Centroid
    magnitudes = stft_matrix + 1e-12
    centroids = np.sum(magnitudes * freqs, axis=1) / np.sum(magnitudes, axis=1)
    centroid_mean = float(np.mean(centroids))
    centroid_std = float(np.std(centroids))

    # 5. Spectral Bandwidth
    bandwidths = np.sqrt(np.sum(magnitudes * ((freqs - centroids[:, None]) ** 2), axis=1) / np.sum(magnitudes, axis=1))
    bandwidth_mean = float(np.mean(bandwidths))

    # 6. Spectral Roll-off (85%)
    cumsum_mag = np.cumsum(magnitudes, axis=1)
    total_energy = cumsum_mag[:, -1:]
    rolloff_indices = np.argmax(cumsum_mag >= 0.85 * total_energy, axis=1)
    rolloff_freqs = freqs[rolloff_indices]
    rolloff_mean = float(np.mean(rolloff_freqs))

    # 7. 40-Channel Mel Filterbank (MFCC proxy)
    n_mels = 40
    mel_min = 0.0
    mel_max = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, len(freqs)))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, min(f_m, len(freqs))):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(bin_points[m] - bin_points[m - 1], 1)
        for k in range(f_m, min(f_m_plus, len(freqs))):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(bin_points[m + 1] - bin_points[m], 1)

    filter_energies = np.dot(magnitudes, fbank.T)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)
    log_mel = 20.0 * np.log10(filter_energies)
    mfcc_means = np.mean(log_mel, axis=0)

    features = {
        "zcr": float(zcr),
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bandwidth_mean": bandwidth_mean,
        "rolloff_mean": rolloff_mean,
        "mfcc_vector": [float(x) for x in mfcc_means[:40]],
        "feature_dim": 40 + 7
    }
    return features
