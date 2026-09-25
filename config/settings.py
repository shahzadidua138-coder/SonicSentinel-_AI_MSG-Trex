"""
config/settings.py - Global DSP & Acoustic Model Configuration
SonicSentinel AI - NextWave Acoustic Intelligence Platform
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, 'sonicsentinel.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PLOTS_FOLDER = os.path.join(BASE_DIR, 'static', 'plots')
SAMPLE_AUDIO_FOLDER = os.path.join(BASE_DIR, 'sample_audio')
CONFIG_FOLDER = os.path.join(BASE_DIR, 'config')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_AUDIO_FOLDER, exist_ok=True)

# Acoustic DSP Standardization (SRS §1.2 & §8)
SAMPLE_RATE = 22050          # Standardized 22.05 kHz
CHANNELS = 1                 # Mono conversion
FRAME_LENGTH = 2048          # FFT window size
HOP_LENGTH = 512             # STFT hop length
N_MFCC = 40                  # 40 Mel-frequency cepstral coefficients
N_MELS = 128                 # 128 Mel frequency bins
WINDOW_DURATION = 1.5        # 1.5s sliding window for live mic
WINDOW_STRIDE = 0.75         # 50% overlap stride

# Audio Quality Gatekeeper Thresholds (SRS §8)
SILENCE_RMS_THRESHOLD = 0.005        # Audio with RMS < 0.005 in >80% frames is Silent
SILENCE_FRAME_RATIO = 0.80           # 80% silent frames triggers rejection
CLIPPING_THRESHOLD = 0.999           # Absolute amplitude touching limit
CLIPPING_SAMPLE_RATIO = 0.01         # >1% samples clipped flags Unusable
SNR_GOOD_THRESHOLD = 20.0            # dB
SNR_ACCEPTABLE_THRESHOLD = 10.0      # dB

# Dual-Model Arbitration & Top-Two Margin (SRS §7.3)
DELTA_STRONG_MATCH = 0.15            # |C_py - C_gtm| <= 0.15
DELTA_ACCEPTABLE_MATCH = 0.30        # |C_py - C_gtm| <= 0.30
MIN_TOP2_MARGIN = 0.10               # Separation between rank 1 and rank 2
MIN_HIGH_CONFIDENCE = 0.75           # High confidence bar
MIN_UNCERTAIN_CONFIDENCE = 0.50      # Under 0.50 flags Uncertain Result

# Multi-Window Repeated Detection Persistence (SRS §9 & §10)
CONSECUTIVE_WINDOWS_REQUIRED = 2     # Critical threats require 2-window confirmation unless conf >= 0.85
CRITICAL_INSTANT_CONFIDENCE = 0.85   # Instant alert bypass if both models agree >= 85%

# Mandatory Sound Categories (SRS §1.1)
CATEGORIES = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression",
    "Person Asking for Help",
    "Background Noise"
]
