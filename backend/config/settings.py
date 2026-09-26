"""
SonicSentinel AI - Application Configuration Settings
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DATA_DIR = BASE_DIR.parent / "audio data"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
SPECTROGRAMS_DIR = STATIC_DIR / "spectrograms"
WAVEFORMS_DIR = STATIC_DIR / "waveforms"
MODELS_DIR = BASE_DIR / "python_models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ALERT_RULES_DIR = BASE_DIR / "alert_rules"
SAMPLE_AUDIO_DIR = BASE_DIR / "sample_audio"

DATABASE_DIR = BASE_DIR / "database"

# Create directories if they don't exist
for d in [UPLOADS_DIR, SPECTROGRAMS_DIR, WAVEFORMS_DIR, MODELS_DIR, DATABASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Flask Configuration
# ─────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "sonic-sentinel-secret-key-change-in-prod")
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
_default_db_path = (DATABASE_DIR / "sonicsentinel.db").resolve().as_posix()
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_default_db_path}")

# ─────────────────────────────────────────────
# JWT Configuration
# ─────────────────────────────────────────────
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwt-secret-key-sonic-sentinel-2026-secure-32bytes")
JWT_EXPIRATION_HOURS = int(os.environ.get("JWT_EXPIRATION_HOURS", 24))

# ─────────────────────────────────────────────
# Audio Configuration
# ─────────────────────────────────────────────
SAMPLE_RATE = 22050              # Target sample rate (Hz)
N_MFCC = 40                      # Number of MFCC coefficients
N_MELS = 128                     # Number of Mel filter banks
HOP_LENGTH = 512
N_FFT = 2048
MAX_AUDIO_DURATION = 30          # Maximum audio duration in seconds
MIN_AUDIO_DURATION = 0.5         # Minimum audio duration in seconds
SEGMENT_DURATION = 3.0           # Segment duration for processing (seconds)
MAX_FILE_SIZE_MB = 50            # Max upload file size in MB
LIVE_WINDOW_DURATION = 2.0       # Live microphone window duration (seconds)

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SILENCE_THRESHOLD = 0.01         # RMS below this = silence
CLIPPING_THRESHOLD = 0.99        # Amplitude above this = clipping

# ─────────────────────────────────────────────
# Sound Categories
# ─────────────────────────────────────────────
SOUND_CLASSES = [
    "aggression",
    "alarm_siren",
    "animal_sound",
    "background_noise",
    "glass_breaking",
    "gunshot",
    "machinery_fault",
    "panic_scream",
    "person_asking_for_help",
    "vehicle_horn",
]

# Human-readable labels mapping
CLASS_LABELS = {
    "aggression": "Aggression",
    "alarm_siren": "Alarm or Siren",
    "animal_sound": "Animal Sound",
    "background_noise": "Background Noise",
    "glass_breaking": "Glass Breaking",
    "gunshot": "Gunshot",
    "machinery_fault": "Machinery Fault",
    "panic_scream": "Panic Scream",
    "person_asking_for_help": "Person Asking for Help",
    "vehicle_horn": "Vehicle Horn",
}

NUM_CLASSES = len(SOUND_CLASSES)

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
MODEL_VERSION = "1.0.0"

# ─────────────────────────────────────────────
# Model Training Configuration
# ─────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
CV_FOLDS = 5

# ─────────────────────────────────────────────
# Alert Severity Mapping
# ─────────────────────────────────────────────
SEVERITY_MAP = {
    "background_noise": "informational",
    "vehicle_horn": "low",
    "animal_sound": "low",
    "machinery_fault": "high",
    "glass_breaking": "high",
    "alarm_siren": "high",
    "aggression": "high",
    "panic_scream": "critical",
    "person_asking_for_help": "critical",
    "gunshot": "critical",
}

CRITICAL_CLASSES = {"gunshot", "panic_scream", "person_asking_for_help"}
HIGH_CLASSES = {"aggression", "glass_breaking", "alarm_siren", "machinery_fault"}

# ─────────────────────────────────────────────
# Confidence Thresholds
# ─────────────────────────────────────────────
MIN_CONFIDENCE_THRESHOLD = 0.60       # Minimum confidence for auto-classification
TOP_TWO_MARGIN_THRESHOLD = 0.15       # Min gap between top-2 classes
MODEL_AGREEMENT_THRESHOLD = 0.20      # Max confidence difference for agreement
CRITICAL_CONFIDENCE_THRESHOLD = 0.70  # Higher threshold for critical events
CONSECUTIVE_DETECTIONS_REQUIRED = 2   # For live monitoring confirmation

# ─────────────────────────────────────────────
# CORS Configuration
# ─────────────────────────────────────────────
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

# ─────────────────────────────────────────────
# Audio Quality Thresholds
# ─────────────────────────────────────────────
QUALITY_THRESHOLDS = {
    "good": {"min_rms": 0.05, "max_clip_ratio": 0.01, "min_snr": 20},
    "acceptable": {"min_rms": 0.02, "max_clip_ratio": 0.05, "min_snr": 10},
    "poor": {"min_rms": 0.005, "max_clip_ratio": 0.15, "min_snr": 5},
    # Below poor = unusable
}
