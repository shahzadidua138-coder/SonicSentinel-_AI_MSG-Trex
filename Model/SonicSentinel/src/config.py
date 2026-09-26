# ============================================================
# SonicSentinel AI - Application Configuration
# ============================================================
import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Config:
    """Base configuration class."""

    # --- Flask Core ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "sonic-sentinel-dev-key-change-in-production")
    DEBUG = False
    TESTING = False

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(BASE_DIR, 'database', 'sonicsentinel.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- File Upload ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload
    ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

    # --- Audio Processing ---
    TARGET_SAMPLE_RATE = 22050
    TARGET_CHANNELS = 1  # Mono
    SEGMENT_DURATION = 3.0  # seconds
    MAX_AUDIO_DURATION = 300  # 5 minutes max
    MIN_AUDIO_DURATION = 0.5  # minimum 0.5 seconds

    # --- Feature Extraction ---
    N_MFCC = 40
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    N_CHROMA = 12

    # --- ML Models ---
    MODEL_DIR = os.path.join(BASE_DIR, "python_models")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

    # --- Sound Classes (10 Mandatory) ---
    SOUND_CLASSES = [
        "Machinery Fault",
        "Glass Breaking",
        "Alarm or Siren",
        "Vehicle Horn",
        "Animal Sound",
        "Gunshot",
        "Panic Scream",
        "Aggression",
        "Person Asking for Help",
        "Background Noise",
    ]

    # --- Critical Classes ---
    CRITICAL_CLASSES = [
        "Gunshot",
        "Glass Breaking",
        "Panic Scream",
        "Aggression",
        "Person Asking for Help",
        "Alarm or Siren",
        "Machinery Fault",
    ]

    # --- Confidence Thresholds ---
    DEFAULT_CONFIDENCE_THRESHOLD = 0.60
    LOW_CONFIDENCE_THRESHOLD = 0.40
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    TOP_TWO_MARGIN_THRESHOLD = 0.15

    # --- Alert Severity Mapping ---
    SEVERITY_MAPPING = {
        "Background Noise": "Informational",
        "Vehicle Horn": "Low",
        "Animal Sound": "Low",
        "Machinery Fault": "High",
        "Glass Breaking": "High",
        "Alarm or Siren": "High",
        "Aggression": "High",
        "Panic Scream": "Critical",
        "Person Asking for Help": "Critical",
        "Gunshot": "Critical",
    }

    # --- Audio Quality Thresholds ---
    SILENCE_THRESHOLD_DB = -40
    CLIPPING_THRESHOLD = 0.99
    SNR_GOOD = 20
    SNR_ACCEPTABLE = 10
    SNR_POOR = 5

    # --- Session / Auth ---
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # --- Data Paths ---
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    AUDIO_DATASET_DIR = os.path.join(DATA_DIR, "audio_dataset")
    METADATA_CSV = os.path.join(AUDIO_DATASET_DIR, "metadata.csv")

    # --- Reports ---
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")

    # --- Data Retention (days) ---
    DATA_RETENTION_DAYS = 90

    # --- Repeated Detection ---
    REPEATED_DETECTION_WINDOW = 10  # seconds
    REPEATED_DETECTION_COUNT = 2  # minimum consecutive detections


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

# Top-level helper variables for clean imports across backend modules
DATABASE_PATH = os.path.join(BASE_DIR, "database", "sonicsentinel.db")
UPLOAD_FOLDER = Config.UPLOAD_FOLDER
MODEL_DIR = Config.MODEL_DIR
SOUND_CLASSES = Config.SOUND_CLASSES
SEVERITY_MAPPING = Config.SEVERITY_MAPPING
METADATA_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "audio_dataset", "metadata.csv")

ML_CONFIG = {
    "min_confidence_threshold": Config.DEFAULT_CONFIDENCE_THRESHOLD,
    "supported_algorithms": ["SVM", "Random Forest", "XGBoost"],
    "features_dimension": 373
}

ALERT_CONFIG = {
    "min_confidence_threshold": Config.DEFAULT_CONFIDENCE_THRESHOLD,
    "deduplication_window_seconds": 30,
    "auto_escalate_minutes": 15
}

GTM_SIMULATION_CONFIG = {
    "agreement_rate": 0.90
}

