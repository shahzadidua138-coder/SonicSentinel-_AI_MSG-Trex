# ============================================================
# SonicSentinel AI - Audio Event Model
# ============================================================
import enum
import uuid
from datetime import datetime, timezone
from src.extensions import db


class EventStatus(enum.Enum):
    """Lifecycle statuses for an audio event (SRS requirement lxii)."""
    UPLOADED = "Uploaded"
    CLASSIFIED = "Classified"
    UNCERTAIN = "Uncertain"
    ALERT_GENERATED = "Alert Generated"
    MANUAL_REVIEW = "Manual Review"
    REVIEWED = "Reviewed"
    CLOSED = "Closed"


class AudioEvent(db.Model):
    """
    Core table that stores every uploaded or live-captured audio event.

    Tracks the full lifecycle from upload → classification → alert → review → close.
    Stores metadata, quality assessment, and file hashes for duplicate detection.
    """
    __tablename__ = "audio_events"

    id = db.Column(db.Integer, primary_key=True)
    audio_id = db.Column(db.String(50), unique=True, nullable=False, index=True, default=lambda: f"AUD-{uuid.uuid4().hex[:8].upper()}")
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)

    # --- Audio Metadata ---
    file_format = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.Float, nullable=False)
    sampling_rate = db.Column(db.Integer, nullable=False)
    channels = db.Column(db.Integer, default=1)
    bit_depth = db.Column(db.Integer, nullable=True)
    file_size = db.Column(db.Integer, nullable=False)  # bytes

    # --- Source Information ---
    source_type = db.Column(db.String(20), default="upload")  # upload / microphone
    recording_environment = db.Column(db.String(50), nullable=True)
    device = db.Column(db.String(50), nullable=True)

    # --- Audio Quality ---
    audio_quality = db.Column(db.String(20), nullable=True)  # Good/Acceptable/Poor/Unusable
    snr_db = db.Column(db.Float, nullable=True)
    has_clipping = db.Column(db.Boolean, default=False)
    is_silent = db.Column(db.Boolean, default=False)

    # --- Duplicate Detection (SHA-256) ---
    sha256_hash = db.Column(db.String(64), nullable=True, index=True)
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of = db.Column(db.String(30), nullable=True)

    # --- Status ---
    status = db.Column(db.Enum(EventStatus), default=EventStatus.UPLOADED, nullable=False)

    # --- Timestamps ---
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    processed_at = db.Column(db.DateTime, nullable=True)

    # --- Foreign Keys ---
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # --- Relationships ---
    prediction = db.relationship(
        "Prediction", backref="audio_event", uselist=False, cascade="all, delete-orphan"
    )
    alerts = db.relationship("Alert", backref="audio_event", lazy="dynamic", cascade="all, delete-orphan")
    reviews = db.relationship("Review", backref="audio_event", lazy="dynamic", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<AudioEvent {self.audio_id} [{self.status.value}]>"
