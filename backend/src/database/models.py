"""
SonicSentinel AI - Database Models (SQLAlchemy ORM)
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────
class UserRole(str, enum.Enum):
    NORMAL_USER = "normal_user"
    AUDIO_REVIEWER = "audio_reviewer"
    SECURITY_OPERATOR = "security_operator"
    MAINTENANCE_OPERATOR = "maintenance_operator"
    ADMINISTRATOR = "administrator"


class EventStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    CLASSIFIED = "classified"
    UNCERTAIN = "uncertain"
    ALERT_GENERATED = "alert_generated"
    MANUAL_REVIEW = "manual_review"
    REVIEWED = "reviewed"
    CLOSED = "closed"


class AlertSeverity(str, enum.Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    DISMISSED = "dismissed"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class AudioQuality(str, enum.Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


class ModelConsistency(str, enum.Enum):
    ACCEPTABLE_MATCH = "acceptable_match"
    WEAK_MATCH = "weak_match"
    MODEL_DISAGREEMENT = "model_disagreement"
    UNCERTAIN_RESULT = "uncertain_result"


class MicrophoneStatus(str, enum.Enum):
    AVAILABLE = "available"
    ACTIVE = "active"
    PAUSED = "paused"
    DISCONNECTED = "disconnected"
    PERMISSION_DENIED = "permission_denied"


# ─────────────────────────────────────────────
# User Model
# ─────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SAEnum(UserRole), default=UserRole.NORMAL_USER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    audio_events = relationship("AudioEvent", back_populates="user")
    reviews = relationship("Review", back_populates="reviewer")
    audit_logs = relationship("AuditLog", back_populates="user")
    microphone_sessions = relationship("MicrophoneSession", back_populates="user")


# ─────────────────────────────────────────────
# Audio Event Model
# ─────────────────────────────────────────────
class AudioEvent(Base):
    __tablename__ = "audio_events"

    id = Column(Integer, primary_key=True, index=True)
    audio_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # File Info
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    file_path = Column(String(1000))
    file_size_bytes = Column(Integer)
    file_hash = Column(String(64))  # SHA-256 for duplicate detection
    audio_format = Column(String(20))

    # Audio Metadata
    duration_seconds = Column(Float)
    sample_rate = Column(Integer)
    num_channels = Column(Integer)
    bit_depth = Column(Integer, nullable=True)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)

    # Audio Quality
    audio_quality = Column(SAEnum(AudioQuality), nullable=True)
    rms_energy = Column(Float, nullable=True)
    clipping_ratio = Column(Float, nullable=True)
    snr_db = Column(Float, nullable=True)
    silence_ratio = Column(Float, nullable=True)
    quality_notes = Column(Text, nullable=True)

    # Python Model Results
    python_predicted_class = Column(String(100), nullable=True)
    python_confidence = Column(Float, nullable=True)
    python_confidence_scores = Column(JSON, nullable=True)  # {class: score}
    python_model_version = Column(String(50), nullable=True)

    # GTM Model Results
    gtm_predicted_class = Column(String(100), nullable=True)
    gtm_confidence = Column(Float, nullable=True)
    gtm_confidence_scores = Column(JSON, nullable=True)  # {class: score}
    gtm_model_version = Column(String(50), nullable=True)

    # Comparison
    model_consistency = Column(SAEnum(ModelConsistency), nullable=True)
    confidence_difference = Column(Float, nullable=True)
    top_two_margin = Column(Float, nullable=True)
    models_agree = Column(Boolean, nullable=True)

    # Final Decision
    final_predicted_class = Column(String(100), nullable=True)
    is_uncertain = Column(Boolean, default=False)
    has_overlapping_sounds = Column(Boolean, default=False)
    overlapping_classes = Column(JSON, nullable=True)
    is_duplicate = Column(Boolean, default=False)

    # Event Status
    status = Column(SAEnum(EventStatus), default=EventStatus.UPLOADED)
    severity = Column(SAEnum(AlertSeverity), nullable=True)
    recommended_action = Column(Text, nullable=True)
    requires_review = Column(Boolean, default=False)

    # Visualization Paths
    waveform_path = Column(String(1000), nullable=True)
    spectrogram_path = Column(String(1000), nullable=True)

    # Processing Info
    processing_timestamp = Column(DateTime, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    is_live_capture = Column(Boolean, default=False)
    session_id = Column(String(100), nullable=True)
    segment_index = Column(Integer, nullable=True)
    segment_start_time = Column(Float, nullable=True)
    segment_end_time = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="audio_events")
    alerts = relationship("Alert", back_populates="audio_event")
    review = relationship("Review", back_populates="audio_event", uselist=False)
    segments = relationship("AudioSegment", back_populates="audio_event")


# ─────────────────────────────────────────────
# Audio Segment Model
# ─────────────────────────────────────────────
class AudioSegment(Base):
    __tablename__ = "audio_segments"

    id = Column(Integer, primary_key=True, index=True)
    audio_event_id = Column(Integer, ForeignKey("audio_events.id"), nullable=False)
    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    python_predicted_class = Column(String(100), nullable=True)
    python_confidence = Column(Float, nullable=True)
    python_confidence_scores = Column(JSON, nullable=True)

    gtm_predicted_class = Column(String(100), nullable=True)
    gtm_confidence = Column(Float, nullable=True)
    gtm_confidence_scores = Column(JSON, nullable=True)

    audio_quality = Column(SAEnum(AudioQuality), nullable=True)
    segment_file_path = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    audio_event = relationship("AudioEvent", back_populates="segments")


# ─────────────────────────────────────────────
# Alert Model
# ─────────────────────────────────────────────
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), unique=True, index=True, nullable=False)
    audio_event_id = Column(Integer, ForeignKey("audio_events.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    sound_category = Column(String(100), nullable=False)
    severity = Column(SAEnum(AlertSeverity), nullable=False)
    status = Column(SAEnum(AlertStatus), default=AlertStatus.ACTIVE)

    python_confidence = Column(Float, nullable=True)
    gtm_confidence = Column(Float, nullable=True)
    confidence_difference = Column(Float, nullable=True)
    models_agreed = Column(Boolean, nullable=True)

    message = Column(Text)
    recommended_action = Column(Text)
    consecutive_detections = Column(Integer, default=1)

    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    audio_event = relationship("AudioEvent", back_populates="alerts")
    user = relationship("User", foreign_keys=[user_id])
    acknowledged_by = relationship("User", foreign_keys=[acknowledged_by_id])


# ─────────────────────────────────────────────
# Review Model
# ─────────────────────────────────────────────
class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    audio_event_id = Column(Integer, ForeignKey("audio_events.id"), nullable=False, unique=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    original_python_class = Column(String(100))
    original_gtm_class = Column(String(100))
    original_final_class = Column(String(100))

    reviewer_decision = Column(String(100))   # confirmed/corrected class
    reviewer_comments = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    is_override = Column(Boolean, default=False)
    is_false_positive = Column(Boolean, default=False)
    is_false_negative = Column(Boolean, default=False)

    reviewed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    audio_event = relationship("AudioEvent", back_populates="review")
    reviewer = relationship("User", back_populates="reviews")


# ─────────────────────────────────────────────
# Microphone Session Model
# ─────────────────────────────────────────────
class MicrophoneSession(Base):
    __tablename__ = "microphone_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(SAEnum(MicrophoneStatus), default=MicrophoneStatus.ACTIVE)
    window_duration_seconds = Column(Float, default=2.0)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    total_windows_processed = Column(Integer, default=0)
    total_alerts_generated = Column(Integer, default=0)

    user = relationship("User", back_populates="microphone_sessions")


# ─────────────────────────────────────────────
# Model Version Tracking
# ─────────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50))  # "python_svm", "python_rf", "python_cnn", "gtm"
    version = Column(String(50))
    description = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    model_file_path = Column(String(1000), nullable=True)
    is_active = Column(Boolean, default=False)
    training_timestamp = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    hyperparameters = Column(JSON, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)


# ─────────────────────────────────────────────
# Audit Log Model
# ─────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)  # login, upload, predict, alert, review, export...
    resource_type = Column(String(100), nullable=True)  # audio_event, alert, review...
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="audit_logs")
