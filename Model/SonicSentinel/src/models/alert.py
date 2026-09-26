# ============================================================
# SonicSentinel AI - Alert Model
# ============================================================
import enum
import uuid
from datetime import datetime, timezone
from src.extensions import db


class AlertSeverity(str, enum.Enum):
    """Alert severity levels as defined in the SRS."""
    INFORMATIONAL = "Informational"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AlertStatus(str, enum.Enum):
    """Alert lifecycle status values."""
    ACTIVE = "Active"
    ACKNOWLEDGED = "Acknowledged"
    DISMISSED = "Dismissed"
    ESCALATED = "Escalated"


class Alert(db.Model):
    """
    Stores generated alerts for detected critical / high-severity events.
    Supports acknowledgement, dismissal, and escalation workflows.
    """
    __tablename__ = "alerts"

    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.String(50), unique=True, nullable=False, index=True, default=lambda: f"ALT-{uuid.uuid4().hex[:8].upper()}")
    audio_event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=True)

    # --- Alert Details ---
    title = db.Column(db.String(150), nullable=True)
    sound_class = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(20), nullable=False, default=AlertSeverity.MEDIUM.value)
    confidence = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100), default="Zone 1")
    message = db.Column(db.String(500), nullable=False)
    recommended_action = db.Column(db.String(300), nullable=True)
    notes = db.Column(db.Text, nullable=True)

    # --- Status ---
    status = db.Column(db.String(20), nullable=False, default=AlertStatus.ACTIVE.value)
    acknowledged_by_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    acknowledged_at = db.Column(db.DateTime, nullable=True)

    # --- Timestamps ---
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "audio_event_id": self.audio_event_id,
            "prediction_id": self.prediction_id,
            "title": self.title or f"[{self.severity}] {self.sound_class}",
            "sound_class": self.sound_class,
            "severity": self.severity,
            "confidence": self.confidence,
            "location": self.location,
            "message": self.message,
            "status": self.status,
            "notes": self.notes,
            "acknowledged_by_id": self.acknowledged_by_id,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<Alert {self.alert_id} [{self.severity}] {self.sound_class}>"
