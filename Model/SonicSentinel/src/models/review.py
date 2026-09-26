# ============================================================
# SonicSentinel AI - Review Model
# ============================================================
import enum
import uuid
from datetime import datetime, timezone
from src.extensions import db


class ReviewStatus(str, enum.Enum):
    """Review queue status values."""
    PENDING = "Pending"
    APPROVED = "Approved"
    OVERRIDDEN = "Overridden"
    REJECTED = "Rejected"


class Review(db.Model):
    """
    Manual review records for uncertain, conflicting, or low-confidence events.
    """
    __tablename__ = "reviews"

    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.String(50), unique=True, nullable=False, index=True, default=lambda: f"REV-{uuid.uuid4().hex[:8].upper()}")
    audio_event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    # --- Review Decision & Overrides ---
    status = db.Column(db.String(20), nullable=False, default=ReviewStatus.PENDING.value)
    override_sound_class = db.Column(db.String(50), nullable=True)
    is_overridden = db.Column(db.Boolean, default=False)
    flagged_for_retraining = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text, nullable=True)

    # --- Timestamps ---
    reviewed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))



    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "review_id": self.review_id,
            "audio_event_id": self.audio_event_id,
            "prediction_id": self.prediction_id,
            "reviewer_id": self.reviewer_id,
            "status": self.status,
            "override_sound_class": self.override_sound_class,
            "is_overridden": self.is_overridden,
            "flagged_for_retraining": self.flagged_for_retraining,
            "notes": self.notes,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<Review {self.review_id} [{self.status}]>"
