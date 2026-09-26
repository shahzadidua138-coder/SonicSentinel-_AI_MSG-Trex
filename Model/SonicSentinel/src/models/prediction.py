# ============================================================
# SonicSentinel AI - Prediction Model
# ============================================================
from datetime import datetime, timezone
from src.extensions import db


class Prediction(db.Model):
    """
    Stores Python model (SVM, RF, XGBoost) + GTM benchmark model predictions and comparison telemetry.
    """
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    audio_event_id = db.Column(db.Integer, db.ForeignKey("audio_events.id"), unique=True, nullable=False)

    # --- Python Model Results ---
    python_predicted_class = db.Column(db.String(50), nullable=False)
    python_confidence = db.Column(db.Float, nullable=False)
    python_model_version = db.Column(db.String(50), default="v1.0.0")
    python_algorithm = db.Column(db.String(50), default="XGBoost")

    # --- GTM Model Results ---
    gtm_predicted_class = db.Column(db.String(50), nullable=True)
    gtm_confidence = db.Column(db.Float, nullable=True)

    # --- Final Combined Decision ---
    final_predicted_class = db.Column(db.String(50), nullable=False)
    final_confidence = db.Column(db.Float, nullable=False)
    is_uncertain = db.Column(db.Boolean, default=False)
    models_agree = db.Column(db.Boolean, default=True)
    confidence_delta = db.Column(db.Float, default=0.0)
    latency_ms = db.Column(db.Float, default=25.0)

    # --- Timestamps ---
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audio_event_id": self.audio_event_id,
            "python_predicted_class": self.python_predicted_class,
            "python_confidence": self.python_confidence,
            "python_model_version": self.python_model_version,
            "python_algorithm": self.python_algorithm,
            "gtm_predicted_class": self.gtm_predicted_class,
            "gtm_confidence": self.gtm_confidence,
            "final_predicted_class": self.final_predicted_class,
            "final_confidence": self.final_confidence,
            "is_uncertain": self.is_uncertain,
            "models_agree": self.models_agree,
            "confidence_delta": self.confidence_delta,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<Prediction {self.final_predicted_class} ({self.final_confidence:.2f})>"
