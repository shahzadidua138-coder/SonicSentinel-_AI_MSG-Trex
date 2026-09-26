# ============================================================
# SonicSentinel AI - Audit Log Model
# ============================================================
from datetime import datetime, timezone
from src.extensions import db


class AuditLog(db.Model):
    """
    Complete audit trail recording all significant user actions.

    Tracks: logins, uploads, microphone sessions, predictions, alerts,
    reviews, overrides, exports, model updates, etc. (SRS req lxxvi)
    """
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    action = db.Column(db.String(50), nullable=False, index=True)
    # e.g., "login", "upload", "predict", "alert_generated", "review", "override", "export"
    resource_type = db.Column(db.String(50), nullable=True)
    resource_id = db.Column(db.String(50), nullable=True)
    details = db.Column(db.JSON, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    timestamp = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by user {self.user_id}>"
