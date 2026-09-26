# ============================================================
# SonicSentinel AI - Model Version Tracking Model
# ============================================================
from datetime import datetime, timezone
from src.extensions import db


class ModelVersion(db.Model):
    """
    Tracks trained versions of Python ML models (SVM, Random Forest, XGBoost)
    and Google Teachable Machine benchmark models.
    """
    __tablename__ = "model_versions"

    id = db.Column(db.Integer, primary_key=True)
    version_name = db.Column(db.String(50), nullable=False, default="v1.0.0")
    algorithm = db.Column(db.String(50), nullable=False)  # "SVM", "Random Forest", "XGBoost", "GTM"
    framework = db.Column(db.String(50), default="Python-Scikit/XGBoost")
    features_dimensionality = db.Column(db.Integer, default=373)
    is_active = db.Column(db.Boolean, default=True)

    # --- Performance Telemetry ---
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)

    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version_name": self.version_name,
            "algorithm": self.algorithm,
            "framework": self.framework,
            "features_dimensionality": self.features_dimensionality,
            "is_active": self.is_active,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self) -> str:
        return f"<ModelVersion {self.algorithm} {self.version_name} [Active={self.is_active}]>"
