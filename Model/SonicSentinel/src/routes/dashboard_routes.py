# ============================================================
# SonicSentinel AI - Dashboard Telemetry Routes
# ============================================================
"""
API Endpoints powering the real-time Security Operations Center (SOC) dashboard:
  - Aggregate detection statistics
  - Threat class distribution breakdown
  - Dual-model agreement & latency metrics
  - Quality assessment breakdown
"""

from flask import Blueprint, jsonify
from sqlalchemy import func

from src.extensions import db
from src.models.audio_event import AudioEvent
from src.models.prediction import Prediction
from src.models.alert import Alert, AlertStatus
from src.models.review import Review

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api/dashboard")


@dashboard_bp.route("/stats", methods=["GET"])
def get_dashboard_stats():
    """
    Computes aggregate metrics for real-time dashboard visualization.
    """
    total_audio_events = AudioEvent.query.count()
    active_alerts_count = Alert.query.filter(
        Alert.status.in_([AlertStatus.ACTIVE.value, AlertStatus.ESCALATED.value])
    ).count()
    pending_reviews_count = Review.query.filter_by(status="Pending").count()

    # Model telemetry
    predictions = Prediction.query.all()
    total_preds = len(predictions)

    if total_preds > 0:
        agreed_count = sum(1 for p in predictions if p.models_agree)
        agreement_rate = round((agreed_count / total_preds) * 100.0, 2)
        avg_latency = round(sum(p.latency_ms for p in predictions if p.latency_ms) / total_preds, 2)
        avg_confidence = round(sum(p.final_confidence for p in predictions) / total_preds, 4)
    else:
        agreement_rate = 100.0
        avg_latency = 0.0
        avg_confidence = 0.0

    # Class Breakdown
    class_counts = db.session.query(
        Prediction.final_predicted_class,
        func.count(Prediction.id)
    ).group_by(Prediction.final_predicted_class).all()

    class_distribution = {cls: count for cls, count in class_counts}

    # Quality Breakdown
    quality_counts = db.session.query(
        AudioEvent.quality_assessment,
        func.count(AudioEvent.id)
    ).group_by(AudioEvent.quality_assessment).all()

    quality_distribution = {q: count for q, count in quality_counts}

    return jsonify({
        "summary": {
            "total_audio_events": total_audio_events,
            "active_alerts": active_alerts_count,
            "pending_reviews": pending_reviews_count,
            "total_predictions": total_preds,
            "dual_model_agreement_rate": agreement_rate,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence
        },
        "class_distribution": class_distribution,
        "quality_distribution": quality_distribution
    }), 200
