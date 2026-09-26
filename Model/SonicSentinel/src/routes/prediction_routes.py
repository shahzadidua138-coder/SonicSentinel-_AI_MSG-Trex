# ============================================================
# SonicSentinel AI - Prediction & Comparison Routes
# ============================================================
"""
API Endpoints for Model Prediction Telemetry, Dual-Model Comparison, 
and Algorithm Performance Inspection.
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required

from src.models.prediction import Prediction

prediction_bp = Blueprint("predictions", __name__, url_prefix="/api/predictions")


@prediction_bp.route("", methods=["GET"])

def list_predictions():
    """
    Lists paginated predictions with filter options (uncertainty, model agreement, algorithm).
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    uncertain_only = request.args.get("uncertain", type=bool)
    algo_filter = request.args.get("algorithm")

    query = Prediction.query

    if uncertain_only is not None:
        query = query.filter_by(is_uncertain=uncertain_only)
    if algo_filter:
        query = query.filter_by(python_algorithm=algo_filter)

    pagination = query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "predictions": [p.to_dict() for p in pagination.items],
        "total": pagination.total,
        "page": page,
        "pages": pagination.pages
    }), 200


@prediction_bp.route("/<int:pred_id>/compare", methods=["GET"])

def compare_models(pred_id: int):
    """
    Returns comparative breakdown between Python ML model and Google Teachable Machine model.
    """
    prediction = Prediction.query.get_or_404(pred_id)

    comparison = {
        "prediction_id": prediction.id,
        "audio_event_id": prediction.audio_event_id,
        "python_model": {
            "algorithm": prediction.python_algorithm,
            "version": prediction.python_model_version,
            "predicted_class": prediction.python_predicted_class,
            "confidence": prediction.python_confidence
        },
        "gtm_model": {
            "name": "Google Teachable Machine",
            "predicted_class": prediction.gtm_predicted_class,
            "confidence": prediction.gtm_confidence
        },
        "comparison_metrics": {
            "models_agree": prediction.models_agree,
            "confidence_delta": prediction.confidence_delta,
            "final_decision": prediction.final_predicted_class,
            "is_uncertain": prediction.is_uncertain,
            "latency_ms": prediction.latency_ms
        }
    }

    return jsonify(comparison), 200
