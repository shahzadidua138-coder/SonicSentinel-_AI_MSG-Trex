# ============================================================
# SonicSentinel AI - Model Management Routes
# ============================================================
"""
API Endpoints for ML Model Version Tracking, Training Status, 
and Algorithm Performance Metrics.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user

from src.models.model_version import ModelVersion
from src.models.user import UserRole
from src.services.audit_service import AuditService

model_bp = Blueprint("models", __name__, url_prefix="/api/models")


@model_bp.route("/versions", methods=["GET"])

def list_model_versions():
    """
    Lists all registered model versions (SVM, Random Forest, XGBoost, GTM).
    """
    versions = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    return jsonify({
        "model_versions": [v.to_dict() for v in versions]
    }), 200


@model_bp.route("/active", methods=["GET"])

def get_active_model_info():
    """
    Returns telemetry for currently active production Python ML model and GTM model.
    """
    active_py_model = ModelVersion.query.filter_by(is_active=True, framework="Python-Scikit/XGBoost").first()
    active_gtm_model = ModelVersion.query.filter_by(is_active=True, framework="Google Teachable Machine").first()

    return jsonify({
        "python_model": active_py_model.to_dict() if active_py_model else {
            "name": "SonicSentinel XGBoost Classifier",
            "version": "v1.0.0",
            "algorithms": ["SVM", "Random Forest", "XGBoost"],
            "status": "Ready"
        },
        "gtm_model": active_gtm_model.to_dict() if active_gtm_model else {
            "name": "Google Teachable Machine Benchmark",
            "version": "v1.0.0",
            "status": "Ready"
        }
    }), 200
