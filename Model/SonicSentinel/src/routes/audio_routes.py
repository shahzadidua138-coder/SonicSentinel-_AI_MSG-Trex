# ============================================================
# SonicSentinel AI - Audio API Routes
# ============================================================
"""
API Endpoints for Audio Upload, Inspection, Playback Streaming, 
and Feature Telemetry Retrieval.
"""

import os
from flask import Blueprint, request, jsonify, send_file
from flask_login import login_required, current_user

from src.models.audio_event import AudioEvent
from src.services.audio_service import AudioService
from src.services.prediction_service import PredictionService
from src.services.alert_service import AlertService
from src.services.review_service import ReviewService
from src.services.audit_service import AuditService

audio_bp = Blueprint("audio", __name__, url_prefix="/api/audio")
audio_service = AudioService()
prediction_service = PredictionService()
alert_service = AlertService()
review_service = ReviewService()


@audio_bp.route("/upload", methods=["POST"])

def upload_audio():
    """
    Handles audio file upload, preprocessing, feature extraction, 
    dual-model inference, and automated alert/review trigger.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_obj = request.files["file"]
    if file_obj.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    location = request.form.get("location", "Zone 1")
    user_id = current_user.id if current_user.is_authenticated else None

    try:
        # 1. Process & Save Audio Event
        audio_event, processed_data = audio_service.process_and_save_audio(
            file_obj=file_obj,
            location=location,
            uploaded_by_id=user_id
        )

        # 2. Run Prediction Pipeline
        prediction_record, pred_summary = prediction_service.predict_audio_event(
            audio_event_id=audio_event.id,
            feature_vector=processed_data["feature_vector"]
        )

        # 3. Evaluate Alerts
        alert_record, alert_summary = alert_service.evaluate_and_create_alert(
            audio_event=audio_event,
            prediction=prediction_record
        )

        # 4. Evaluate Review Queue Trigger
        review_task = None
        if prediction_record.is_uncertain or not prediction_record.models_agree:
            review_task = review_service.create_review_task(
                audio_event_id=audio_event.id,
                prediction_id=prediction_record.id,
                reason="Dual-model discrepancy or low confidence prediction"
            )

        # Audit Log
        AuditService.log_action(
            action="AUDIO_UPLOADED",
            user_id=user_id,
            target_entity=f"AudioEvent:{audio_event.id}",
            details={"filename": audio_event.file_name, "class": prediction_record.final_predicted_class}
        )

        return jsonify({
            "message": "Audio processed successfully",
            "audio_event": audio_event.to_dict(),
            "prediction": prediction_record.to_dict(),
            "alert": alert_record.to_dict() if alert_record else None,
            "review_flagged": review_task is not None
        }), 201

    except Exception as e:
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500


@audio_bp.route("/events", methods=["GET"])

def list_audio_events():
    """
    Lists paginated Audio Event records with optional filtering.
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    status_filter = request.args.get("status")
    quality_filter = request.args.get("quality")

    query = AudioEvent.query

    if status_filter:
        query = query.filter_by(status=status_filter)
    if quality_filter:
        query = query.filter_by(quality_assessment=quality_filter)

    pagination = query.order_by(AudioEvent.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "events": [event.to_dict() for event in pagination.items],
        "total": pagination.total,
        "page": page,
        "pages": pagination.pages
    }), 200


@audio_bp.route("/events/<int:event_id>", methods=["GET"])

def get_audio_event_detail(event_id: int):
    """
    Returns complete details of a specific Audio Event including predictions & alerts.
    """
    audio_event = AudioEvent.query.get_or_404(event_id)
    return jsonify({"audio_event": audio_event.to_dict()}), 200


@audio_bp.route("/events/<int:event_id>/stream", methods=["GET"])

def stream_audio(event_id: int):
    """
    Streams the raw audio file for browser playback.
    """
    audio_event = AudioEvent.query.get_or_404(event_id)
    if not os.path.exists(audio_event.file_path):
        return jsonify({"error": "Audio file not found on disk"}), 404

    return send_file(
        audio_event.file_path,
        mimetype="audio/wav"
    )
