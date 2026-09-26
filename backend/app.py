"""
SonicSentinel AI - Flask REST API
Main application entry point with all routes.
"""
from flask import Flask, request, jsonify, send_file, g
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os, uuid, json, time, csv, io
from pathlib import Path
from datetime import datetime, timedelta
import traceback

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import (
    SECRET_KEY, JWT_SECRET_KEY, DATABASE_URL, HOST, PORT, DEBUG,
    UPLOADS_DIR, SPECTROGRAMS_DIR, WAVEFORMS_DIR, CORS_ORIGINS,
    MAX_FILE_SIZE_MB, SUPPORTED_FORMATS, SOUND_CLASSES, CLASS_LABELS,
    SEVERITY_MAP
)
from src.database.models import (
    Base, User, AudioEvent, Alert, Review, AuditLog,
    MicrophoneSession, ModelVersion,
    UserRole, EventStatus, AlertSeverity, AlertStatus,
    AudioQuality, ModelConsistency
)
from src.services.prediction_service import PredictionService
from src.utils.visualization import generate_waveform, generate_spectrogram

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

CORS(app, origins=CORS_ORIGINS)
jwt = JWTManager(app)

# Database
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Prediction Service (singleton)
pred_service = PredictionService()
try:
    pred_service.load_models()
except Exception as e:
    print(f"Warning: Models not loaded: {e}")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def db_session():
    return SessionLocal()


def log_audit(db, user_id, action, resource_type=None, resource_id=None,
              details=None, success=True, error_message=None):
    """Create an audit log entry."""
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id else None,
        details=details,
        ip_address=request.remote_addr,
        user_agent=request.headers.get("User-Agent", "")[:500],
        success=success,
        error_message=error_message,
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "SonicSentinel AI Backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": pred_service._models_loaded,
    })


# ─────────────────────────────────────────────
# Auth Routes
# ─────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    required = ["username", "email", "password", "full_name"]
    for field in required:
        if not data.get(field):
            return jsonify({"error": f"'{field}' is required"}), 400

    db = db_session()
    try:
        # Check duplicates
        existing = db.query(User).filter(
            (User.username == data["username"]) | (User.email == data["email"])
        ).first()
        if existing:
            return jsonify({"error": "Username or email already registered"}), 409

        user = User(
            user_id=f"USR-{uuid.uuid4().hex[:8].upper()}",
            username=data["username"],
            email=data["email"],
            password_hash=generate_password_hash(data["password"]),
            full_name=data["full_name"],
            role=UserRole(data.get("role", "normal_user")),
        )
        db.add(user)
        db.commit()
        log_audit(db, user.id, "register", "user", user.user_id)

        return jsonify({
            "message": "Registration successful",
            "user_id": user.user_id,
            "username": user.username,
        }), 201

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    db = db_session()
    try:
        identifier = data.get("username") or data.get("email")
        user = db.query(User).filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()

        if not user or not check_password_hash(user.password_hash, data.get("password", "")):
            log_audit(db, None, "login_failed", details={"identifier": identifier},
                      success=False, error_message="Invalid credentials")
            return jsonify({"error": "Invalid credentials"}), 401

        if not user.is_active:
            return jsonify({"error": "Account is inactive"}), 403

        user.last_login = datetime.utcnow()
        db.commit()

        token = create_access_token(
            identity=user.user_id,
            additional_claims={"role": user.role.value, "user_id": user.id}
        )
        log_audit(db, user.id, "login", "user", user.user_id)

        return jsonify({
            "access_token": token,
            "user": {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ─────────────────────────────────────────────
# Audio Upload & Classification Routes
# ─────────────────────────────────────────────
@app.route("/api/audio/upload", methods=["POST"])
@jwt_required()
def upload_audio():
    """Upload and classify audio file."""
    current_user_id = get_jwt_identity()
    db = db_session()

    try:
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Check format
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return jsonify({"error": f"Unsupported format '{ext}'. Use: {list(SUPPORTED_FORMATS)}"}), 400

        # Get user from DB
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Save file
        audio_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        safe_name = f"{audio_id}{ext}"
        file_path = UPLOADS_DIR / safe_name
        file.save(str(file_path))

        # Compute hash (duplicate detection)
        file_hash = pred_service.preprocessor.compute_file_hash(str(file_path))
        dup = db.query(AudioEvent).filter(AudioEvent.file_hash == file_hash).first()
        is_duplicate = dup is not None

        # Optional GTM result from request
        gtm_result = None
        if request.form.get("gtm_result"):
            try:
                gtm_result = json.loads(request.form["gtm_result"])
            except Exception:
                pass

        # Classification
        classification = pred_service.classify_audio_file(
            str(file_path), gtm_result=gtm_result
        )

        if not classification["success"]:
            os.remove(str(file_path))
            return jsonify({"error": "Classification failed", "details": classification}), 422

        # Generate visualizations
        py_result = classification["python_prediction"]
        quality = classification["audio_quality"]
        meta = classification["metadata"]
        final = classification["final_decision"]
        comparison = classification["model_comparison"]

        try:
            waveform_path = generate_waveform(str(file_path), audio_id, str(WAVEFORMS_DIR))
            spectrogram_path = generate_spectrogram(str(file_path), audio_id, str(SPECTROGRAMS_DIR))
        except Exception:
            waveform_path = None
            spectrogram_path = None

        # Store in DB
        event = AudioEvent(
            audio_id=audio_id,
            user_id=user.id,
            filename=safe_name,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            file_hash=file_hash,
            audio_format=ext,
            duration_seconds=meta.get("duration_seconds"),
            sample_rate=meta.get("sample_rate"),
            num_channels=meta.get("num_channels"),
            upload_timestamp=datetime.utcnow(),
            audio_quality=AudioQuality(quality.get("quality", "good")),
            rms_energy=quality.get("rms"),
            clipping_ratio=quality.get("clip_ratio"),
            snr_db=quality.get("snr_db"),
            silence_ratio=quality.get("silence_ratio"),
            quality_notes=quality.get("notes"),
            python_predicted_class=py_result["predicted_class"],
            python_confidence=py_result["confidence"],
            python_confidence_scores=py_result["confidence_scores"],
            python_model_version=py_result["model_version"],
            gtm_predicted_class=(classification.get("gtm_prediction") or {}).get("predicted_class"),
            gtm_confidence=(classification.get("gtm_prediction") or {}).get("confidence"),
            gtm_confidence_scores=(classification.get("gtm_prediction") or {}).get("confidence_scores"),
            model_consistency=ModelConsistency(
                comparison.get("consistency_status", "uncertain_result").replace("-", "_")
            ) if comparison.get("consistency_status") and comparison["consistency_status"] != "gtm_unavailable" else None,
            confidence_difference=comparison.get("confidence_difference"),
            models_agree=comparison.get("models_agree"),
            final_predicted_class=final["detected_sound_category"],
            is_uncertain=classification["uncertainty_analysis"]["is_uncertain"],
            has_overlapping_sounds=classification["uncertainty_analysis"]["has_overlapping_sounds"],
            overlapping_classes=classification["uncertainty_analysis"]["overlapping_classes"],
            is_duplicate=is_duplicate,
            status=EventStatus(final.get("event_status", "classified")),
            severity=AlertSeverity(final.get("event_severity", "informational")),
            recommended_action=final.get("recommended_action"),
            requires_review=final.get("manual_review_required", False),
            waveform_path=waveform_path,
            spectrogram_path=spectrogram_path,
            processing_timestamp=datetime.utcnow(),
            processing_time_ms=classification.get("processing_time_ms"),
            is_live_capture=False,
        )
        db.add(event)
        db.flush()

        # Generate alert if needed
        alert_eval = classification["alert_evaluation"]
        alert_id = None
        if alert_eval.get("should_alert"):
            alert = Alert(
                alert_id=f"ALT-{uuid.uuid4().hex[:10].upper()}",
                audio_event_id=event.id,
                user_id=user.id,
                sound_category=py_result["predicted_class"],
                severity=AlertSeverity(final["event_severity"]),
                status=AlertStatus.ACTIVE,
                python_confidence=py_result["confidence"],
                confidence_difference=comparison.get("confidence_difference"),
                models_agreed=comparison.get("models_agree"),
                message=f"Critical sound detected: {CLASS_LABELS.get(py_result['predicted_class'])}",
                recommended_action=alert_eval.get("recommended_action"),
                consecutive_detections=alert_eval.get("consecutive_detections", 1),
                triggered_at=datetime.utcnow(),
            )
            db.add(alert)
            db.flush()
            alert_id = alert.alert_id

        # Route to manual review if needed
        if final.get("manual_review_required"):
            event.status = EventStatus.MANUAL_REVIEW

        db.commit()
        log_audit(db, user.id, "upload", "audio_event", audio_id, details={
            "predicted_class": py_result["predicted_class"],
            "confidence": py_result["confidence"],
        })

        return jsonify({
            "audio_id": audio_id,
            "message": "Audio classified successfully",
            "is_duplicate": is_duplicate,
            "metadata": meta,
            "audio_quality": quality,
            "python_prediction": py_result,
            "cnn_prediction": classification.get("cnn_prediction"),
            "model_comparison": comparison,
            "uncertainty_analysis": classification["uncertainty_analysis"],
            "final_decision": final,
            "alert_id": alert_id,
            "waveform_url": f"/api/audio/{audio_id}/waveform" if waveform_path else None,
            "spectrogram_url": f"/api/audio/{audio_id}/spectrogram" if spectrogram_path else None,
            "processing_time_ms": classification.get("processing_time_ms"),
            "segments_processed": classification.get("segments_processed"),
        }), 200

    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


@app.route("/api/audio/<audio_id>", methods=["GET"])
@jwt_required()
def get_audio_event(audio_id):
    """Get details of a specific audio event."""
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event:
            return jsonify({"error": "Audio event not found"}), 404
        return jsonify(_serialize_event(event))
    finally:
        db.close()


@app.route("/api/audio/<audio_id>/waveform", methods=["GET"])
@jwt_required()
def get_waveform(audio_id):
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event or not event.waveform_path:
            return jsonify({"error": "Waveform not found"}), 404
        return send_file(event.waveform_path, mimetype="image/png")
    finally:
        db.close()


@app.route("/api/audio/<audio_id>/spectrogram", methods=["GET"])
@jwt_required()
def get_spectrogram(audio_id):
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event or not event.spectrogram_path:
            return jsonify({"error": "Spectrogram not found"}), 404
        return send_file(event.spectrogram_path, mimetype="image/png")
    finally:
        db.close()


# ─────────────────────────────────────────────
# Live Monitoring Routes
# ─────────────────────────────────────────────
@app.route("/api/live/classify", methods=["POST"])
@jwt_required()
def classify_live_window():
    """Classify a live microphone audio window (base64 encoded)."""
    import base64
    import numpy as np

    current_user_id = get_jwt_identity()
    data = request.get_json()

    try:
        # Accept raw audio as base64 float32 array
        audio_b64 = data.get("audio_data")
        session_id = data.get("session_id", str(uuid.uuid4()))
        gtm_result = data.get("gtm_result")

        if audio_b64:
            audio_bytes = base64.b64decode(audio_b64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        else:
            return jsonify({"error": "No audio data provided"}), 400

        result = pred_service.classify_live_window(audio_array, session_id, gtm_result)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live/session/start", methods=["POST"])
@jwt_required()
def start_live_session():
    """Start a live microphone monitoring session."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        session_id = f"SES-{uuid.uuid4().hex[:10].upper()}"
        session = MicrophoneSession(
            session_id=session_id,
            user_id=user.id,
            started_at=datetime.utcnow(),
        )
        db.add(session)
        db.commit()
        return jsonify({"session_id": session_id, "message": "Session started"})
    finally:
        db.close()


@app.route("/api/live/session/<session_id>/stop", methods=["POST"])
@jwt_required()
def stop_live_session(session_id):
    """Stop a live microphone monitoring session."""
    db = db_session()
    try:
        session = db.query(MicrophoneSession).filter(
            MicrophoneSession.session_id == session_id
        ).first()
        if session:
            session.ended_at = datetime.utcnow()
            db.commit()
        return jsonify({"message": "Session stopped", "session_id": session_id})
    finally:
        db.close()


# ─────────────────────────────────────────────
# Alerts Routes
# ─────────────────────────────────────────────
@app.route("/api/alerts", methods=["GET"])
@jwt_required()
def list_alerts():
    """List alerts with optional filters."""
    db = db_session()
    try:
        query = db.query(Alert)

        severity = request.args.get("severity")
        status = request.args.get("status")
        category = request.args.get("category")
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        if severity:
            query = query.filter(Alert.severity == AlertSeverity(severity))
        if status:
            query = query.filter(Alert.status == AlertStatus(status))
        if category:
            query = query.filter(Alert.sound_category == category)

        total = query.count()
        alerts = query.order_by(Alert.triggered_at.desc()).offset(offset).limit(limit).all()

        return jsonify({
            "total": total,
            "alerts": [_serialize_alert(a) for a in alerts]
        })
    finally:
        db.close()


@app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
@jwt_required()
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        data = request.get_json() or {}
        action = data.get("action", "acknowledge")  # acknowledge, dismiss, escalate

        if action == "acknowledge":
            alert.status = AlertStatus.ACKNOWLEDGED
        elif action == "dismiss":
            alert.status = AlertStatus.DISMISSED
        elif action == "escalate":
            alert.status = AlertStatus.ESCALATED

        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by_id = user.id
        alert.notes = data.get("notes")
        db.commit()
        log_audit(db, user.id, f"alert_{action}", "alert", alert_id)

        return jsonify({"message": f"Alert {action}d", "alert_id": alert_id})
    finally:
        db.close()


# ─────────────────────────────────────────────
# Manual Review Routes
# ─────────────────────────────────────────────
@app.route("/api/review/queue", methods=["GET"])
@jwt_required()
def get_review_queue():
    """Get list of audio events pending review."""
    db = db_session()
    try:
        events = db.query(AudioEvent).filter(
            AudioEvent.requires_review == True,
            AudioEvent.status == EventStatus.MANUAL_REVIEW
        ).order_by(AudioEvent.created_at.desc()).limit(100).all()
        return jsonify({"review_queue": [_serialize_event(e) for e in events]})
    finally:
        db.close()


@app.route("/api/review/<audio_id>", methods=["POST"])
@jwt_required()
def submit_review(audio_id):
    """Submit a manual review decision."""
    current_user_id = get_jwt_identity()
    db = db_session()
    data = request.get_json()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if user.role not in [UserRole.AUDIO_REVIEWER, UserRole.ADMINISTRATOR]:
            return jsonify({"error": "Insufficient permissions"}), 403

        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event:
            return jsonify({"error": "Event not found"}), 404

        review = Review(
            audio_event_id=event.id,
            reviewer_id=user.id,
            original_python_class=event.python_predicted_class,
            original_gtm_class=event.gtm_predicted_class,
            original_final_class=event.final_predicted_class,
            reviewer_decision=data.get("decision"),
            reviewer_comments=data.get("comments"),
            recommended_action=data.get("recommended_action"),
            is_override=data.get("is_override", False),
            is_false_positive=data.get("is_false_positive", False),
            is_false_negative=data.get("is_false_negative", False),
        )
        db.add(review)

        event.status = EventStatus.REVIEWED
        if data.get("is_override"):
            event.final_predicted_class = data.get("decision")

        db.commit()
        log_audit(db, user.id, "review", "audio_event", audio_id)
        return jsonify({"message": "Review submitted", "audio_id": audio_id})
    finally:
        db.close()


# ─────────────────────────────────────────────
# Dashboard & Analytics Routes
# ─────────────────────────────────────────────
@app.route("/api/dashboard/stats", methods=["GET"])
@jwt_required()
def dashboard_stats():
    """Get dashboard statistics."""
    db = db_session()
    try:
        total_events = db.query(AudioEvent).count()
        total_alerts = db.query(Alert).count()
        active_alerts = db.query(Alert).filter(Alert.status == AlertStatus.ACTIVE).count()
        critical_alerts = db.query(Alert).filter(Alert.severity == AlertSeverity.CRITICAL).count()
        review_pending = db.query(AudioEvent).filter(
            AudioEvent.requires_review == True,
            AudioEvent.status == EventStatus.MANUAL_REVIEW
        ).count()

        # Category breakdown
        category_counts = {}
        for cls in SOUND_CLASSES:
            count = db.query(AudioEvent).filter(
                AudioEvent.final_predicted_class == cls
            ).count()
            category_counts[cls] = count

        # Recent events
        recent = db.query(AudioEvent).order_by(
            AudioEvent.created_at.desc()
        ).limit(10).all()

        # Recent critical alerts
        recent_alerts = db.query(Alert).filter(
            Alert.severity.in_([AlertSeverity.CRITICAL, AlertSeverity.HIGH])
        ).order_by(Alert.triggered_at.desc()).limit(5).all()

        return jsonify({
            "total_events": total_events,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "pending_review": review_pending,
            "category_breakdown": category_counts,
            "recent_events": [_serialize_event(e, brief=True) for e in recent],
            "recent_critical_alerts": [_serialize_alert(a) for a in recent_alerts],
        })
    finally:
        db.close()


@app.route("/api/events", methods=["GET"])
@jwt_required()
def list_events():
    """List audio events with filtering and search."""
    db = db_session()
    try:
        query = db.query(AudioEvent)

        # Filters
        category = request.args.get("category")
        severity = request.args.get("severity")
        quality = request.args.get("quality")
        review_status = request.args.get("review_status")
        search = request.args.get("search")
        date_from = request.args.get("date_from")
        date_to = request.args.get("date_to")
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        if category:
            query = query.filter(AudioEvent.final_predicted_class == category)
        if severity:
            query = query.filter(AudioEvent.severity == AlertSeverity(severity))
        if quality:
            query = query.filter(AudioEvent.audio_quality == AudioQuality(quality))
        if review_status == "pending":
            query = query.filter(AudioEvent.requires_review == True)
        if search:
            query = query.filter(AudioEvent.original_filename.ilike(f"%{search}%"))
        if date_from:
            query = query.filter(AudioEvent.created_at >= date_from)
        if date_to:
            query = query.filter(AudioEvent.created_at <= date_to)

        total = query.count()
        events = query.order_by(AudioEvent.created_at.desc()).offset(offset).limit(limit).all()

        return jsonify({
            "total": total,
            "events": [_serialize_event(e, brief=True) for e in events]
        })
    finally:
        db.close()


# ─────────────────────────────────────────────
# Batch Audio Upload (FR v)
# ─────────────────────────────────────────────
@app.route("/api/audio/batch", methods=["POST"])
@jwt_required()
def batch_upload_audio():
    """Batch upload and classify multiple audio files."""
    current_user_id = get_jwt_identity()
    db = db_session()

    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        if user.role not in [UserRole.AUDIO_REVIEWER, UserRole.SECURITY_OPERATOR,
                              UserRole.MAINTENANCE_OPERATOR, UserRole.ADMINISTRATOR]:
            return jsonify({"error": "Batch upload requires elevated role"}), 403

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files provided"}), 400
        if len(files) > 20:
            return jsonify({"error": "Max 20 files per batch"}), 400

        results = []
        for file in files:
            if not file.filename:
                results.append({"filename": "", "error": "Empty filename"})
                continue

            ext = Path(file.filename).suffix.lower()
            if ext not in SUPPORTED_FORMATS:
                results.append({"filename": file.filename,
                                 "error": f"Unsupported format '{ext}'"})
                continue

            audio_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
            safe_name = f"{audio_id}{ext}"
            file_path = UPLOADS_DIR / safe_name
            file.save(str(file_path))

            file_hash = pred_service.preprocessor.compute_file_hash(str(file_path))
            dup = db.query(AudioEvent).filter(AudioEvent.file_hash == file_hash).first()
            is_duplicate = dup is not None

            classification = pred_service.classify_audio_file(str(file_path))

            if not classification["success"]:
                os.remove(str(file_path))
                results.append({"filename": file.filename,
                                 "error": "Classification failed",
                                 "details": classification["errors"]})
                continue

            py_result = classification["python_prediction"]
            quality = classification["audio_quality"]
            meta = classification["metadata"]
            final = classification["final_decision"]
            comparison = classification["model_comparison"]

            try:
                waveform_path = generate_waveform(str(file_path), audio_id, str(WAVEFORMS_DIR))
                spectrogram_path = generate_spectrogram(str(file_path), audio_id, str(SPECTROGRAMS_DIR))
            except Exception:
                waveform_path = None
                spectrogram_path = None

            event = AudioEvent(
                audio_id=audio_id,
                user_id=user.id,
                filename=safe_name,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size_bytes=file_path.stat().st_size,
                file_hash=file_hash,
                audio_format=ext,
                duration_seconds=meta.get("duration_seconds"),
                sample_rate=meta.get("sample_rate"),
                num_channels=meta.get("num_channels"),
                upload_timestamp=datetime.utcnow(),
                audio_quality=AudioQuality(quality.get("quality", "good")),
                rms_energy=quality.get("rms"),
                clipping_ratio=quality.get("clip_ratio"),
                snr_db=quality.get("snr_db"),
                silence_ratio=quality.get("silence_ratio"),
                quality_notes=quality.get("notes"),
                python_predicted_class=py_result["predicted_class"],
                python_confidence=py_result["confidence"],
                python_confidence_scores=py_result["confidence_scores"],
                python_model_version=py_result["model_version"],
                model_consistency=ModelConsistency(
                    comparison.get("consistency_status", "uncertain_result").replace("-", "_")
                ) if comparison.get("consistency_status") and comparison["consistency_status"] != "gtm_unavailable" else None,
                confidence_difference=comparison.get("confidence_difference"),
                models_agree=comparison.get("models_agree"),
                final_predicted_class=final["detected_sound_category"],
                is_uncertain=classification["uncertainty_analysis"]["is_uncertain"],
                has_overlapping_sounds=classification["uncertainty_analysis"]["has_overlapping_sounds"],
                overlapping_classes=classification["uncertainty_analysis"]["overlapping_classes"],
                is_duplicate=is_duplicate,
                status=EventStatus(final.get("event_status", "classified")),
                severity=AlertSeverity(final.get("event_severity", "informational")),
                recommended_action=final.get("recommended_action"),
                requires_review=final.get("manual_review_required", False),
                waveform_path=waveform_path,
                spectrogram_path=spectrogram_path,
                processing_timestamp=datetime.utcnow(),
                processing_time_ms=classification.get("processing_time_ms"),
                is_live_capture=False,
            )
            db.add(event)
            db.flush()

            alert_eval = classification["alert_evaluation"]
            if alert_eval.get("should_alert"):
                alert = Alert(
                    alert_id=f"ALT-{uuid.uuid4().hex[:10].upper()}",
                    audio_event_id=event.id,
                    user_id=user.id,
                    sound_category=py_result["predicted_class"],
                    severity=AlertSeverity(final["event_severity"]),
                    status=AlertStatus.ACTIVE,
                    python_confidence=py_result["confidence"],
                    confidence_difference=comparison.get("confidence_difference"),
                    models_agreed=comparison.get("models_agree"),
                    message=f"Critical sound detected: {CLASS_LABELS.get(py_result['predicted_class'])}",
                    recommended_action=alert_eval.get("recommended_action"),
                    triggered_at=datetime.utcnow(),
                )
                db.add(alert)
                db.flush()

            if final.get("manual_review_required"):
                event.status = EventStatus.MANUAL_REVIEW

            results.append({
                "audio_id": audio_id,
                "filename": file.filename,
                "is_duplicate": is_duplicate,
                "predicted_class": py_result["predicted_class"],
                "confidence": py_result["confidence"],
                "severity": final["event_severity"],
                "alert_generated": alert_eval.get("should_alert", False),
                "waveform_url": f"/api/audio/{audio_id}/waveform" if waveform_path else None,
                "spectrogram_url": f"/api/audio/{audio_id}/spectrogram" if spectrogram_path else None,
            })

        db.commit()
        log_audit(db, user.id, "batch_upload", "audio_event", details={"count": len(files)})
        return jsonify({"batch_results": results, "total": len(results)}), 200

    except Exception as e:
        db.rollback()
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


# ─────────────────────────────────────────────
# Downloadable Analysis Report (FR lxix)
# ─────────────────────────────────────────────
@app.route("/api/audio/<audio_id>/report", methods=["GET"])
@jwt_required()
def download_analysis_report(audio_id):
    """Generate and download a full analysis report for a specific audio event."""
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event:
            return jsonify({"error": "Audio event not found"}), 404

        review = db.query(Review).filter(Review.audio_event_id == event.id).first()

        report = {
            "report_type": "SonicSentinel AI Analysis Report",
            "generated_at": datetime.utcnow().isoformat(),
            "audio_metadata": {
                "audio_id": event.audio_id,
                "filename": event.original_filename,
                "format": event.audio_format,
                "duration_seconds": event.duration_seconds,
                "sample_rate": event.sample_rate,
                "num_channels": event.num_channels,
                "file_size_bytes": event.file_size_bytes,
                "upload_timestamp": event.upload_timestamp.isoformat() if event.upload_timestamp else None,
                "is_duplicate": event.is_duplicate,
            },
            "audio_quality": {
                "quality_level": event.audio_quality.value if event.audio_quality else None,
                "rms_energy": event.rms_energy,
                "clipping_ratio": event.clipping_ratio,
                "snr_db": event.snr_db,
                "silence_ratio": event.silence_ratio,
                "notes": event.quality_notes,
            },
            "python_prediction": {
                "predicted_class": event.python_predicted_class,
                "confidence": event.python_confidence,
                "confidence_scores": event.python_confidence_scores,
                "model_version": event.python_model_version,
            },
            "gtm_prediction": {
                "predicted_class": event.gtm_predicted_class,
                "confidence": event.gtm_confidence,
                "confidence_scores": event.gtm_confidence_scores,
                "model_version": event.gtm_model_version,
            },
            "model_comparison": {
                "models_agree": event.models_agree,
                "confidence_difference": event.confidence_difference,
                "consistency_status": event.model_consistency.value if event.model_consistency else None,
            },
            "final_decision": {
                "detected_sound_category": event.final_predicted_class,
                "event_severity": event.severity.value if event.severity else None,
                "event_status": event.status.value if event.status else None,
                "is_uncertain": event.is_uncertain,
                "has_overlapping_sounds": event.has_overlapping_sounds,
                "overlapping_classes": event.overlapping_classes,
                "recommended_action": event.recommended_action,
                "requires_review": event.requires_review,
            },
            "visualizations": {
                "waveform_url": f"/api/audio/{audio_id}/waveform" if event.waveform_path else None,
                "spectrogram_url": f"/api/audio/{audio_id}/spectrogram" if event.spectrogram_path else None,
            },
            "review_decision": {
                "reviewer_decision": review.reviewer_decision if review else None,
                "reviewer_comments": review.reviewer_comments if review else None,
                "is_override": review.is_override if review else None,
                "is_false_positive": review.is_false_positive if review else None,
                "is_false_negative": review.is_false_negative if review else None,
                "reviewed_at": review.reviewed_at.isoformat() if review and review.reviewed_at else None,
            },
            "processing_info": {
                "processing_time_ms": event.processing_time_ms,
                "processing_timestamp": event.processing_timestamp.isoformat() if event.processing_timestamp else None,
            },
        }

        # Return as downloadable JSON
        report_bytes = json.dumps(report, indent=2).encode("utf-8")
        return send_file(
            io.BytesIO(report_bytes),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"report_{audio_id}.json",
        )
    finally:
        db.close()


# ─────────────────────────────────────────────
# Data Export — CSV (FR lxx)
# ─────────────────────────────────────────────
@app.route("/api/events/export", methods=["GET"])
@jwt_required()
def export_events():
    """Export audio events as CSV. Admins only."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        events = db.query(AudioEvent).order_by(AudioEvent.created_at.desc()).limit(5000).all()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "audio_id", "original_filename", "audio_format",
            "duration_seconds", "sample_rate", "file_size_bytes",
            "upload_timestamp", "audio_quality",
            "python_predicted_class", "python_confidence",
            "gtm_predicted_class", "gtm_confidence",
            "models_agree", "confidence_difference", "consistency_status",
            "final_predicted_class", "severity", "status",
            "is_uncertain", "has_overlapping_sounds",
            "requires_review", "is_duplicate",
            "rms_energy", "snr_db", "clipping_ratio",
            "processing_time_ms", "created_at",
        ])

        for e in events:
            writer.writerow([
                e.audio_id, e.original_filename, e.audio_format,
                e.duration_seconds, e.sample_rate, e.file_size_bytes,
                e.upload_timestamp.isoformat() if e.upload_timestamp else "",
                e.audio_quality.value if e.audio_quality else "",
                e.python_predicted_class, e.python_confidence,
                e.gtm_predicted_class, e.gtm_confidence,
                e.models_agree, e.confidence_difference,
                e.model_consistency.value if e.model_consistency else "",
                e.final_predicted_class,
                e.severity.value if e.severity else "",
                e.status.value if e.status else "",
                e.is_uncertain, e.has_overlapping_sounds,
                e.requires_review, e.is_duplicate,
                e.rms_energy, e.snr_db, e.clipping_ratio,
                e.processing_time_ms,
                e.created_at.isoformat() if e.created_at else "",
            ])

        output.seek(0)
        log_audit(db, user.id, "export", "audio_events", details={"count": len(events)})
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"sonicsentinel_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        )
    finally:
        db.close()


# ─────────────────────────────────────────────
# Admin Alert Rules Configuration (FR liii)
# ─────────────────────────────────────────────
@app.route("/api/admin/alert-rules", methods=["GET"])
@jwt_required()
def get_alert_rules():
    """Get current alert rules configuration."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        rules_path = Path(__file__).resolve().parent / "alert_rules" / "alert_rules.json"
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        return jsonify(rules)
    finally:
        db.close()


@app.route("/api/admin/alert-rules", methods=["PUT"])
@jwt_required()
def update_alert_rules():
    """Update alert rules configuration. Administrators only."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        new_rules = request.get_json()
        if not new_rules or "alert_rules" not in new_rules:
            return jsonify({"error": "Invalid rules format. 'alert_rules' key required."}), 400

        rules_path = Path(__file__).resolve().parent / "alert_rules" / "alert_rules.json"
        with open(rules_path, "w", encoding="utf-8") as f:
            json.dump(new_rules, f, indent=2)

        # Reload into prediction service
        pred_service._alert_rules = pred_service._load_alert_rules()

        log_audit(db, user.id, "model_updates", "alert_rules",
                  details={"updated_rules_count": len(new_rules["alert_rules"])})
        return jsonify({"message": "Alert rules updated successfully"})
    finally:
        db.close()


# ─────────────────────────────────────────────
# Admin Anomaly Monitoring (FR lxxviii)
# ─────────────────────────────────────────────
@app.route("/api/admin/anomalies", methods=["GET"])
@jwt_required()
def get_system_anomalies():
    """Detect system-level anomalies for administrator monitoring."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        from datetime import timedelta
        recent_window = datetime.utcnow() - timedelta(hours=24)

        # Failed logins in last 24h
        failed_logins = db.query(AuditLog).filter(
            AuditLog.action == "login_failed",
            AuditLog.timestamp >= recent_window,
        ).count()

        # Duplicate files in last 24h
        duplicate_uploads = db.query(AudioEvent).filter(
            AudioEvent.is_duplicate == True,
            AudioEvent.created_at >= recent_window,
        ).count()

        # Critical alerts in last 24h
        critical_alert_count = db.query(Alert).filter(
            Alert.severity == AlertSeverity.CRITICAL,
            Alert.triggered_at >= recent_window,
        ).count()

        # Low confidence events in last 24h (possible model failure)
        low_confidence_events = db.query(AudioEvent).filter(
            AudioEvent.python_confidence < 0.40,
            AudioEvent.created_at >= recent_window,
        ).count()

        # Poor quality events
        poor_quality_events = db.query(AudioEvent).filter(
            AudioEvent.audio_quality == AudioQuality.UNUSABLE,
            AudioEvent.created_at >= recent_window,
        ).count()

        anomalies = []
        if failed_logins > 10:
            anomalies.append({"type": "repeated_failed_logins",
                               "count": failed_logins, "severity": "high"})
        if duplicate_uploads > 5:
            anomalies.append({"type": "excessive_duplicate_uploads",
                               "count": duplicate_uploads, "severity": "medium"})
        if critical_alert_count > 20:
            anomalies.append({"type": "excessive_critical_alerts",
                               "count": critical_alert_count, "severity": "high"})
        if low_confidence_events > 15:
            anomalies.append({"type": "unusual_low_confidence_spike",
                               "count": low_confidence_events, "severity": "medium"})

        return jsonify({
            "monitoring_window_hours": 24,
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "stats": {
                "failed_logins_24h": failed_logins,
                "duplicate_uploads_24h": duplicate_uploads,
                "critical_alerts_24h": critical_alert_count,
                "low_confidence_events_24h": low_confidence_events,
                "poor_quality_events_24h": poor_quality_events,
            },
        })
    finally:
        db.close()


# ─────────────────────────────────────────────
# Data Retention Configuration (FR lxxx)
# ─────────────────────────────────────────────
@app.route("/api/admin/retention", methods=["GET"])
@jwt_required()
def get_retention_policy():
    """Get current data retention configuration."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403
        return jsonify({
            "audio_retention_days": int(os.environ.get("AUDIO_RETENTION_DAYS", 90)),
            "event_retention_days": int(os.environ.get("EVENT_RETENTION_DAYS", 365)),
            "alert_retention_days": int(os.environ.get("ALERT_RETENTION_DAYS", 90)),
        })
    finally:
        db.close()


@app.route("/api/admin/retention", methods=["PUT"])
@jwt_required()
def update_retention_policy():
    """Update data retention policy and purge expired records."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        data = request.get_json() or {}
        audio_days = int(data.get("audio_retention_days", 90))
        event_days = int(data.get("event_retention_days", 365))

        cutoff = datetime.utcnow() - timedelta(days=event_days)
        expired_events = db.query(AudioEvent).filter(
            AudioEvent.status == EventStatus.CLOSED,
            AudioEvent.created_at < cutoff,
        ).all()

        purged = 0
        for ev in expired_events:
            # Remove audio file if it exists
            if ev.file_path and Path(ev.file_path).exists():
                try:
                    os.remove(ev.file_path)
                except Exception:
                    pass
            db.delete(ev)
            purged += 1

        db.commit()
        log_audit(db, user.id, "exports", "retention_policy",
                  details={"purged_events": purged, "event_retention_days": event_days})
        return jsonify({
            "message": "Retention policy applied",
            "purged_events": purged,
            "audio_retention_days": audio_days,
            "event_retention_days": event_days,
        })
    finally:
        db.close()


# ─────────────────────────────────────────────
# Visualization Routes
# ─────────────────────────────────────────────
@app.route("/api/audio/<audio_id>/visualize", methods=["GET"])
@jwt_required()
def visualize_audio(audio_id):
    """Regenerate and return visualization URLs."""
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event:
            return jsonify({"error": "Event not found"}), 404

        result = {}
        if event.file_path and Path(event.file_path).exists():
            waveform_path = generate_waveform(event.file_path, audio_id, str(WAVEFORMS_DIR))
            spectrogram_path = generate_spectrogram(event.file_path, audio_id, str(SPECTROGRAMS_DIR))
            event.waveform_path = waveform_path
            event.spectrogram_path = spectrogram_path
            db.commit()
            result["waveform_url"] = f"/api/audio/{audio_id}/waveform"
            result["spectrogram_url"] = f"/api/audio/{audio_id}/spectrogram"

        return jsonify(result)
    finally:
        db.close()


# ─────────────────────────────────────────────
# Audio File Streaming/Playback (FR lviii)
# ─────────────────────────────────────────────
@app.route("/api/audio/<audio_id>/play", methods=["GET"])
@jwt_required()
def stream_audio(audio_id):
    """Stream/playback audio file for reviewer."""
    db = db_session()
    try:
        event = db.query(AudioEvent).filter(AudioEvent.audio_id == audio_id).first()
        if not event or not event.file_path:
            return jsonify({"error": "Audio not found"}), 404
        if not Path(event.file_path).exists():
            return jsonify({"error": "Audio file missing from storage"}), 404
        mimetype_map = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                        ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
        mime = mimetype_map.get(event.audio_format, "audio/wav")
        return send_file(event.file_path, mimetype=mime, as_attachment=False)
    finally:
        db.close()


# ─────────────────────────────────────────────
# Helper Serializers
# ─────────────────────────────────────────────
def _serialize_event(event: AudioEvent, brief: bool = False) -> dict:
    base = {
        "audio_id": event.audio_id,
        "filename": event.original_filename,
        "status": event.status.value if event.status else None,
        "severity": event.severity.value if event.severity else None,
        "audio_quality": event.audio_quality.value if event.audio_quality else None,
        "final_predicted_class": event.final_predicted_class,
        "python_predicted_class": event.python_predicted_class,
        "python_confidence": event.python_confidence,
        "is_uncertain": event.is_uncertain,
        "requires_review": event.requires_review,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }
    if not brief:
        base.update({
            "python_confidence_scores": event.python_confidence_scores,
            "gtm_predicted_class": event.gtm_predicted_class,
            "gtm_confidence": event.gtm_confidence,
            "models_agree": event.models_agree,
            "confidence_difference": event.confidence_difference,
            "duration_seconds": event.duration_seconds,
            "sample_rate": event.sample_rate,
            "rms_energy": event.rms_energy,
            "snr_db": event.snr_db,
            "is_duplicate": event.is_duplicate,
            "recommended_action": event.recommended_action,
            "has_overlapping_sounds": event.has_overlapping_sounds,
            "overlapping_classes": event.overlapping_classes,
            "processing_time_ms": event.processing_time_ms,
        })
    return base


def _serialize_alert(alert: Alert) -> dict:
    return {
        "alert_id": alert.alert_id,
        "sound_category": alert.sound_category,
        "severity": alert.severity.value if alert.severity else None,
        "status": alert.status.value if alert.status else None,
        "message": alert.message,
        "recommended_action": alert.recommended_action,
        "python_confidence": alert.python_confidence,
        "consecutive_detections": alert.consecutive_detections,
        "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else None,
    }


# ─────────────────────────────────────────────
# Admin Analytics Extended (FR lxviii)
# ─────────────────────────────────────────────
@app.route("/api/admin/analytics", methods=["GET"])
@jwt_required()
def admin_analytics():
    """Extended analytics for administrator: false positives, model disagreement, alert response."""
    current_user_id = get_jwt_identity()
    db = db_session()
    try:
        user = db.query(User).filter(User.user_id == current_user_id).first()
        if not user or user.role != UserRole.ADMINISTRATOR:
            return jsonify({"error": "Administrator access required"}), 403

        total_events = db.query(AudioEvent).count()
        total_reviewed = db.query(Review).count()
        false_positives = db.query(Review).filter(Review.is_false_positive == True).count()
        false_negatives = db.query(Review).filter(Review.is_false_negative == True).count()
        model_disagreements = db.query(AudioEvent).filter(AudioEvent.models_agree == False).count()
        poor_quality = db.query(AudioEvent).filter(
            AudioEvent.audio_quality.in_([AudioQuality.POOR, AudioQuality.UNUSABLE])
        ).count()
        avg_confidence = db.query(AudioEvent).with_entities(
            AudioEvent.python_confidence
        ).all()
        avg_conf_val = (
            sum(c[0] for c in avg_confidence if c[0] is not None) / max(len(avg_confidence), 1)
        )

        # Alert response: acknowledged/dismissed vs active
        acknowledged_alerts = db.query(Alert).filter(
            Alert.status.in_([AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED])
        ).count()
        active_alerts = db.query(Alert).filter(Alert.status == AlertStatus.ACTIVE).count()

        # Confidence distribution buckets
        conf_distribution = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
        for (conf,) in avg_confidence:
            if conf is None:
                continue
            if conf < 0.25:
                conf_distribution["0-25%"] += 1
            elif conf < 0.50:
                conf_distribution["25-50%"] += 1
            elif conf < 0.75:
                conf_distribution["50-75%"] += 1
            else:
                conf_distribution["75-100%"] += 1

        return jsonify({
            "total_events": total_events,
            "total_reviewed": total_reviewed,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "model_disagreements": model_disagreements,
            "poor_quality_recordings": poor_quality,
            "average_python_confidence": round(avg_conf_val, 4),
            "confidence_distribution": conf_distribution,
            "alert_response": {
                "active": active_alerts,
                "acknowledged_resolved": acknowledged_alerts,
            },
        })
    finally:
        db.close()


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB} MB"}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 SonicSentinel AI Backend Starting...")
    print(f"   Host: {HOST}:{PORT}")
    print(f"   Debug: {DEBUG}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
