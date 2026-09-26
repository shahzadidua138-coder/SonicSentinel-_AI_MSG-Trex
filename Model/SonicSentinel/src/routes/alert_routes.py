# ============================================================
# SonicSentinel AI - Alert API Routes
# ============================================================
"""
API Endpoints for Security Alert Operations:
  - Active & historical alert feed
  - Operator Acknowledgement & Dismissal
  - Auto-escalation checks
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from src.models.alert import Alert, AlertStatus, AlertSeverity
from src.services.alert_service import AlertService
from src.services.audit_service import AuditService

alert_bp = Blueprint("alerts", __name__, url_prefix="/api/alerts")
alert_service = AlertService()


@alert_bp.route("", methods=["GET"])

def list_alerts():
    """
    Lists paginated security alerts with filtering options (severity, status, location).
    """
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    status_filter = request.args.get("status")
    severity_filter = request.args.get("severity")
    location_filter = request.args.get("location")

    query = Alert.query

    if status_filter:
        query = query.filter_by(status=status_filter)
    if severity_filter:
        query = query.filter_by(severity=severity_filter)
    if location_filter:
        query = query.filter_by(location=location_filter)

    pagination = query.order_by(Alert.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "alerts": [a.to_dict() for a in pagination.items],
        "total": pagination.total,
        "page": page,
        "pages": pagination.pages
    }), 200


@alert_bp.route("/active-summary", methods=["GET"])

def active_alert_summary():
    """
    Returns live count summary of active alerts grouped by severity.
    Trigger auto-escalation scanner prior to returning response.
    """
    alert_service.check_and_escalate_alerts()

    active_alerts = Alert.query.filter(
        Alert.status.in_([AlertStatus.ACTIVE.value, AlertStatus.ESCALATED.value])
    ).all()

    summary = {
        "total_active": len(active_alerts),
        "critical": sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL.value),
        "high": sum(1 for a in active_alerts if a.severity == AlertSeverity.HIGH.value),
        "medium": sum(1 for a in active_alerts if a.severity == AlertSeverity.MEDIUM.value),
        "low": sum(1 for a in active_alerts if a.severity == AlertSeverity.LOW.value),
        "escalated": sum(1 for a in active_alerts if a.status == AlertStatus.ESCALATED.value)
    }

    return jsonify(summary), 200


@alert_bp.route("/<int:alert_id>/acknowledge", methods=["POST"])

def acknowledge_alert(alert_id: int):
    """
    Operator endpoint to acknowledge an active alert.
    """
    data = request.get_json() or {}
    notes = data.get("notes")
    user_id = current_user.id if current_user.is_authenticated else 1

    try:
        alert = alert_service.acknowledge_alert(
            alert_id=alert_id,
            user_id=user_id,
            notes=notes
        )

        AuditService.log_action(
            action="ALERT_ACKNOWLEDGED",
            user_id=user_id,
            target_entity=f"Alert:{alert.id}",
            details={"sound_class": alert.sound_class, "severity": alert.severity}
        )

        return jsonify({
            "message": "Alert acknowledged",
            "alert": alert.to_dict()
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@alert_bp.route("/<int:alert_id>/dismiss", methods=["POST"])

def dismiss_alert(alert_id: int):
    """
    Operator endpoint to dismiss an alert as false positive or non-critical.
    """
    data = request.get_json() or {}
    reason = data.get("reason", "Operator Dismissal")
    user_id = current_user.id if current_user.is_authenticated else 1

    try:
        alert = alert_service.dismiss_alert(
            alert_id=alert_id,
            user_id=user_id,
            reason=reason
        )

        AuditService.log_action(
            action="ALERT_DISMISSED",
            user_id=user_id,
            target_entity=f"Alert:{alert.id}",
            details={"reason": reason}
        )

        return jsonify({
            "message": "Alert dismissed",
            "alert": alert.to_dict()
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
