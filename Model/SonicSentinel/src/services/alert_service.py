# ============================================================
# SonicSentinel AI - Alert Management Service
# ============================================================
"""
Alert Service managing automated alert generation, status transitions, 
acknowledgement, dismissal, and escalation workflows.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from src.extensions import db
from src.models.alert import Alert, AlertStatus, AlertSeverity
from src.models.audio_event import AudioEvent
from src.models.prediction import Prediction
from alert_rules.rule_engine import AlertRuleEngine


class AlertService:
    """
    Manages alert creation, lifecycle state changes, and escalation handling.
    """

    def __init__(self):
        self.rule_engine = AlertRuleEngine()

    def evaluate_and_create_alert(
        self,
        audio_event: AudioEvent,
        prediction: Prediction
    ) -> Tuple[Optional[Alert], Dict[str, Any]]:
        """
        Evaluates a prediction against rule engine and creates an Alert if approved.
        """
        # Fetch recent active/acknowledged alerts for deduplication check
        recent_db_alerts = Alert.query.filter_by(
            location=audio_event.location
        ).order_by(Alert.created_at.desc()).limit(10).all()

        recent_alerts_data = [
            {
                "class": a.sound_class,
                "created_at": a.created_at
            }
            for a in recent_db_alerts
        ]

        # Evaluate rules
        eval_res = self.rule_engine.evaluate_prediction(
            predicted_class=prediction.final_predicted_class,
            confidence=prediction.final_confidence,
            location=audio_event.location,
            recent_alerts=recent_alerts_data
        )

        if not eval_res["should_alert"]:
            return None, eval_res

        # Create Alert
        alert = Alert(
            audio_event_id=audio_event.id,
            prediction_id=prediction.id,
            title=eval_res["title"],
            sound_class=prediction.final_predicted_class,
            severity=eval_res["severity"],
            confidence=prediction.final_confidence,
            location=audio_event.location,
            message=eval_res["message"],
            status=AlertStatus.ACTIVE.value
        )

        db.session.add(alert)
        db.session.commit()

        eval_res["alert_id"] = alert.id
        return alert, eval_res

    def acknowledge_alert(self, alert_id: int, user_id: int, notes: Optional[str] = None) -> Alert:
        """
        Marks an alert as Acknowledged by a user.
        """
        alert = db.session.get(Alert, alert_id)
        if not alert:
            raise ValueError(f"Alert with ID {alert_id} not found.")

        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_by_id = user_id
        alert.acknowledged_at = datetime.now(timezone.utc)
        if notes:
            alert.notes = notes

        db.session.commit()
        return alert

    def dismiss_alert(self, alert_id: int, user_id: int, reason: Optional[str] = None) -> Alert:
        """
        Marks an alert as Dismissed (false positive or ignored).
        """
        alert = db.session.get(Alert, alert_id)
        if not alert:
            raise ValueError(f"Alert with ID {alert_id} not found.")

        alert.status = AlertStatus.DISMISSED.value
        alert.acknowledged_by_id = user_id
        alert.acknowledged_at = datetime.now(timezone.utc)
        if reason:
            alert.notes = f"Dismissed: {reason}"

        db.session.commit()
        return alert

    def check_and_escalate_alerts(self) -> List[Alert]:
        """
        Scans active alerts and auto-escalates critical/high alerts exceeding timeout.
        """
        active_alerts = Alert.query.filter_by(status=AlertStatus.ACTIVE.value).all()
        escalated_alerts = []

        for alert in active_alerts:
            if alert.severity in [AlertSeverity.CRITICAL.value, AlertSeverity.HIGH.value]:
                if self.rule_engine.check_escalation_needed(alert.created_at, alert.status):
                    alert.status = AlertStatus.ESCALATED.value
                    alert.notes = (alert.notes or "") + " [System Auto-Escalated: Unacknowledged alert timeout]"
                    escalated_alerts.append(alert)

        if escalated_alerts:
            db.session.commit()

        return escalated_alerts
