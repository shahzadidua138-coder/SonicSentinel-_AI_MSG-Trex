# ============================================================
# SonicSentinel AI - Alert Rule Engine
# ============================================================
"""
Rule Engine for Automated Alert Generation:
  - Severity mapping per sound event class
  - Minimum confidence filtering
  - Background/silence suppression
  - Alert deduplication (cooldown window)
  - Escalation rule logic
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from src.config import ALERT_CONFIG, SEVERITY_MAPPING


class AlertRuleEngine:
    """
    Evaluates predictions against security rules to decide if an Alert should be generated.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ALERT_CONFIG
        self.severity_map = SEVERITY_MAPPING

    def evaluate_prediction(
        self,
        predicted_class: str,
        confidence: float,
        location: str = "Zone 1",
        recent_alerts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates whether a prediction should generate an alert.

        Args:
            predicted_class: Predicted sound class name
            confidence: Model confidence score (0.0 to 1.0)
            location: Device/Zone location identifier
            recent_alerts: List of recent alerts for deduplication check

        Returns:
            Dict containing:
              - should_alert: bool
              - severity: str ('Critical', 'High', 'Medium', 'Low')
              - suppress_reason: Optional[str]
              - title: str
              - message: str
        """
        # 1. Background Noise / Silence Suppression
        if predicted_class in ["Background Noise", "Silence", "Normal Ambient"]:
            return {
                "should_alert": False,
                "severity": "Low",
                "suppress_reason": f"Class '{predicted_class}' is normal ambient sound.",
                "title": "",
                "message": ""
            }

        # 2. Minimum Confidence Filtering
        min_conf = self.config.get("min_confidence_threshold", 0.60)
        if confidence < min_conf:
            return {
                "should_alert": False,
                "severity": "Low",
                "suppress_reason": f"Confidence ({confidence:.2f}) below minimum threshold ({min_conf}).",
                "title": "",
                "message": ""
            }

        # 3. Determine Severity Level
        severity = self.severity_map.get(predicted_class, "Medium")

        # 4. Deduplication / Cooldown Check
        cooldown_sec = self.config.get("deduplication_window_seconds", 30)
        if recent_alerts:
            now = datetime.now(timezone.utc)
            for prev_alert in recent_alerts:
                prev_class = prev_alert.get("class")
                prev_time = prev_alert.get("created_at")

                if prev_class == predicted_class and prev_time:
                    if isinstance(prev_time, str):
                        prev_dt = datetime.fromisoformat(prev_time.replace("Z", "+00:00"))
                    else:
                        prev_dt = prev_time

                    if (now - prev_dt).total_seconds() < cooldown_sec:
                        return {
                            "should_alert": False,
                            "severity": severity,
                            "suppress_reason": f"Duplicate event '{predicted_class}' within cooldown window ({cooldown_sec}s).",
                            "title": "",
                            "message": ""
                        }

        # 5. Alert Trigger Approved
        title = f"[{severity.upper()}] Threat Detected: {predicted_class}"
        message = (
            f"Detected {predicted_class} with {confidence * 100:.1f}% confidence "
            f"at location {location}."
        )

        return {
            "should_alert": True,
            "severity": severity,
            "suppress_reason": None,
            "title": title,
            "message": message
        }

    def check_escalation_needed(self, alert_created_at: datetime, status: str) -> bool:
        """
        Checks if an unacknowledged critical/high alert has exceeded the auto-escalation timeout.
        """
        if status != "Active":
            return False

        timeout_min = self.config.get("auto_escalate_minutes", 15)
        now = datetime.now(timezone.utc)
        
        if alert_created_at.tzinfo is None:
            alert_created_at = alert_created_at.replace(tzinfo=timezone.utc)

        return (now - alert_created_at) > timedelta(minutes=timeout_min)
