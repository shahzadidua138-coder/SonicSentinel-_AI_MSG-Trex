"""
src/services/alert_engine.py - Threat Evaluation & Alert Dispatch Engine
Implements SRS §9:
- Severity hierarchy mapping (Critical, High, Medium, Low, Informational)
- Multi-window repeated confirmation tracking for live sliding windows
- Hot-reloadable alert policies via config/alert_rules.json
"""

import os
import json
from config.settings import (
    CONFIG_FOLDER,
    CONSECUTIVE_WINDOWS_REQUIRED,
    CRITICAL_INSTANT_CONFIDENCE
)

RULES_PATH = os.path.join(CONFIG_FOLDER, 'alert_rules.json')

# Sliding window history cache: sensor_id -> list of detected classes
_WINDOW_HISTORY = {}


def load_alert_rules() -> dict:
    """Loads alert rules JSON configuration from disk."""
    if os.path.exists(RULES_PATH):
        try:
            with open(RULES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def evaluate_alert(prediction_result: dict, session_id: str = "default_stream") -> dict:
    """
    Evaluates classification result against alert policies:
    1. Determines threat severity (Critical, High, Medium, Low, Informational).
    2. Checks repeated confirmation across sliding windows.
    3. Formulates immediate response actions and escalation path.
    """
    final_class = prediction_result.get("final_class", "Background Noise")
    py_conf = prediction_result.get("python_confidence", 0.5)
    gtm_conf = prediction_result.get("gtm_confidence", 0.5)
    consistency = prediction_result.get("consistency_status", "Weak Match")
    quality = prediction_result.get("quality_grade", "Good")

    rules = load_alert_rules()
    category_rule = rules.get("categories", {}).get(final_class, {
        "severity": "Informational",
        "immediate_action": "Standard acoustic event logged.",
        "confirmation_rule": "Standard logging.",
        "color": "#64748b",
        "icon": "fas fa-info-circle",
        "escalation_target": "System Log"
    })

    severity = category_rule.get("severity", "Informational")
    recommended_action = category_rule.get("immediate_action", "Log event telemetry.")

    # Multi-window sliding confirmation tracking
    if session_id not in _WINDOW_HISTORY:
        _WINDOW_HISTORY[session_id] = []
    
    _WINDOW_HISTORY[session_id].append(final_class)
    if len(_WINDOW_HISTORY[session_id]) > 5:
        _WINDOW_HISTORY[session_id].pop(0)

    recent_matches = sum(1 for c in _WINDOW_HISTORY[session_id] if c == final_class)
    is_repeated = recent_matches >= CONSECUTIVE_WINDOWS_REQUIRED

    # Instant confirmation bypass if both models agree with >= 85% confidence
    instant_bypass = (py_conf >= CRITICAL_INSTANT_CONFIDENCE and gtm_conf >= CRITICAL_INSTANT_CONFIDENCE and consistency in ["Strong Match", "Acceptable Match"])

    is_confirmed = is_repeated or instant_bypass

    # Quality suppression rule: Unusable audio blocks automated critical alarms
    if quality == "Unusable":
        alert_status = "Closed"
        alert_triggered = False
        recommended_action = "QUALITY GATE ACTIVE: Signal integrity failed. Automated alarm blocked to prevent false panic."
    elif consistency in ["Model Disagreement", "Uncertain Result"]:
        alert_status = "Active"
        alert_triggered = True
        recommended_action = f"MANUAL FORENSIC AUDIT REQUIRED: {consistency}. Routed to Audio Reviewer queue."
    elif severity in ["Critical", "High"]:
        if is_confirmed:
            alert_status = "Active"
            alert_triggered = True
        else:
            alert_status = "Pending Confirmation"
            alert_triggered = False
            recommended_action = f"CONFIRMING THREAT: Awaiting {CONSECUTIVE_WINDOWS_REQUIRED} consecutive window verification."
    else:
        alert_status = "Closed"
        alert_triggered = False

    return {
        "threat_category": final_class,
        "severity": severity,
        "alert_triggered": alert_triggered,
        "status": alert_status,
        "is_confirmed_repeated": bool(is_confirmed),
        "recommended_action": recommended_action,
        "color": category_rule.get("color", "#06b6d4"),
        "icon": category_rule.get("icon", "fas fa-bell"),
        "escalation_target": category_rule.get("escalation_target", "Security Operations")
    }


def reload_rules() -> bool:
    """Hot-reloads alert rules into memory."""
    global _WINDOW_HISTORY
    _WINDOW_HISTORY.clear()
    rules = load_alert_rules()
    return bool(rules)
