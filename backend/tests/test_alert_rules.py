"""
SonicSentinel AI - Unit Tests for Alert Rules & Decision Engine
"""
import pytest
import json
from pathlib import Path
import sys

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config.settings import CRITICAL_CLASSES, SOUND_CLASSES


@pytest.fixture
def alert_rules_data():
    rules_path = backend_dir / "alert_rules" / "alert_rules.json"
    assert rules_path.exists(), f"Alert rules file missing at {rules_path}"
    with open(rules_path, "r") as f:
        return json.load(f)


def test_alert_rules_contains_mandatory_classes(alert_rules_data):
    rules = alert_rules_data["alert_rules"]
    rule_classes = [r["sound_category"] for r in rules]
    
    # Must include all 10 mandatory sound categories
    for cls in SOUND_CLASSES:
        assert cls in rule_classes, f"Missing alert rule for class: {cls}"


def test_critical_classes_severity(alert_rules_data):
    rules = alert_rules_data["alert_rules"]
    for r in rules:
        if r["sound_category"] in CRITICAL_CLASSES:
            assert r["severity"] in ["critical", "high"], (
                f"Critical class {r['sound_category']} must have critical or high severity"
            )
            assert "recommended_action" in r and len(r["recommended_action"]) > 0


def test_global_settings_structure(alert_rules_data):
    assert "global_settings" in alert_rules_data
    globals_ = alert_rules_data["global_settings"]
    assert "default_min_confidence" in globals_
    assert "uncertain_confidence_threshold" in globals_
    assert 0.0 <= globals_["default_min_confidence"] <= 1.0
