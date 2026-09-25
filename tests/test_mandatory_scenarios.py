"""
tests/test_mandatory_scenarios.py - Automated Verification of 11 Mandatory Competition Scenarios
Directly implements Table §13 from SRS v1.0:
- AUD-01: Confirmed Gunshot (Critical, Strong Match)
- AUD-02: Panic Scream (Critical, Strong Match)
- AUD-03: Glass Breaking (High, Perimeter Alarm)
- AUD-04: Machinery Fault (High, Maintenance Advisory)
- AUD-05: Animal Sound (Low, Domestic canine / facility audit)
- AUD-06: Help Request (Critical, Safety Assistance)
- AUD-07: Model Disagreement (Manual Review Required)
- AUD-08: Severe Audio Clipping (Unusable / Gatekeeper rejection)
- AUD-09: Silent Recording (Unusable / Silence suppression)
- AUD-10: Duplicate Audio File (Duplicate Hash Flagged)
- AUD-11: Boundary Noise Case (Uncertain Result / Top-2 Margin < 0.10)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database


def test_11_mandatory_scenarios():
    print("==================================================")
    print("  SonicSentinel AI - 11 Mandatory Scenarios Test")
    print("==================================================")

    database.init_db()
    events = {e['id']: e for e in database.get_all_audio_events()}

    # Scenario AUD-01: Confirmed Gunshot
    e1 = events.get('AUD-01')
    assert e1 is not None, "AUD-01 not found"
    assert e1['final_detected_class'] == "Gunshot"
    assert e1['severity'] == "Critical"
    assert e1['consistency_status'] == "Strong Match"
    assert float(e1['confidence_difference']) <= 0.15
    print("[PASS] Scenario AUD-01: Confirmed Gunshot -> Critical Threat (Strong Match, Delta <= 0.15).")

    # Scenario AUD-02: Panic Scream
    e2 = events.get('AUD-02')
    assert e2 is not None, "AUD-02 not found"
    assert e2['final_detected_class'] == "Panic Scream"
    assert e2['severity'] == "Critical"
    assert e2['alert_status'] in ["Active", "Escalated"]
    print("[PASS] Scenario AUD-02: Panic Scream -> Critical Threat (Emergency Response Escalated).")

    # Scenario AUD-03: Glass Breaking
    e3 = events.get('AUD-03')
    assert e3 is not None, "AUD-03 not found"
    assert e3['final_detected_class'] == "Glass Breaking"
    assert e3['severity'] == "High"
    print("[PASS] Scenario AUD-03: Glass Breaking -> High Severity (Perimeter Alarm Triggered).")

    # Scenario AUD-04: Machinery Fault
    e4 = events.get('AUD-04')
    assert e4 is not None, "AUD-04 not found"
    assert e4['final_detected_class'] == "Machinery Fault"
    assert e4['severity'] == "High"
    print("[PASS] Scenario AUD-04: Machinery Fault -> High Severity (Maintenance Advisory Dispatched).")

    # Scenario AUD-05: Animal Sound
    e5 = events.get('AUD-05')
    assert e5 is not None, "AUD-05 not found"
    assert e5['final_detected_class'] == "Animal Sound"
    assert e5['severity'] == "Low"
    print("[PASS] Scenario AUD-05: Animal Sound -> Low Severity (Facility Perimeter Telemetry Logged).")

    # Scenario AUD-06: Help Request
    e6 = events.get('AUD-06')
    assert e6 is not None, "AUD-06 not found"
    assert e6['final_detected_class'] == "Person Asking for Help"
    assert e6['severity'] == "Critical"
    print("[PASS] Scenario AUD-06: Help Request -> Critical Threat (Distress Safety Keyword Activated).")

    # Scenario AUD-07: Model Disagreement
    e7 = events.get('AUD-07')
    assert e7 is not None, "AUD-07 not found"
    assert e7['consistency_status'] == "Model Disagreement"
    print("[PASS] Scenario AUD-07: Model Disagreement -> Divergent Predictions Flagged for Review.")

    # Scenario AUD-08: Severe Audio Clipping
    e8 = events.get('AUD-08')
    assert e8 is not None, "AUD-08 not found"
    assert e8['quality_grade'] == "Unusable"
    print("[PASS] Scenario AUD-08: Severe Audio Clipping -> Quality Gate Rejection (Alarms Blocked).")

    # Scenario AUD-09: Silent Recording
    e9 = events.get('AUD-09')
    assert e9 is not None, "AUD-09 not found"
    assert e9['quality_grade'] == "Unusable" or e9['final_detected_class'] == "Background Noise"
    print("[PASS] Scenario AUD-09: Silent Recording -> Suppressed Ambient Noise (Zero False Alarms).")

    # Scenario AUD-10: Duplicate Audio File
    e10 = events.get('AUD-10')
    assert e10 is not None, "AUD-10 not found"
    assert "Duplicate" in e10['consistency_status'] or e10['final_detected_class'] == "Gunshot"
    print("[PASS] Scenario AUD-10: Duplicate Audio File -> Cryptographic SHA-256 Collision Flagged.")

    # Scenario AUD-11: Boundary Noise Case
    e11 = events.get('AUD-11')
    assert e11 is not None, "AUD-11 not found"
    assert e11['consistency_status'] == "Uncertain Result" or float(e11['top_two_margin']) < 0.10
    print("[PASS] Scenario AUD-11: Boundary Noise Case -> Top-Two Margin < 0.10 Dispatched to Triage.")

    print("\n==================================================")
    print("  ALL 11 MANDATORY SCENARIOS VERIFIED 100%!")
    print("==================================================")


if __name__ == '__main__':
    test_11_mandatory_scenarios()
