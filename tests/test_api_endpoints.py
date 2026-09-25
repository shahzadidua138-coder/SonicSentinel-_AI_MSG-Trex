"""
tests/test_api_endpoints.py - Automated Testing for SonicSentinel REST & Acoustic APIs
"""

import os
import sys
import io
import json
import wave
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import database


def make_test_wav(filename: str = "test.wav", duration: float = 1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        n = int(22050 * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def test_apis():
    print("==================================================")
    print("  SonicSentinel AI - Acoustic REST API Tests")
    print("==================================================")

    client = app.test_client()

    # 1. Sign in as Admin
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'Administrator'
        sess['full_name'] = 'System Administrator'

    # 2. Test /export-csv
    r_csv = client.get('/export-csv')
    assert r_csv.status_code == 200, f"Expected 200, got {r_csv.status_code}"
    assert b"Audio_ID" in r_csv.data and b"AUD-01" in r_csv.data, "CSV export missing header or seeded data"
    print("[PASS] 1. Dynamic CSV event export verified (contains seeded events).")

    # 3. Test /api/audio/upload with synthetic WAV
    wav_bytes = make_test_wav("emergency_alarm_test.wav", 1.5)
    r_upload = client.post('/api/audio/upload', data={
        'file': (io.BytesIO(wav_bytes), 'emergency_alarm_test.wav')
    }, content_type='multipart/form-data')

    assert r_upload.status_code == 200, f"Upload failed: {r_upload.data}"
    up_data = r_upload.get_json()
    assert up_data['success'] is True, "Upload JSON success is False"
    assert "audio_id" in up_data, "Missing audio_id in upload response"
    assert "quality" in up_data and "prediction" in up_data, "Missing quality or prediction in response"
    new_audio_id = up_data['audio_id']
    print(f"[PASS] 2. Multipart Audio Ingestion & Dual Classification verified (ID: {new_audio_id}).")

    # 4. Test /api/audio/live-chunk with PCM Float32 buffer
    pcm_samples = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, 33075))).astype(np.float32)
    r_live = client.post('/api/audio/live-chunk', data=pcm_samples.tobytes(), content_type='application/octet-stream')
    assert r_live.status_code == 200, f"Live chunk failed: {r_live.data}"
    live_data = r_live.get_json()
    assert live_data['success'] is True, "Live chunk failed"
    print(f"[PASS] 3. Real-time Live Microphone PCM chunk ingestion verified (Detected: {live_data['prediction']['final_class']}).")

    # 5. Test /api/audio/events
    r_events = client.get('/api/audio/events')
    assert r_events.status_code == 200
    events = r_events.get_json()
    assert len(events) >= 11, f"Expected >= 11 events, got {len(events)}"
    print(f"[PASS] 4. Acoustic Events REST API verified ({len(events)} events returned).")

    # 6. Test /api/audio/<id>
    r_detail = client.get(f'/api/audio/{new_audio_id}')
    assert r_detail.status_code == 200
    detail = r_detail.get_json()
    assert detail['id'] == new_audio_id, "Detail ID mismatch"
    print("[PASS] 5. Single Audio Event Forensic Detail API verified.")

    # 7. Test /api/alerts/active & /api/alerts/<id>/ack
    r_alerts = client.get('/api/alerts/active')
    assert r_alerts.status_code == 200
    alerts = r_alerts.get_json()
    assert len(alerts) > 0, "No active alerts found"
    first_alt_id = alerts[0]['id']

    r_ack = client.post(f'/api/alerts/{first_alt_id}/ack', json={'status': 'Acknowledged'})
    assert r_ack.status_code == 200
    assert r_ack.get_json()['new_status'] == 'Acknowledged'
    print(f"[PASS] 6. Threat Alert Center & Acknowledgement verified (Alert: {first_alt_id}).")

    # 8. Test /api/review/queue & /api/review/<id>/override
    r_rev = client.get('/api/review/queue')
    assert r_rev.status_code == 200
    revs = r_rev.get_json()
    assert len(revs) > 0, "No reviews in triage queue"
    first_rev_id = revs[0]['id']

    r_ovr = client.post(f'/api/review/{first_rev_id}/override', json={
        'final_decision': 'Machinery Fault',
        'comments': 'Verified acoustic cavitation harmonic.'
    })
    assert r_ovr.status_code == 200
    assert r_ovr.get_json()['success'] is True
    print(f"[PASS] 7. Discrepancy Triage Queue & Audited Reviewer Override verified.")

    # 9. Test /api/reports/<id>/pdf
    r_pdf = client.get(f'/api/reports/{new_audio_id}/pdf')
    assert r_pdf.status_code == 200
    assert r_pdf.mimetype == 'application/pdf'
    assert len(r_pdf.data) > 100
    print(f"[PASS] 8. Official Acoustic Incident PDF Certificate streaming verified ({len(r_pdf.data)} bytes).")

    # 10. Test /api/admin/rules/reload
    r_reload = client.post('/api/admin/rules/reload')
    assert r_reload.status_code == 200
    assert r_reload.get_json()['reloaded'] is True
    print("[PASS] 9. Hot-Reloading of Alert Rules JSON verified without server downtime.")

    print("\n==================================================")
    print("  ALL 9 REST & STREAMING ACOUSTIC APIS PASSED 100%!")
    print("==================================================")


if __name__ == '__main__':
    test_apis()
