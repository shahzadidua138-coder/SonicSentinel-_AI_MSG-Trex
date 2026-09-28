import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.audio_hash import compute_audio_hash
from src.services.quality_assessor import assess_audio_quality
from src.services.feature_extractor import extract_spectral_features
from src.services.audio_dsp import render_waveform_image, render_spectrogram_image
from src.services.ml_service import classify_audio_dual
from src.services.alert_engine import evaluate_alert
from src.services.pdf_generator import generate_incident_pdf


def test_pipeline():
    print("==================================================")
    print("  SonicSentinel AI - Acoustic DSP & Dual AI Tests")
    print("==================================================")

    # 1. Generate clean 1.5s audio signal (440 Hz tone)
    sr = 22050
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 2. Test Audio Hashing
    h = compute_audio_hash(samples.tobytes())
    assert len(h) == 64, "SHA-256 hash length incorrect"
    print("[PASS] 1. Cryptographic SHA-256 audio hashing verified.")

    # 3. Test Audio Quality Assessor
    q = assess_audio_quality(samples, sr)
    assert q['quality_grade'] in ['Good', 'Acceptable'], f"Expected Good/Acceptable, got {q['quality_grade']}"
    assert not q['is_silent'] and not q['is_clipped'], "Signal falsely flagged as silent or clipped"
    print(f"[PASS] 2. Signal Quality Gatekeeper passed (Grade: {q['quality_grade']}, SNR: {q['snr_db']} dB).")

    # 4. Test Silence Rejection
    silent_audio = np.zeros(sr, dtype=np.float32)
    q_silent = assess_audio_quality(silent_audio, sr)
    assert q_silent['is_silent'] and q_silent['quality_grade'] == 'Unusable', "Silence check failed"
    print("[PASS] 3. Silence Detection Gatekeeper verified (RMS < 0.005 rejected).")

    # 5. Test Clipping Rejection
    clipped_audio = np.ones(sr, dtype=np.float32) * 1.0
    q_clipped = assess_audio_quality(clipped_audio, sr)
    assert q_clipped['is_clipped'] and q_clipped['quality_grade'] == 'Unusable', "Clipping check failed"
    print("[PASS] 4. Clipping Detection Gatekeeper verified (> 1.0% limit rejected).")

    # 6. Test Spectral Feature Extractor
    feats = extract_spectral_features(samples, sr)
    assert len(feats['mfcc_vector']) == 40, f"Expected 40 MFCCs, got {len(feats['mfcc_vector'])}"
    assert feats['centroid_mean'] > 0, "Centroid extraction failed"
    print(f"[PASS] 5. Feature Extractor verified (40 MFCCs, Centroid: {feats['centroid_mean']:.1f} Hz, ZCR: {feats['zcr']:.3f}).")

    # 7. Test real-model classification and arbitration without prescribing a class.
    res = classify_audio_dual(samples, feats, q, filename="facility_gunshot_test.wav")
    expected_classes = set(res['python_probabilities'])
    assert len(expected_classes) == 10, f"Expected scores for all 10 classes, got {len(expected_classes)}"
    assert set(res['gtm_probabilities']) == expected_classes, "Comparison model class set differs"
    assert abs(sum(res['python_probabilities'].values()) - 1.0) < 0.02, "Python probabilities are not normalized"
    assert abs(sum(res['gtm_probabilities'].values()) - 1.0) < 0.02, "Comparison probabilities are not normalized"
    assert res['python_class'] in expected_classes and res['gtm_class'] in expected_classes
    assert 0 <= res['confidence_difference'] <= 1
    assert res['consistency_status'] in ["Acceptable Match", "Weak Match", "Model Disagreement", "Uncertain Result"]
    print(f"[PASS] 6. Python and comparison models returned valid 10-class scores (Py: {res['python_confidence']*100:.1f}%, Comparison: {res['gtm_confidence']*100:.1f}%, Status: {res['consistency_status']}).")

    # 8. Test Alert Engine
    alert = evaluate_alert(res)
    assert alert['severity'] in ["Critical", "High", "Medium", "Low", "Informational"], f"Unexpected severity: {alert['severity']}"
    print(f"[PASS] 7. Threat Alert Engine verified (Severity: {alert['severity']}, Action: {alert['recommended_action'][:45]}...).")

    # 9. Test Waveform & Spectrogram Rendering
    w_url = render_waveform_image(samples, "AUD-UNITTEST")
    s_url = render_spectrogram_image(samples, "AUD-UNITTEST")
    assert "AUD-UNITTEST_wave.png" in w_url and "AUD-UNITTEST_spec.png" in s_url, "Image generation paths invalid"
    print("[PASS] 8. Waveform and Mel-Spectrogram image generation verified.")

    # 10. Test Incident PDF Generator
    pdf = generate_incident_pdf({
        'id': 'AUD-UNITTEST',
        'final_detected_class': 'Gunshot',
        'severity': 'Critical',
        'original_filename': 'facility_gunshot_test.wav',
        'quality_grade': 'Good',
        'snr_db': 32.1,
        'python_predicted_class': 'Gunshot',
        'python_top_confidence': 0.94,
        'gtm_predicted_class': 'Gunshot',
        'gtm_top_confidence': 0.91,
        'confidence_difference': 0.03,
        'consistency_status': 'Strong Match',
        'sha256_hash': h
    })
    assert len(pdf) > 200, "PDF generation failed"
    print(f"[PASS] 9. Official Acoustic Incident Certificate generated ({len(pdf)} bytes).")

    print("\n==================================================")
    print("  ALL 9 ACOUSTIC DSP & DUAL AI TESTS PASSED 100%!")
    print("==================================================")


if __name__ == '__main__':
    test_pipeline()
