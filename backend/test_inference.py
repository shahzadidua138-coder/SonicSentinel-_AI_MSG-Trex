"""
SonicSentinel AI - Quick Live Inference Verification Script
"""
import sys
from pathlib import Path

# Safe encoding on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from src.services.prediction_service import PredictionService

def main():
    svc = PredictionService()
    svc.load_models()

    audio_dir = backend_dir.parent / "audio data"
    sample_files = list((audio_dir / "glass_breaking").glob("*.wav"))
    if not sample_files:
        print("[WARN] No audio sample found in audio data/glass_breaking")
        return

    sample = sample_files[0]
    print(f"\nEvaluating real audio sample: {sample.name}")
    print("-" * 55)

    res = svc.classify_audio_file(str(sample))
    decision = res.get("final_decision", {})
    py_res = res.get("python_prediction", {})
    quality = res.get("audio_quality", {})
    alert = res.get("alert_evaluation", {})

    print(f"Detected Category:   {decision.get('detected_sound_category')}")
    print(f"Sound Label:         {decision.get('detected_sound_label')}")
    print(f"Confidence Level:    {py_res.get('confidence', 0)*100:.2f}%")
    print(f"Audio Quality:       {quality.get('quality')}")
    print(f"Event Severity:      {decision.get('event_severity')}")
    print(f"Alert Status:        {decision.get('alert_status')}")
    print(f"Recommended Action:  {decision.get('recommended_action')}")
    print(f"Manual Review Reqd:  {decision.get('manual_review_required')}")
    print("-" * 55)
    print("Inference Verification Successful!")

if __name__ == "__main__":
    main()
