# ============================================================
# SonicSentinel AI - Data Seeding Script
# ============================================================
"""
Populates database with initial sample AudioEvents, Predictions, 
Alerts, and Review queue items using metadata from the dataset.
"""

import os
import sys
import csv
import random
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.app import create_app
from src.extensions import db
from src.models.audio_event import AudioEvent
from src.models.prediction import Prediction
from src.models.alert import Alert, AlertStatus, AlertSeverity
from src.models.review import Review, ReviewStatus
from src.config import SEVERITY_MAPPING, METADATA_CSV_PATH


def seed_data(sample_limit: int = 50):
    app = create_app()
    with app.app_context():
        print("Seeding database with sample records...")

        if not os.path.exists(METADATA_CSV_PATH):
            print(f"Metadata file {METADATA_CSV_PATH} not found. Skipping dataset seed.")
            return

        with open(METADATA_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Sample rows across different classes
        random.seed(42)
        sample_rows = random.sample(rows, min(sample_limit, len(rows)))

        locations = ["Zone 1 - Main Gate", "Zone 2 - Perimeter Fence", "Zone 3 - Loading Dock", "Zone 4 - Server Room"]

        for idx, row in enumerate(sample_rows):
            filename = row.get("filename", f"sample_{idx}.wav")
            sound_class = row.get("class", "Machinery Fault")
            split = row.get("data_split", "train")

            # Create AudioEvent
            created_at = datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 72))
            audio_event = AudioEvent(
                filename=filename,
                original_filename=filename,
                file_path=os.path.join("data", "audio_dataset", filename),
                file_format="wav",
                duration=3.0,
                sampling_rate=22050,
                channels=1,
                file_size=132300,
                sha256_hash=f"dummy_hash_{idx}_{filename[:10]}",
                is_duplicate=False,
                snr_db=round(random.uniform(15.0, 35.0), 2),
                has_clipping=False,
                audio_quality="Optimal Quality",
                status="Processed",
                uploaded_at=created_at
            )
            db.session.add(audio_event)
            db.session.flush()

            # Create Prediction
            py_conf = round(random.uniform(0.70, 0.98), 4)
            gtm_conf = round(random.uniform(0.68, 0.96), 4)
            models_agree = random.random() > 0.15

            gtm_class = sound_class if models_agree else "Background Noise"
            is_uncertain = py_conf < 0.60 or not models_agree

            prediction = Prediction(
                audio_event_id=audio_event.id,
                python_predicted_class=sound_class,
                python_confidence=py_conf,
                python_model_version="v1.0.0",
                python_algorithm="XGBoost",
                gtm_predicted_class=gtm_class,
                gtm_confidence=gtm_conf,
                final_predicted_class=sound_class,
                final_confidence=py_conf,
                is_uncertain=is_uncertain,
                models_agree=models_agree,
                confidence_delta=round(abs(py_conf - gtm_conf), 4),
                latency_ms=round(random.uniform(12.0, 45.0), 2),
                created_at=created_at
            )
            db.session.add(prediction)
            db.session.flush()

            # Create Alert if critical/high/medium
            severity = SEVERITY_MAPPING.get(sound_class, "Medium")
            if sound_class not in ["Background Noise", "Silence"] and py_conf >= 0.60:
                alert_status = random.choice([
                    AlertStatus.ACTIVE.value,
                    AlertStatus.ACKNOWLEDGED.value,
                    AlertStatus.DISMISSED.value
                ])

                alert = Alert(
                    audio_event_id=audio_event.id,
                    prediction_id=prediction.id,
                    title=f"[{severity.upper()}] Threat Detected: {sound_class}",
                    sound_class=sound_class,
                    severity=severity,
                    confidence=py_conf,
                    location=random.choice(locations),
                    message=f"Detected {sound_class} with {py_conf*100:.1f}% confidence.",
                    status=alert_status,
                    created_at=created_at
                )
                db.session.add(alert)

            # Create Review queue item if uncertain or model mismatch
            if is_uncertain:
                review = Review(
                    audio_event_id=audio_event.id,
                    prediction_id=prediction.id,
                    status=ReviewStatus.PENDING.value,
                    notes="Flagged due to dual-model agreement mismatch",
                    created_at=created_at
                )
                db.session.add(review)

        db.session.commit()
        print("Data seeding completed successfully!")


if __name__ == "__main__":
    seed_data()
