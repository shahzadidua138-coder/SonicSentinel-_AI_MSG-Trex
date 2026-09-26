# ============================================================
# SonicSentinel AI - Audio Processing Service
# ============================================================
"""
Service layer handling audio upload, validation, duplicate checking, 
preprocessing, feature extraction, and database persistence.
"""

import os
from typing import Dict, Any, Tuple, Optional
from werkzeug.datastructures import FileStorage

from src.extensions import db
from src.models.audio_event import AudioEvent
from audio_preprocessing.preprocessor import AudioPreprocessor
from feature_extraction.extractor import FeatureExtractor
from src.config import UPLOAD_FOLDER


class AudioService:
    """
    Manages the full lifecycle of incoming audio files prior to model scoring.
    """

    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()

    def process_and_save_audio(
        self,
        file_obj: FileStorage,
        location: str = "Zone 1",
        uploaded_by_id: Optional[int] = None
    ) -> Tuple[AudioEvent, Dict[str, Any]]:
        """
        Processes an uploaded audio file:
          1. Saves file to upload folder
          2. Assesses audio quality & computes SHA-256 hash
          3. Checks for duplicate audio file
          4. Runs audio preprocessor (load, mono, resample, normalize, trim)
          5. Extracts feature vector (373-dim)
          6. Persists AudioEvent model record

        Returns:
            Tuple of (AudioEvent ORM model, Processed Data Dictionary)
        """
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = file_obj.filename or "audio.wav"
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file_obj.save(temp_path)

        # 1. Preprocess & Quality Assessment
        processed_data = self.preprocessor.preprocess(temp_path)

        # 2. Check for duplicate audio
        file_hash = processed_data["sha256_hash"]
        existing = AudioEvent.query.filter_by(sha256_hash=file_hash).first()
        is_duplicate = existing is not None

        # 3. Extract Features
        y = processed_data["processed_audio"]
        sr = processed_data["sample_rate"]
        features = self.extractor.extract_features(y, sr)
        feature_vector = self.extractor.get_feature_vector(features)

        # 4. Save file permanently
        final_filename = f"{file_hash[:12]}_{filename}"
        final_path = os.path.join(UPLOAD_FOLDER, final_filename)
        if temp_path != final_path and not os.path.exists(final_path):
            os.rename(temp_path, final_path)
        elif temp_path != final_path and os.path.exists(final_path):
            os.remove(temp_path)

        # 5. Create Database Record
        audio_event = AudioEvent(
            file_name=final_filename,
            file_path=final_path,
            duration_seconds=processed_data["duration"],
            sample_rate=sr,
            channels=processed_data["channels"],
            file_size_bytes=processed_data["file_size_bytes"],
            sha256_hash=file_hash,
            is_duplicate=is_duplicate,
            snr_db=processed_data["snr_db"],
            clipping_ratio=processed_data["clipping_ratio"],
            quality_assessment=processed_data["quality_assessment"],
            location=location,
            status="Processed",
            uploaded_by_id=uploaded_by_id
        )

        db.session.add(audio_event)
        db.session.commit()

        processed_data["feature_vector"] = feature_vector
        processed_data["features_dict"] = features
        return audio_event, processed_data
