# ============================================================
# SonicSentinel AI - Standalone Model Training Pipeline
# ============================================================
"""
Full Dataset Training Pipeline:
  1. Reads metadata.csv from data/audio_dataset
  2. Extracts 373-dim acoustic feature vectors for all audio files
  3. Prepares Train/Val/Test splits
  4. Trains SVM, Random Forest, and XGBoost models
  5. Computes full evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
  6. Serializes model binaries (.joblib) and metadata (.json) to models/ folder
"""

import os
import sys
import csv
import numpy as np
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from audio_preprocessing.preprocessor import AudioPreprocessor
from feature_extraction.extractor import FeatureExtractor
from python_models.trainer import ModelTrainer
from src.config import METADATA_CSV_PATH, MODEL_DIR, SOUND_CLASSES
from src.app import create_app
from src.extensions import db
from src.models.model_version import ModelVersion


def run_training_pipeline(dataset_dir: str = "d:/T/data/audio_dataset", version: str = "v1.0.0"):
    print("=" * 60)
    print(" SonicSentinel AI - ML Model Training Pipeline ")
    print("=" * 60)

    csv_path = os.path.join(dataset_dir, "metadata.csv")
    audio_dir = os.path.join(dataset_dir, "audio")

    if not os.path.exists(csv_path):
        print(f"Error: Metadata file not found at {csv_path}")
        return

    print(f"Reading dataset metadata from: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total dataset samples found: {len(rows)}")

    preprocessor = AudioPreprocessor()
    extractor = FeatureExtractor()

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    print("\nExtracting features from dataset samples...")
    processed_count = 0
    skipped_count = 0

    for idx, row in enumerate(rows):
        file_name = row.get("file_name", row.get("filename", ""))
        rel_path = row.get("file_path", "")
        sound_class = row["class"]
        split = row["data_split"]

        full_audio_path = os.path.join(dataset_dir, split, file_name)
        if not os.path.exists(full_audio_path):
            full_audio_path = os.path.abspath(os.path.join(dataset_dir, "..", "..", rel_path))

        if not os.path.exists(full_audio_path):
            skipped_count += 1
            continue

        try:
            # Preprocess audio
            prep_res = preprocessor.preprocess(full_audio_path)
            y_audio = prep_res.get("preprocessed_audio", prep_res.get("processed_audio"))
            sr = prep_res.get("sr", prep_res.get("sample_rate", 22050))

            if y_audio is None or len(y_audio) == 0:
                import librosa
                y_audio, sr = librosa.load(full_audio_path, sr=22050)

            # Extract 373-dim feature vector
            feats_dict = extractor.extract_features(y_audio, sr)
            vector = extractor.get_feature_vector(feats_dict)

            if split == "train":
                X_train.append(vector)
                y_train.append(sound_class)
            elif split == "val":
                X_val.append(vector)
                y_val.append(sound_class)
            else:
                X_test.append(vector)
                y_test.append(sound_class)

            processed_count += 1
            if processed_count % 300 == 0:
                print(f"  Processed {processed_count}/{len(rows)} audio files...")

        except Exception as e:
            skipped_count += 1
            continue

    print(f"\nFeature extraction complete! Processed: {processed_count}, Skipped: {skipped_count}")

    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Insufficient samples extracted to proceed with training.")
        return

    X_train_arr = np.array(X_train)
    X_test_arr = np.array(X_test)

    # Initialize Trainer
    trainer = ModelTrainer(output_dir=MODEL_DIR)

    # Train SVM, Random Forest, XGBoost
    print("\nStarting model training for SVM, Random Forest, and XGBoost...")
    training_summary = trainer.train_all(
        X_train=X_train_arr,
        y_train=y_train,
        X_test=X_test_arr,
        y_test=y_test,
        tune_hyperparams=False
    )

    # Save artifacts
    saved_paths = trainer.save_models(version=version)
    print(f"\nModel artifacts saved successfully to: {MODEL_DIR}")

    best_algo = training_summary["best_algorithm"]
    best_metrics = training_summary["results"][best_algo]

    # Update database ModelVersion record
    app = create_app()
    with app.app_context():
        # Deactivate old versions
        ModelVersion.query.filter_by(framework="Python-Scikit/XGBoost").update({"is_active": False})

        new_version = ModelVersion(
            version_name=version,
            algorithm=best_algo.upper(),
            framework="Python-Scikit/XGBoost",
            features_dimensionality=373,
            is_active=True,
            accuracy=best_metrics["accuracy"],
            precision=best_metrics["precision_macro"],
            recall=best_metrics["recall_macro"],
            f1_score=best_metrics["f1_macro"],
            description=f"Best performing Python ML model: {best_algo.upper()} trained on audio features."
        )
        db.session.add(new_version)
        db.session.commit()

    print("\n" + "=" * 60)
    print(f" Training Complete! Best Algorithm: {best_algo.upper()} ")
    print(f" F1-Score: {best_metrics['f1_macro']} | Accuracy: {best_metrics['accuracy']}")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
