"""
SonicSentinel AI - Fast Local Model Trainer & Serializer
Extracts acoustic features from the project dataset, trains Random Forest & SVM,
evaluates performance, prevents overfitting with cross-validation & regularization,
and saves the model artifacts into backend/python_models/.
"""
import sys
import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure console supports UTF-8 safely on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config.settings import (
    AUDIO_DATA_DIR, MODELS_DIR, SOUND_CLASSES, SAMPLE_RATE,
    BEST_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, RANDOM_SEED
)
from audio_preprocessing.preprocessor import AudioPreprocessor
from feature_extraction.feature_extractor import FeatureExtractor


def train_and_export_models(samples_per_class: int = 30):
    """
    Trains SVM and Random Forest on balanced samples from each sound category.
    Saves best model, scaler, and label encoder to MODELS_DIR.
    """
    print("=" * 65)
    print("SonicSentinel AI -- Local Model Training & Export Pipeline")
    print("=" * 65)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MODELS_DIR / "features_cache.npz"
    
    if cache_path.exists():
        print("[INFO] Loading cached feature matrix from features_cache.npz...")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        print(f"[OK] Loaded cached features shape: {X.shape}, labels: {len(y)}")
    else:
        preprocessor = AudioPreprocessor(target_sr=SAMPLE_RATE)
        extractor = FeatureExtractor(sr=SAMPLE_RATE)
        
        X_list = []
        y_list = []
        file_records = []
        
        print(f"\n[1/5] Extracting acoustic features (up to {samples_per_class} clips per class)...")
        for cls in SOUND_CLASSES:
            cls_dir = AUDIO_DATA_DIR / cls
            if not cls_dir.exists():
                print(f"  [WARN] Directory not found: {cls_dir}")
                continue
            
            audio_files = sorted(list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")))
            selected_files = audio_files[:samples_per_class]
            print(f"  - Class '{cls}': processing {len(selected_files)} files...")
            
            for fp in selected_files:
                try:
                    y_audio, sr = preprocessor.load_audio(str(fp))
                    y_audio = preprocessor.normalize_amplitude(y_audio)
                    y_audio = preprocessor.pad_or_truncate(y_audio, int(3.0 * SAMPLE_RATE))
                    features = extractor.extract_all(y_audio)
                    
                    if features is not None and len(features) > 0 and not np.isnan(features).any():
                        X_list.append(features)
                        y_list.append(cls)
                        file_records.append({"file": fp.name, "class": cls})
                except Exception as e:
                    print(f"    Failed on {fp.name}: {e}")
                    continue
                    
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        # Cache extracted features for rapid subsequent retraining
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"\n[OK] Extracted and cached feature matrix shape: {X.shape}, Labels: {len(y)}")
    
    # 2. Encoding and Scaling
    print("\n[2/5] Fitting LabelEncoder and StandardScaler...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train / Test Split
    print("\n[3/5] Stratified Train/Test Split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.20, random_state=RANDOM_SEED, stratify=y_encoded
    )
    
    # 4. Model Training & Cross Validation (Overfitting Protection)
    print("\n[4/5] Training Models with 5-Fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    # Model A: Random Forest (Balanced, Regularized)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
    rf.fit(X_train, y_train)
    rf_test_preds = rf.predict(X_test)
    rf_test_acc = accuracy_score(y_test, rf_test_preds)
    rf_test_f1 = f1_score(y_test, rf_test_preds, average="macro")
    
    print(f"\n  -> Random Forest Classifier:")
    print(f"     5-Fold CV Accuracy: {rf_cv_scores.mean() * 100:.2f}% (+/- {rf_cv_scores.std() * 100:.2f}%)")
    print(f"     Test Accuracy:      {rf_test_acc * 100:.2f}%")
    print(f"     Test Macro F1:      {rf_test_f1:.4f}")
    
    # Model B: Support Vector Classifier (RBF kernel, regularized C=10.0)
    svm = SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_SEED
    )
    svm_cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring="accuracy")
    svm.fit(X_train, y_train)
    svm_test_preds = svm.predict(X_test)
    svm_test_acc = accuracy_score(y_test, svm_test_preds)
    svm_test_f1 = f1_score(y_test, svm_test_preds, average="macro")
    
    print(f"\n  -> Support Vector Machine (RBF):")
    print(f"     5-Fold CV Accuracy: {svm_cv_scores.mean() * 100:.2f}% (+/- {svm_cv_scores.std() * 100:.2f}%)")
    print(f"     Test Accuracy:      {svm_test_acc * 100:.2f}%")
    print(f"     Test Macro F1:      {svm_test_f1:.4f}")
    
    # Selection
    if rf_test_f1 >= svm_test_f1:
        best_model = rf
        best_name = "RandomForestClassifier"
        best_acc = rf_test_acc
        best_f1 = rf_test_f1
        best_cv = rf_cv_scores.mean()
        best_preds = rf_test_preds
    else:
        best_model = svm
        best_name = "SVC(kernel='rbf')"
        best_acc = svm_test_acc
        best_f1 = svm_test_f1
        best_cv = svm_cv_scores.mean()
        best_preds = svm_test_preds
        
    print(f"\n[BEST] Selected Model: {best_name} (Test Accuracy: {best_acc*100:.2f}%, F1: {best_f1:.4f})")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, best_preds, target_names=le.classes_))
    
    # 5. Export Artifacts
    print("\n[5/5] Exporting Model Artifacts to python_models/...")
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
        
    metadata = {
        "model_name": best_name,
        "sample_count": len(X),
        "feature_dim": int(X.shape[1]),
        "classes": list(le.classes_),
        "cv_accuracy": float(best_cv),
        "test_accuracy": float(best_acc),
        "test_macro_f1": float(best_f1),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "regularization": {
            "anti_overfitting": "Stratified 5-Fold Cross Validation + Min Leaf Regularization / Soft Margin",
            "feature_standardization": "StandardScaler"
        }
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[OK] Saved best model to: {BEST_MODEL_PATH}")
    print(f"[OK] Saved scaler to:     {SCALER_PATH}")
    print(f"[OK] Saved encoder to:    {LABEL_ENCODER_PATH}")
    print(f"[OK] Saved metadata to:   {MODELS_DIR / 'model_metadata.json'}")
    print("\n" + "=" * 65)
    print("Training and Export Complete Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    train_and_export_models(samples_per_class=30)
