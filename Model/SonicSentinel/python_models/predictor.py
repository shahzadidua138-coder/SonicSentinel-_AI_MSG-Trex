# ============================================================
# SonicSentinel AI - Python ML Inference Predictor
# ============================================================
"""
Inference Engine for Python ML Models (SVM, Random Forest, XGBoost):
  - Loads trained model artifacts (.joblib)
  - Preprocesses feature vectors using saved scaler
  - Computes class probabilities and top prediction
  - Evaluates confidence against minimum threshold (0.60)
  - Identifies top-K alternative predictions
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.config import MODEL_DIR, ML_CONFIG, SOUND_CLASSES, SEVERITY_MAPPING


class ModelPredictor:
    """
    Inference handler for loading trained models and scoring extracted feature vectors.
    """

    def __init__(self, model_version: str = "v1.0.0", algorithm: str = "xgboost"):
        self.model_version = model_version
        self.algorithm = algorithm
        self.model_dir = MODEL_DIR
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.classes = SOUND_CLASSES
        self.is_loaded = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Loads scaler, label encoder, and requested algorithm model.
        Falls back gracefully to default unversioned files if exact version files are missing.
        """
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.model_version}.joblib")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")

        le_path = os.path.join(self.model_dir, f"label_encoder_{self.model_version}.joblib")
        if not os.path.exists(le_path):
            le_path = os.path.join(self.model_dir, "label_encoder.joblib")

        model_path = os.path.join(self.model_dir, f"{self.algorithm.lower()}_{self.model_version}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, f"{self.algorithm.lower()}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.model_dir, "best_model.joblib")

        if os.path.exists(scaler_path) and os.path.exists(le_path) and os.path.exists(model_path):
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            self.model = joblib.load(model_path)
            self.classes = list(self.label_encoder.classes_)
            self.is_loaded = True
        else:
            self.is_loaded = False


    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Runs prediction on a 1D or 2D feature vector.

        Returns:
            Dict containing:
              - predicted_class: str
              - confidence: float (0.0 to 1.0)
              - probability_distribution: Dict[str, float]
              - top_k: List[Dict[str, Any]]
              - is_uncertain: bool
              - severity: str
              - algorithm_used: str
              - model_version: str
        """
        if not self.is_loaded:
            # Fallback heuristic prediction if model files not trained yet
            return self._heuristic_fallback()

        # Ensure vector shape (1, n_features)
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Scale features
        scaled_features = self.scaler.transform(feature_vector)

        # Get probabilities
        probabilities = self.model.predict_proba(scaled_features)[0]

        # Get top prediction
        top_idx = int(np.argmax(probabilities))
        predicted_class = str(self.classes[top_idx])
        confidence = float(probabilities[top_idx])

        # Map probability distribution
        prob_dist = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(self.classes, probabilities)
        }

        # Sort top-K predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        top_k = [
            {
                "class": str(self.classes[idx]),
                "probability": round(float(probabilities[idx]), 4)
            }
            for idx in sorted_indices[:3]
        ]

        # Uncertainty check
        min_conf = ML_CONFIG.get("min_confidence_threshold", 0.60)
        is_uncertain = confidence < min_conf

        # Determine severity
        severity = SEVERITY_MAPPING.get(predicted_class, "Medium")

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probability_distribution": prob_dist,
            "top_k": top_k,
            "is_uncertain": is_uncertain,
            "severity": severity,
            "algorithm_used": self.algorithm,
            "model_version": self.model_version,
            "status": "success"
        }

    def _heuristic_fallback(self) -> Dict[str, Any]:
        """
        Provides fallback prediction response when ML model binary files are not yet generated.
        """
        return {
            "predicted_class": "Background Noise",
            "confidence": 0.50,
            "probability_distribution": {cls: 0.10 for cls in self.classes},
            "top_k": [{"class": "Background Noise", "probability": 0.50}],
            "is_uncertain": True,
            "severity": "Low",
            "algorithm_used": f"{self.algorithm} (Untrained Fallback)",
            "model_version": self.model_version,
            "status": "fallback_model_not_found"
        }
