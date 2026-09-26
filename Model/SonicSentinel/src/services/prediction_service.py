# ============================================================
# SonicSentinel AI - Dual-Model Prediction Service
# ============================================================
"""
Prediction Service orchestrating dual-model inference:
  1. Python ML Model (SVM / Random Forest / XGBoost)
  2. Google Teachable Machine (GTM) benchmark comparison
  3. Consensus & Uncertainty evaluation
  4. Database persistence of prediction telemetry
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.extensions import db
from src.models.prediction import Prediction
from python_models.predictor import ModelPredictor
from src.config import GTM_SIMULATION_CONFIG


class PredictionService:
    """
    Executes inference across both Python ML models and GTM benchmark model,
    comparing results and flags mismatches for human review.
    """

    def __init__(self, algorithm: str = "xgboost", model_version: str = "v1.0.0"):
        self.predictor = ModelPredictor(algorithm=algorithm, model_version=model_version)

    def predict_audio_event(
        self,
        audio_event_id: int,
        feature_vector: np.ndarray
    ) -> Tuple[Prediction, Dict[str, Any]]:
        """
        Runs dual-model predictions for an AudioEvent.

        Returns:
            Tuple of (Prediction ORM record, prediction results dictionary)
        """
        start_time = time.time()

        # 1. Run Python ML Inference (SVM / RF / XGBoost)
        py_res = self.predictor.predict(feature_vector)
        py_class = py_res["predicted_class"]
        py_conf = py_res["confidence"]

        # 2. Simulate/Run GTM Model Benchmark Inference
        gtm_res = self._simulate_gtm_prediction(py_class, py_conf)
        gtm_class = gtm_res["predicted_class"]
        gtm_conf = gtm_res["confidence"]

        latency_ms = (time.time() - start_time) * 1000.0

        # 3. Dual Model Comparison & Decision Logic
        models_agree = py_class == gtm_class
        confidence_delta = abs(py_conf - gtm_conf)

        if models_agree and py_conf >= 0.60:
            final_class = py_class
            final_conf = round((py_conf + gtm_conf) / 2.0, 4)
            is_uncertain = False
        elif py_conf >= 0.75:
            final_class = py_class
            final_conf = py_conf
            is_uncertain = not models_agree
        elif gtm_conf >= 0.75:
            final_class = gtm_class
            final_conf = gtm_conf
            is_uncertain = not models_agree
        else:
            final_class = py_class
            final_conf = py_conf
            is_uncertain = True

        # 4. Save Prediction Record
        pred_record = Prediction(
            audio_event_id=audio_event_id,
            python_predicted_class=py_class,
            python_confidence=py_conf,
            python_model_version=self.predictor.model_version,
            python_algorithm=self.predictor.algorithm,
            gtm_predicted_class=gtm_class,
            gtm_confidence=gtm_conf,
            final_predicted_class=final_class,
            final_confidence=final_conf,
            is_uncertain=is_uncertain,
            models_agree=models_agree,
            confidence_delta=round(confidence_delta, 4),
            latency_ms=round(latency_ms, 2)
        )

        db.session.add(pred_record)
        db.session.commit()

        summary = {
            "prediction_id": pred_record.id,
            "final_predicted_class": final_class,
            "final_confidence": final_conf,
            "is_uncertain": is_uncertain,
            "python_model": py_res,
            "gtm_model": gtm_res,
            "models_agree": models_agree,
            "latency_ms": round(latency_ms, 2)
        }

        return pred_record, summary

    def _simulate_gtm_prediction(self, py_class: str, py_conf: float) -> Dict[str, Any]:
        """
        Simulates Google Teachable Machine model outputs based on configured agreement rates.
        In a production browser environment, this connects to TensorFlow.js / Teachable Machine model.
        """
        agreement_rate = GTM_SIMULATION_CONFIG.get("agreement_rate", 0.90)

        # Simulate agreement or slight variance
        if np.random.random() < agreement_rate:
            gtm_class = py_class
            gtm_conf = max(0.50, min(0.99, py_conf + np.random.uniform(-0.05, 0.05)))
        else:
            # Alternate class simulation
            gtm_class = "Background Noise" if py_class != "Background Noise" else "Machinery Fault"
            gtm_conf = float(np.random.uniform(0.50, 0.70))

        return {
            "predicted_class": gtm_class,
            "confidence": round(gtm_conf, 4),
            "model_type": "Google Teachable Machine (Web/TF.js)"
        }
