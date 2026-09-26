"""
SonicSentinel AI - Core Prediction & Classification Service
Orchestrates audio preprocessing, feature extraction, model inference,
model comparison, quality checks, and alert generation.
"""
import numpy as np
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    SOUND_CLASSES, CLASS_LABELS, SEVERITY_MAP, CRITICAL_CLASSES,
    MIN_CONFIDENCE_THRESHOLD, TOP_TWO_MARGIN_THRESHOLD,
    MODEL_AGREEMENT_THRESHOLD, CRITICAL_CONFIDENCE_THRESHOLD,
    CONSECUTIVE_DETECTIONS_REQUIRED, MODEL_VERSION,
    BEST_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, CNN_MODEL_PATH
)
from audio_preprocessing.preprocessor import AudioPreprocessor
from feature_extraction.feature_extractor import FeatureExtractor


class PredictionService:
    """
    Central prediction service for SonicSentinel AI.
    Coordinates:
      - Audio preprocessing
      - Feature extraction
      - Python model (SVM/RF) inference
      - CNN model inference
      - Model comparison and consistency check
      - Uncertainty detection
      - Alert rule evaluation
    """

    def __init__(self):
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self._best_model = None
        self._scaler = None
        self._label_encoder = None
        self._cnn_model = None
        self._models_loaded = False
        self._consecutive_detections: Dict[str, int] = {}
        self._alert_rules = self._load_alert_rules()

    # ─────────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────────
    def load_models(self):
        """Load saved models from disk."""
        try:
            with open(BEST_MODEL_PATH, "rb") as f:
                self._best_model = pickle.load(f)
            with open(SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            with open(LABEL_ENCODER_PATH, "rb") as f:
                self._label_encoder = pickle.load(f)
            print("✓ Python model loaded")
        except FileNotFoundError as e:
            print(f"⚠ Python model not found: {e}. Run training first.")

        # Try loading CNN
        try:
            import tensorflow as tf
            cnn_path = str(CNN_MODEL_PATH)
            if Path(cnn_path).exists():
                self._cnn_model = tf.keras.models.load_model(cnn_path)
                print("✓ CNN model loaded")
        except Exception as e:
            print(f"⚠ CNN model not loaded: {e}")

        self._models_loaded = True

    def _load_alert_rules(self) -> Dict[str, Any]:
        """Load alert rules from JSON file."""
        rules_path = Path(__file__).resolve().parent.parent.parent / "alert_rules" / "alert_rules.json"
        try:
            with open(rules_path) as f:
                data = json.load(f)
            rules_by_class = {}
            for rule in data["alert_rules"]:
                rules_by_class[rule["sound_category"]] = rule
            return {
                "rules": rules_by_class,
                "global": data["global_settings"],
            }
        except Exception as e:
            print(f"Warning: Could not load alert rules: {e}")
            return {"rules": {}, "global": {}}

    # ─────────────────────────────────────────────
    # Python Model Prediction
    # ─────────────────────────────────────────────
    def _predict_python(self, features: np.ndarray) -> Dict[str, Any]:
        """Run inference with the best sklearn model."""
        if self._best_model is None:
            raise RuntimeError("Python model not loaded.")

        X = features.reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        pred_encoded = self._best_model.predict(X_scaled)
        pred_class = self._label_encoder.inverse_transform(pred_encoded)[0]
        proba = self._best_model.predict_proba(X_scaled)[0]

        confidence_scores = {
            cls: float(proba[i])
            for i, cls in enumerate(self._label_encoder.classes_)
        }
        top_confidence = float(proba[pred_encoded[0]])
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        top_two_margin = (
            sorted_scores[0][1] - sorted_scores[1][1]
            if len(sorted_scores) > 1 else 1.0
        )

        return {
            "predicted_class": pred_class,
            "confidence": top_confidence,
            "confidence_scores": confidence_scores,
            "top_two_margin": float(top_two_margin),
            "top_n_predictions": sorted_scores[:3],
            "model_version": MODEL_VERSION,
        }

    # ─────────────────────────────────────────────
    # CNN Prediction
    # ─────────────────────────────────────────────
    def _predict_cnn(self, audio_segment: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run inference with CNN on Mel spectrogram."""
        if self._cnn_model is None:
            return None
        try:
            mel_spec = self.feature_extractor.extract_mel_spectrogram_2d(audio_segment)

            # Normalize
            spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9)
            spec = spec[np.newaxis, ..., np.newaxis]  # (1, n_mels, time, 1)

            proba = self._cnn_model.predict(spec, verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            pred_class = SOUND_CLASSES[pred_idx]
            top_confidence = float(proba[pred_idx])

            confidence_scores = {cls: float(proba[i]) for i, cls in enumerate(SOUND_CLASSES)}
            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            top_two_margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 1.0

            return {
                "predicted_class": pred_class,
                "confidence": top_confidence,
                "confidence_scores": confidence_scores,
                "top_two_margin": float(top_two_margin),
                "top_n_predictions": sorted_scores[:3],
                "model_version": f"CNN-{MODEL_VERSION}",
            }
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return None

    # ─────────────────────────────────────────────
    # Model Comparison
    # ─────────────────────────────────────────────
    def _compare_models(
        self,
        python_result: Dict[str, Any],
        gtm_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare Python and GTM model predictions.
        Returns consistency status and confidence difference.
        """
        if gtm_result is None:
            return {
                "models_agree": None,
                "consistency_status": "gtm_unavailable",
                "confidence_difference": None,
                "class_match": None,
                "top_two_margin": python_result.get("top_two_margin", 0),
            }

        py_class = python_result["predicted_class"]
        gtm_class = gtm_result["predicted_class"]
        py_conf = python_result["confidence"]
        gtm_conf = gtm_result["confidence"]

        class_match = py_class == gtm_class
        conf_diff = abs(py_conf - gtm_conf)

        # Consistency status
        if class_match and conf_diff <= MODEL_AGREEMENT_THRESHOLD:
            consistency = "acceptable_match"
        elif class_match and conf_diff <= 0.35:
            consistency = "weak_match"
        elif not class_match and conf_diff <= 0.40:
            consistency = "model_disagreement"
        else:
            consistency = "uncertain_result"

        models_agree = class_match and conf_diff <= MODEL_AGREEMENT_THRESHOLD

        return {
            "models_agree": models_agree,
            "class_match": class_match,
            "consistency_status": consistency,
            "confidence_difference": float(conf_diff),
            "python_top_class": py_class,
            "gtm_top_class": gtm_class,
            "python_confidence": py_conf,
            "gtm_confidence": gtm_conf,
        }

    # ─────────────────────────────────────────────
    # Uncertainty & Overlapping Sound Detection
    # ─────────────────────────────────────────────
    def _detect_uncertainty(
        self,
        python_result: Dict[str, Any],
        quality_info: Dict[str, Any],
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Determine if prediction is uncertain and detect overlapping sounds."""
        reasons = []
        is_uncertain = False

        conf = python_result["confidence"]
        top_two_margin = python_result.get("top_two_margin", 1.0)
        quality = quality_info.get("quality", "good")

        # Low confidence
        if conf < MIN_CONFIDENCE_THRESHOLD:
            is_uncertain = True
            reasons.append(f"Low confidence ({conf:.2f} < {MIN_CONFIDENCE_THRESHOLD})")

        # Small top-2 margin
        if top_two_margin < TOP_TWO_MARGIN_THRESHOLD:
            is_uncertain = True
            reasons.append(f"Small top-2 margin ({top_two_margin:.2f})")

        # Poor quality
        if quality in ["poor", "unusable"]:
            is_uncertain = True
            reasons.append(f"Audio quality: {quality}")

        # Model disagreement
        if comparison.get("class_match") is False:
            is_uncertain = True
            reasons.append("Models disagree on category")

        # Overlapping sounds: multiple classes with significant confidence
        confidence_scores = python_result.get("confidence_scores", {})
        overlap_threshold = self._alert_rules["global"].get("overlap_detection_threshold", 0.30)
        overlapping = [
            cls for cls, score in confidence_scores.items()
            if score >= overlap_threshold and cls != python_result["predicted_class"]
        ]
        has_overlapping = len(overlapping) > 0

        return {
            "is_uncertain": is_uncertain,
            "has_overlapping_sounds": has_overlapping,
            "overlapping_classes": overlapping,
            "uncertainty_reasons": reasons,
        }

    # ─────────────────────────────────────────────
    # Alert Rules
    # ─────────────────────────────────────────────
    def _evaluate_alert_rules(
        self,
        predicted_class: str,
        python_result: Dict[str, Any],
        comparison: Dict[str, Any],
        quality_info: Dict[str, Any],
        uncertainty: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate alert rules and determine if alert should be generated."""
        rule = self._alert_rules["rules"].get(predicted_class, {})
        severity = SEVERITY_MAP.get(predicted_class, "informational")

        conf = python_result["confidence"]
        quality = quality_info.get("quality", "good")
        min_conf = rule.get("min_confidence", MIN_CONFIDENCE_THRESHOLD)

        # Consecutive detection tracking
        if session_id:
            key = f"{session_id}_{predicted_class}"
            if predicted_class in CRITICAL_CLASSES or predicted_class in {"glass_breaking", "alarm_siren", "aggression", "machinery_fault"}:
                self._consecutive_detections[key] = self._consecutive_detections.get(key, 0) + 1
            else:
                self._consecutive_detections[key] = 1
            consecutive = self._consecutive_detections.get(key, 1)
        else:
            consecutive = 1

        # Determine if alert should fire
        should_alert = (
            conf >= min_conf
            and not uncertainty["is_uncertain"]
            and quality not in ["unusable"]
        )

        # For critical classes, stricter check
        if predicted_class in CRITICAL_CLASSES:
            should_alert = should_alert and conf >= CRITICAL_CONFIDENCE_THRESHOLD

        # Manual review required?
        requires_review = (
            uncertainty["is_uncertain"]
            or quality == "poor"
            or comparison.get("class_match") is False
            or conf < 0.60
        )

        return {
            "severity": severity,
            "should_alert": should_alert,
            "requires_review": requires_review,
            "consecutive_detections": consecutive,
            "recommended_action": rule.get("recommended_action", "Monitor the situation."),
            "confidence_threshold_met": conf >= min_conf,
            "quality_ok": quality not in ["unusable"],
        }

    # ─────────────────────────────────────────────
    # Final Decision
    # ─────────────────────────────────────────────
    def _make_final_decision(
        self,
        python_result: Dict[str, Any],
        gtm_result: Optional[Dict[str, Any]],
        comparison: Dict[str, Any],
        uncertainty: Dict[str, Any],
        alert_eval: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate the final sound event decision."""
        # Prefer Python model prediction as final
        # If GTM agrees and has higher confidence, we note it but keep Python as primary
        final_class = python_result["predicted_class"]

        if uncertainty["is_uncertain"]:
            status = "uncertain"
        elif alert_eval["should_alert"]:
            status = "alert_generated"
        else:
            status = "classified"

        return {
            "detected_sound_category": final_class,
            "detected_sound_label": CLASS_LABELS.get(final_class, final_class),
            "model_agreement_status": comparison.get("consistency_status"),
            "confidence_level": python_result["confidence"],
            "event_severity": alert_eval["severity"],
            "alert_status": "active" if alert_eval["should_alert"] else "none",
            "recommended_action": alert_eval["recommended_action"],
            "manual_review_required": alert_eval["requires_review"],
            "event_status": status,
        }

    # ─────────────────────────────────────────────
    # Main Classification Entry Point
    # ─────────────────────────────────────────────
    def classify_audio_file(
        self,
        file_path: str,
        gtm_result: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: validate → preprocess → extract features →
        predict → compare → assess quality → generate alert decision.
        """
        t0 = time.time()
        audio_id = str(uuid.uuid4())

        # Step 1: Validate
        validation = self.preprocessor.validate_file(file_path)
        if not validation["valid"]:
            return {
                "audio_id": audio_id,
                "success": False,
                "errors": validation["errors"],
                "status": "rejected",
            }

        # Step 2: Preprocess
        preprocessed = self.preprocessor.preprocess(file_path)
        quality_info = preprocessed["quality"]
        segments = preprocessed["segments"]

        if not segments:
            return {
                "audio_id": audio_id,
                "success": False,
                "errors": ["No valid audio segments extracted."],
                "status": "rejected",
            }

        # Process each segment and aggregate
        all_python_results = []
        all_cnn_results = []

        for seg_data in segments:
            if seg_data["quality"] == "unusable":
                continue
            seg_audio = seg_data["audio"]

            # Feature extraction
            try:
                features = self.feature_extractor.extract_all(seg_audio)
                py_result = self._predict_python(features)
                all_python_results.append(py_result)
            except Exception as e:
                print(f"Python prediction error on segment: {e}")

            # CNN prediction
            cnn_result = self._predict_cnn(seg_audio)
            if cnn_result:
                all_cnn_results.append(cnn_result)

        if not all_python_results:
            return {
                "audio_id": audio_id,
                "success": False,
                "errors": ["All segments unusable or prediction failed."],
                "status": "rejected",
            }

        # Aggregate: pick segment with highest confidence
        best_py = max(all_python_results, key=lambda x: x["confidence"])

        # CNN ensemble
        if all_cnn_results:
            best_cnn = max(all_cnn_results, key=lambda x: x["confidence"])
        else:
            best_cnn = None

        # Use GTM result if provided externally, otherwise use CNN as GTM proxy
        effective_gtm = gtm_result or best_cnn

        # Compare models
        comparison = self._compare_models(best_py, effective_gtm)

        # Uncertainty detection
        uncertainty = self._detect_uncertainty(best_py, quality_info, comparison)

        # Alert rules
        alert_eval = self._evaluate_alert_rules(
            best_py["predicted_class"],
            best_py,
            comparison,
            quality_info,
            uncertainty,
            session_id,
        )

        # Final decision
        final = self._make_final_decision(best_py, effective_gtm, comparison, uncertainty, alert_eval)

        processing_time = (time.time() - t0) * 1000  # ms

        return {
            "audio_id": audio_id,
            "success": True,
            "processing_time_ms": round(processing_time, 1),
            "metadata": validation["metadata"],
            "audio_quality": quality_info,
            "python_prediction": best_py,
            "cnn_prediction": best_cnn,
            "gtm_prediction": gtm_result,
            "model_comparison": comparison,
            "uncertainty_analysis": uncertainty,
            "alert_evaluation": alert_eval,
            "final_decision": final,
            "segments_processed": len(all_python_results),
            "all_segment_predictions": [
                {"predicted_class": r["predicted_class"], "confidence": r["confidence"]}
                for r in all_python_results
            ],
        }

    def classify_live_window(
        self,
        audio_array: np.ndarray,
        session_id: str,
        gtm_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a single live microphone window.
        Lightweight processing for real-time use.
        """
        t0 = time.time()

        # Preprocess
        y, quality_info = self.preprocessor.preprocess_live_window(audio_array)

        if quality_info["quality"] == "unusable":
            return {
                "success": False,
                "quality": quality_info,
                "status": "unusable_audio",
            }

        # Feature extraction
        try:
            features = self.feature_extractor.extract_all(y)
            py_result = self._predict_python(features)
        except Exception as e:
            return {"success": False, "error": str(e)}

        # CNN prediction
        cnn_result = self._predict_cnn(y)
        effective_gtm = gtm_result or cnn_result

        # Compare and evaluate
        comparison = self._compare_models(py_result, effective_gtm)
        uncertainty = self._detect_uncertainty(py_result, quality_info, comparison)
        alert_eval = self._evaluate_alert_rules(
            py_result["predicted_class"],
            py_result,
            comparison,
            quality_info,
            uncertainty,
            session_id,
        )
        final = self._make_final_decision(py_result, effective_gtm, comparison, uncertainty, alert_eval)

        processing_time = (time.time() - t0) * 1000

        return {
            "success": True,
            "processing_time_ms": round(processing_time, 1),
            "audio_quality": quality_info,
            "python_prediction": py_result,
            "cnn_prediction": cnn_result,
            "model_comparison": comparison,
            "uncertainty_analysis": uncertainty,
            "alert_evaluation": alert_eval,
            "final_decision": final,
        }
