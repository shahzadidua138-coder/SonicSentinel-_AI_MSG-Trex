"""
SonicSentinel AI - Model Trainer
Trains SVM, Random Forest, and CNN models, compares them, selects best.
"""
import numpy as np
import pandas as pd
import os, pickle, json, time
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Sklearn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    SOUND_CLASSES, RANDOM_SEED, CV_FOLDS,
    MODELS_DIR, SCALER_PATH, LABEL_ENCODER_PATH,
    BEST_MODEL_PATH, FEATURE_NAMES_PATH, MODEL_VERSION
)

np.random.seed(RANDOM_SEED)


class ModelTrainer:
    """
    Trains and compares SVM, Random Forest, and CNN models.
    Selects best based on macro F1-score and accuracy.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    # ─────────────────────────────────────────────
    # Preprocessing
    # ─────────────────────────────────────────────
    def preprocess_labels(self, y: np.ndarray) -> np.ndarray:
        return self.label_encoder.fit_transform(y)

    def fit_scaler(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    # ─────────────────────────────────────────────
    # Model Definitions
    # ─────────────────────────────────────────────
    def get_svm_model(self, tuned: bool = True):
        """Support Vector Machine with RBF kernel."""
        if tuned:
            return SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=5000,
            )
        return SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)

    def get_random_forest_model(self, tuned: bool = True):
        """Random Forest Classifier."""
        if tuned:
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    def get_gradient_boosting_model(self):
        """Gradient Boosting Classifier."""
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_SEED,
        )

    # ─────────────────────────────────────────────
    # Hyperparameter Tuning
    # ─────────────────────────────────────────────
    def tune_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """Grid search for SVM hyperparameters."""
        param_grid = {
            "C": [1, 5, 10, 50],
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto"],
        }
        svm = SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)
        search = RandomizedSearchCV(
            svm, param_grid, n_iter=12,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring="f1_macro", n_jobs=-1, random_state=RANDOM_SEED, verbose=0
        )
        search.fit(X_train, y_train)
        print(f"  SVM Best params: {search.best_params_} | Best CV F1: {search.best_score_:.4f}")
        return search.best_estimator_

    def tune_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Randomized search for RF hyperparameters."""
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 20, 30],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }
        rf = RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        )
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=15,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring="f1_macro", n_jobs=-1, random_state=RANDOM_SEED, verbose=0
        )
        search.fit(X_train, y_train)
        print(f"  RF Best params: {search.best_params_} | Best CV F1: {search.best_score_:.4f}")
        return search.best_estimator_

    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
    ) -> Dict[str, Any]:
        """Evaluate model and return comprehensive metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        classes = self.label_encoder.classes_

        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0
        )

        # Per-class recall (important for critical classes)
        per_class_recall = {}
        for i, cls in enumerate(classes):
            per_class_recall[cls] = recall_score(
                (y_test == i).astype(int),
                (y_pred == i).astype(int),
                zero_division=0
            )

        # Critical class recall
        critical_class_names = {
            "gunshot", "panic_scream", "person_asking_for_help",
            "glass_breaking", "aggression"
        }
        critical_recall_scores = {
            k: v for k, v in per_class_recall.items()
            if k in critical_class_names
        }
        avg_critical_recall = (
            np.mean(list(critical_recall_scores.values()))
            if critical_recall_scores else 0.0
        )

        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "per_class_recall": per_class_recall,
            "critical_class_recall": critical_recall_scores,
            "avg_critical_recall": float(avg_critical_recall),
            "classes": list(classes),
        }

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision (macro):{precision_macro:.4f}")
        print(f"  Recall (macro):   {recall_macro:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")
        print(f"  F1-Score (wt):    {f1_weighted:.4f}")
        print(f"  Avg Critical Rec: {avg_critical_recall:.4f}")
        print(f"\nCritical class recalls:")
        for cls, rec in critical_recall_scores.items():
            status = "✓" if rec >= 0.85 else "✗"
            print(f"  {status} {cls}: {rec:.4f}")

        return metrics

    def cross_validate_model(
        self, model, X: np.ndarray, y: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """Run stratified k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        cv_f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)

        print(f"\n  {model_name} CV ({CV_FOLDS}-fold):")
        print(f"    Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"    F1 Macro: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        return {
            "cv_accuracy_mean": float(cv_acc.mean()),
            "cv_accuracy_std": float(cv_acc.std()),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
        }

    # ─────────────────────────────────────────────
    # Train All Models
    # ─────────────────────────────────────────────
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_tuning: bool = True,
    ) -> Dict[str, Any]:
        """
        Train SVM, Random Forest, and Gradient Boosting.
        Select best model based on macro F1 + critical recall.
        """
        print("\n" + "="*60)
        print("SonicSentinel AI - Model Training")
        print("="*60)

        # Scale features
        X_train_scaled = self.fit_scaler(X_train)
        X_val_scaled = self.scale(X_val)
        X_test_scaled = self.scale(X_test)

        trained_models = {}
        all_metrics = {}

        # ── SVM ──
        print("\n[1/3] Training Support Vector Machine...")
        t0 = time.time()
        if use_tuning:
            svm = self.tune_svm(X_train_scaled, y_train)
        else:
            svm = self.get_svm_model(tuned=True)
            svm.fit(X_train_scaled, y_train)

        svm_time = time.time() - t0
        print(f"  Training time: {svm_time:.1f}s")
        svm_cv = self.cross_validate_model(svm, X_train_scaled, y_train, "SVM")
        svm_metrics = self.evaluate_model(svm, X_test_scaled, y_test, "SVM")
        svm_metrics.update(svm_cv)
        svm_metrics["training_time_seconds"] = svm_time
        trained_models["SVM"] = svm
        all_metrics["SVM"] = svm_metrics

        # Save SVM
        svm_path = self.models_dir / "svm_model.pkl"
        with open(svm_path, "wb") as f:
            pickle.dump(svm, f)
        print(f"  Saved: {svm_path}")

        # ── Random Forest ──
        print("\n[2/3] Training Random Forest...")
        t0 = time.time()
        if use_tuning:
            rf = self.tune_random_forest(X_train_scaled, y_train)
        else:
            rf = self.get_random_forest_model(tuned=True)
            rf.fit(X_train_scaled, y_train)

        rf_time = time.time() - t0
        print(f"  Training time: {rf_time:.1f}s")
        rf_cv = self.cross_validate_model(rf, X_train_scaled, y_train, "RandomForest")
        rf_metrics = self.evaluate_model(rf, X_test_scaled, y_test, "RandomForest")
        rf_metrics.update(rf_cv)
        rf_metrics["training_time_seconds"] = rf_time
        trained_models["RandomForest"] = rf
        all_metrics["RandomForest"] = rf_metrics

        # Save RF
        rf_path = self.models_dir / "rf_model.pkl"
        with open(rf_path, "wb") as f:
            pickle.dump(rf, f)
        print(f"  Saved: {rf_path}")

        # ── Gradient Boosting ──
        print("\n[3/3] Training Gradient Boosting...")
        t0 = time.time()
        gb = self.get_gradient_boosting_model()
        gb.fit(X_train_scaled, y_train)
        gb_time = time.time() - t0
        print(f"  Training time: {gb_time:.1f}s")
        gb_cv = self.cross_validate_model(gb, X_train_scaled, y_train, "GradientBoosting")
        gb_metrics = self.evaluate_model(gb, X_test_scaled, y_test, "GradientBoosting")
        gb_metrics.update(gb_cv)
        gb_metrics["training_time_seconds"] = gb_time
        trained_models["GradientBoosting"] = gb
        all_metrics["GradientBoosting"] = gb_metrics

        gb_path = self.models_dir / "gb_model.pkl"
        with open(gb_path, "wb") as f:
            pickle.dump(gb, f)
        print(f"  Saved: {gb_path}")

        # ── Select Best Model ──
        print("\n" + "="*60)
        print("Model Comparison Summary")
        print("="*60)
        print(f"{'Model':<25} {'Accuracy':>10} {'F1-Macro':>10} {'Crit Rec':>10}")
        print("-"*60)

        best_score = -1
        for name, metrics in all_metrics.items():
            score = (
                metrics["accuracy"] * 0.35
                + metrics["f1_macro"] * 0.45
                + metrics["avg_critical_recall"] * 0.20
            )
            print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['f1_macro']:>10.4f} {metrics['avg_critical_recall']:>10.4f}")
            if score > best_score:
                best_score = score
                self.best_model_name = name
                self.best_model = trained_models[name]

        print(f"\n🏆 Best Model: {self.best_model_name}")

        # Save best model
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(self.best_model, f)

        # Save scaler and label encoder
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(self.label_encoder, f)

        # Save results
        results_path = self.models_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} 
                 for k, v in all_metrics.items()},
                f, indent=2
            )

        self.results = all_metrics
        return {
            "best_model_name": self.best_model_name,
            "all_metrics": all_metrics,
            "trained_models": trained_models,
        }

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────
    def predict(
        self,
        model,
        X: np.ndarray,
        apply_scaler: bool = True,
    ) -> Dict[str, Any]:
        """
        Run prediction on feature vector(s).
        Returns predicted class, confidence scores for all classes.
        """
        if apply_scaler:
            X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        else:
            X_scaled = X.reshape(1, -1) if X.ndim == 1 else X

        pred_encoded = model.predict(X_scaled)
        pred_class = self.label_encoder.inverse_transform(pred_encoded)[0]

        proba = model.predict_proba(X_scaled)[0]
        confidence_scores = {
            cls: float(proba[i])
            for i, cls in enumerate(self.label_encoder.classes_)
        }

        top_confidence = float(proba[pred_encoded[0]])
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        top_two_margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 1.0

        return {
            "predicted_class": pred_class,
            "confidence": top_confidence,
            "confidence_scores": confidence_scores,
            "top_two_margin": float(top_two_margin),
            "top_n_predictions": sorted_scores[:3],
            "model_version": MODEL_VERSION,
        }

    # ─────────────────────────────────────────────
    # Load Saved Model
    # ─────────────────────────────────────────────
    @staticmethod
    def load_models(models_dir: Path = MODELS_DIR) -> Dict[str, Any]:
        """Load saved best model, scaler, and label encoder."""
        with open(BEST_MODEL_PATH, "rb") as f:
            best_model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        return {
            "best_model": best_model,
            "scaler": scaler,
            "label_encoder": label_encoder,
        }
