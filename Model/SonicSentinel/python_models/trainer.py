# ============================================================
# SonicSentinel AI - Python ML Trainer (SVM, Random Forest, XGBoost)
# ============================================================
"""
Trains audio classification models using:
1. Support Vector Classifier (SVM)
2. Random Forest Classifier
3. XGBoost Classifier

Includes standard scaling, model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix),
hyperparameter tuning options, cross-validation, and model serialization (.joblib / .json).
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List, Optional

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

from src.config import SOUND_CLASSES, ML_CONFIG, MODEL_DIR


class ModelTrainer:
    """
    Manages the training, evaluation, and saving of Python ML models:
    SVM, Random Forest, and XGBoost for audio classification.
    """

    SUPPORTED_ALGORITHMS = ["svm", "random_forest", "xgboost"]

    def __init__(self, output_dir: str = MODEL_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(SOUND_CLASSES)
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def prepare_data(self, X_train: np.ndarray, y_train: List[str], 
                     X_val: Optional[np.ndarray] = None, y_val: Optional[List[str]] = None) -> Tuple:
        """
        Fits scaler and label encoder on training data, transforms features and targets.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        y_train_encoded = self.label_encoder.transform(y_train)

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            return X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded

        return X_train_scaled, y_train_encoded

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, 
                  tune_hyperparams: bool = False) -> SVC:
        """
        Trains Support Vector Classifier (SVM).
        """
        if tune_hyperparams:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.001],
                'kernel': ['rbf', 'poly']
            }
            base_model = SVC(probability=True, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
            model.fit(X_train, y_train)

        self.trained_models['svm'] = model
        return model

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            tune_hyperparams: bool = False) -> RandomForestClassifier:
        """
        Trains Random Forest Classifier.
        """
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
            base_model = RandomForestClassifier(random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                criterion='gini',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        self.trained_models['random_forest'] = model
        return model

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      tune_hyperparams: bool = False) -> xgb.XGBClassifier:
        """
        Trains XGBoost Classifier.
        """
        num_classes = len(self.label_encoder.classes_)
        
        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            base_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                eval_metric='mlogloss',
                random_state=42
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=num_classes,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

        self.trained_models['xgboost'] = model
        return model

    def evaluate(self, model_key: str, X_test: np.ndarray, y_test_encoded: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates a trained model on test data and produces detailed performance metrics.
        """
        if model_key not in self.trained_models:
            raise ValueError(f"Model {model_key} has not been trained yet.")

        model = self.trained_models[model_key]
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test_encoded, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_encoded, y_pred, average='macro', zero_division=0
        )

        class_report = classification_report(
            y_test_encoded, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(y_test_encoded, y_pred).tolist()

        metrics_summary = {
            "algorithm": model_key,
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": class_report,
            "classes": self.label_encoder.classes_.tolist()
        }

        self.metrics[model_key] = metrics_summary
        return metrics_summary

    def train_all(self, X_train: np.ndarray, y_train: List[str],
                  X_test: np.ndarray, y_test: List[str],
                  tune_hyperparams: bool = False) -> Dict[str, Any]:
        """
        End-to-end training pipeline for all 3 algorithms (SVM, RF, XGBoost).
        Compares results and identifies the best performing algorithm.
        """
        X_tr_scaled, y_tr_enc, X_te_scaled, y_te_enc = self.prepare_data(
            X_train, y_train, X_test, y_test
        )

        results = {}
        
        # 1. Train SVM
        print("Training SVM Classifier...")
        self.train_svm(X_tr_scaled, y_tr_enc, tune_hyperparams=tune_hyperparams)
        results["svm"] = self.evaluate("svm", X_te_scaled, y_te_enc)

        # 2. Train Random Forest
        print("Training Random Forest Classifier...")
        self.train_random_forest(X_tr_scaled, y_tr_enc, tune_hyperparams=tune_hyperparams)
        results["random_forest"] = self.evaluate("random_forest", X_te_scaled, y_te_enc)

        # 3. Train XGBoost
        print("Training XGBoost Classifier...")
        self.train_xgboost(X_tr_scaled, y_tr_enc, tune_hyperparams=tune_hyperparams)
        results["xgboost"] = self.evaluate("xgboost", X_te_scaled, y_te_enc)

        # Select best model based on F1 Macro
        best_algo = max(results.keys(), key=lambda k: results[k]["f1_macro"])
        print(f"Best Performing Model: {best_algo} with F1-Score: {results[best_algo]['f1_macro']}")

        return {
            "best_algorithm": best_algo,
            "results": results
        }

    def save_models(self, version: str = "v1.0.0") -> Dict[str, str]:
        """
        Serializes trained models, scaler, label encoder, and evaluation metadata.
        Saves both versioned and default unversioned binaries for max compatibility.
        """
        import shutil

        saved_paths = {}

        # Save Scaler
        scaler_path = os.path.join(self.output_dir, f"scaler_{version}.joblib")
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.joblib"))
        saved_paths["scaler"] = scaler_path

        # Save Label Encoder
        le_path = os.path.join(self.output_dir, f"label_encoder_{version}.joblib")
        joblib.dump(self.label_encoder, le_path)
        joblib.dump(self.label_encoder, os.path.join(self.output_dir, "label_encoder.joblib"))
        saved_paths["label_encoder"] = le_path

        # Determine best algorithm
        best_algo = max(self.metrics.keys(), key=lambda k: self.metrics[k]["f1_macro"]) if self.metrics else "xgboost"

        # Save Models
        for name, model in self.trained_models.items():
            model_path = os.path.join(self.output_dir, f"{name}_{version}.joblib")
            joblib.dump(model, model_path)
            joblib.dump(model, os.path.join(self.output_dir, f"{name}.joblib"))
            saved_paths[name] = model_path
            if name == best_algo:
                joblib.dump(model, os.path.join(self.output_dir, "best_model.joblib"))

        # Save Metrics Metadata
        metadata = {
            "version": version,
            "best_algorithm": best_algo,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": self.metrics,
            "classes": self.label_encoder.classes_.tolist(),
            "saved_paths": saved_paths
        }
        meta_path = os.path.join(self.output_dir, f"metadata_{version}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        with open(os.path.join(self.output_dir, "model_metrics.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        saved_paths["metadata"] = meta_path
        return saved_paths

