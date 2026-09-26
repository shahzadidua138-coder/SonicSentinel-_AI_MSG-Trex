# ============================================================
# SonicSentinel AI - ML Model Unit Tests (SVM, RF, XGBoost)
# ============================================================
import os
import shutil
import unittest
import numpy as np
from python_models.trainer import ModelTrainer
from python_models.predictor import ModelPredictor
from src.config import SOUND_CLASSES


class TestMLModels(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = "tests/temp_models"
        os.makedirs(self.test_model_dir, exist_ok=True)
        self.trainer = ModelTrainer(output_dir=self.test_model_dir)

        # Generate synthetic 373-dim data for 4 classes
        n_samples = 40
        self.X_train = np.random.randn(n_samples, 373)
        self.y_train = [SOUND_CLASSES[i % len(SOUND_CLASSES)] for i in range(n_samples)]

        self.X_test = np.random.randn(10, 373)
        self.y_test = [SOUND_CLASSES[i % len(SOUND_CLASSES)] for i in range(10)]

    def tearDown(self):
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)

    def test_train_all_models(self):
        res = self.trainer.train_all(
            self.X_train, self.y_train, self.X_test, self.y_test, tune_hyperparams=False
        )

        self.assertIn("best_algorithm", res)
        self.assertIn("svm", res["results"])
        self.assertIn("random_forest", res["results"])
        self.assertIn("xgboost", res["results"])

        saved = self.trainer.save_models(version="v_test")
        self.assertIn("svm", saved)
        self.assertIn("random_forest", saved)
        self.assertIn("xgboost", saved)
        self.assertIn("scaler", saved)

    def test_inference(self):
        self.trainer.train_all(
            self.X_train, self.y_train, self.X_test, self.y_test, tune_hyperparams=False
        )
        self.trainer.save_models(version="v_test")

        predictor = ModelPredictor(model_version="v_test", algorithm="xgboost")
        predictor.model_dir = self.test_model_dir
        predictor._load_artifacts()

        single_vector = np.random.randn(373)
        pred = predictor.predict(single_vector)

        self.assertIn("predicted_class", pred)
        self.assertIn("confidence", pred)
        self.assertIn("probability_distribution", pred)
        self.assertIn("is_uncertain", pred)


if __name__ == "__main__":
    unittest.main()
