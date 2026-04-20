"""Tests for src/training/model_evaluation.py"""

import os
import json
from unittest.mock import patch

import pytest
import numpy as np

from sklearn.preprocessing import LabelEncoder

import pytest
from unittest.mock import patch

from src.training.model_evaluation import (
    evaluate_models,
    feature_importance_analysis,
    format_ap_label,
    learning_curve_analysis,
    load_bssid_to_ssid_map,
    load_trained_models,
    misclassification_analysis,
    parse_args
)
from src.training.preprocess import load_and_preprocess


@pytest.fixture
def eval_results(trained_models_dir, test_db):
    """
    Run evaluate_models once so multiple test classes can reuse the output.
    
    Args:
        trained_models_dir: Fixture providing path to directory with trained .pkl models.
        test_db: Fixture providing path to test database.

    Returns:
        Tuple of (results, label_encoder, features, X_train, y_train) where:
        - results is the dict output from evaluate_models
        - label_encoder is the LabelEncoder instance used to decode class labels
        - features is the list of feature names (BSSIDs)
        - X_train and y_train are the training data and labels used to train the models (useful for feature importance analysis)
    """
    X_train, X_test, y_train, y_test, features, le, scaler = load_and_preprocess(test_db)
    models = load_trained_models(trained_models_dir)
    results = evaluate_models(models, X_test, y_test, le)
    return results, le, features, X_train, y_train


class TestLoadBssidToSsidMap:
    """
    Tests for the load_bssid_to_ssid_map function, which loads a mapping of BSSID to SSID from the database.
    This mapping is used to create more informative labels for feature importance analysis.
    """
    
    def test_returns_dict_with_all_bssids(self, test_db):
        mapping = load_bssid_to_ssid_map(test_db)
        assert isinstance(mapping, dict)
        assert len(mapping) == 10

    def test_values_are_strings(self, test_db):
        mapping = load_bssid_to_ssid_map(test_db)
        assert all(isinstance(ssid, str) for ssid in mapping.values())


class TestFormatApLabel:
    """Tests for the format_ap_label function, which creates a human-readable label for an AP based on its BSSID and the BSSID->SSID mapping."""
    def test_known_bssid_uses_ssid_and_tail(self):
        label = format_ap_label("aa:bb:cc:dd:ee:ff", {"aa:bb:cc:dd:ee:ff": "eduroam"})
        assert "eduroam" in label
        assert "ee:ff" in label

    def test_unknown_bssid_falls_back_to_unknown(self):
        label = format_ap_label("aa:bb:cc:dd:ee:ff", {})
        assert "Unknown" in label

    def test_empty_ssid_does_not_crash(self):
        label = format_ap_label("aa:bb:cc:dd:ee:ff", {"aa:bb:cc:dd:ee:ff": ""})
        assert isinstance(label, str)


class TestLoadTrainedModels:
    """Tests for the load_trained_models function, which loads trained model .pkl files from a directory and returns a dict of model instances."""
    def test_loads_present_models(self, trained_models_dir):
        models = load_trained_models(trained_models_dir)
        assert set(models.keys()) >= {"Weighted KNN", "Random Forest"}

    def test_loaded_models_have_predict_method(self, trained_models_dir):
        models = load_trained_models(trained_models_dir)
        for model in models.values():
            assert hasattr(model, "predict")

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        assert load_trained_models(str(tmp_path)) == {}

    def test_only_existing_models_should_be_loaded(self, trained_models_dir):
        os.remove(os.path.join(trained_models_dir, "random_forest.pkl"))
        models = load_trained_models(trained_models_dir)
        assert "Weighted KNN" in models
        assert "Random Forest" not in models


class TestEvaluateModels:
    """
    Tests for the evaluate_models function, which takes trained model instances and test data, makes predictions, 
    and computes evaluation metrics like accuracy and confusion matrix.
    """
    def test_returns_entry_per_model(self, eval_results):
        results, _, _, _, _ = eval_results
        assert len(results) >= 2

    def test_result_has_required_fields(self, eval_results):
        results, _, _, _, _ = eval_results
        for result in results.values():
            assert {"y_pred", "accuracy", "confusion_matrix", "report_str"} \
                   .issubset(result.keys())

    def test_accuracy_in_valid_range(self, eval_results):
        results, _, _, _, _ = eval_results
        for result in results.values():
            assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_is_square(self, eval_results):
        results, _, _, _, _ = eval_results
        for result in results.values():
            cm = result["confusion_matrix"]
            assert cm.shape[0] == cm.shape[1]


class TestFeatureImportanceAnalysis:
    """Tests for the feature_importance_analysis function, which generates a PDF report of the most important features (APs) according to the Random Forest model."""
    def test_produces_pdf(self, trained_models_dir, test_db, tmp_path):
        X_train, _, _, _, features, _, _ = load_and_preprocess(test_db)
        out = str(tmp_path)
        feature_importance_analysis(
            features, trained_models_dir, out, db_path=test_db, top_n=5
        )
        assert os.path.exists(os.path.join(out, "feature_importance.pdf"))

    def test_works_without_db_path(self, trained_models_dir, test_db, tmp_path):
        """Should fall back to raw BSSID labels when db_path is missing."""
        _, _, _, _, features, _, _ = load_and_preprocess(test_db)
        out = str(tmp_path)
        feature_importance_analysis(
            features, trained_models_dir, out, db_path=None, top_n=5
        )
        assert os.path.exists(os.path.join(out, "feature_importance.pdf"))

    def test_skipped_when_random_forest_missing(
        self, trained_models_dir, test_db, tmp_path
    ):
        """If RF model file isn't there, function returns early without raising."""
        os.remove(os.path.join(trained_models_dir, "random_forest.pkl"))
        _, _, _, _, features, _, _ = load_and_preprocess(test_db)
        out = str(tmp_path)
        feature_importance_analysis(features, trained_models_dir, out)
        # No PDF produced
        assert not os.path.exists(os.path.join(out, "feature_importance.pdf"))


class TestLearningCurveAnalysis:
    """Tests for the learning_curve_analysis function, which generates a PDF report of model performance as a function of training set size."""
    def test_produces_pdf(self, model_config, test_db, tmp_path):
        X_train, _, y_train, _, _, _, _ = load_and_preprocess(test_db)
        out = str(tmp_path)

        with open(model_config, "r") as f:
            config = json.load(f)

        # Prevent MLP from creating an internal validation split on tiny CV folds
        config["MLP"]["params"]["early_stopping"] = False

        with open(model_config, "w") as f:
            json.dump(config, f, indent=2)

        with patch("src.training.model_evaluation.MODEL_CONFIG_PATH", model_config):
            learning_curve_analysis(X_train, y_train, out)

        assert os.path.exists(os.path.join(out, "learning_curves.pdf"))


class TestMisclassificationAnalysis:
    """
    Tests for the misclassification_analysis function, which generates a PDF report of the most common misclassification pairs 
    (EX. Room A predicted as Room B) based on the confusion matrix and label encoder.
    """
    def test_handles_no_errors_gracefully(self, trained_models_dir, test_db, tmp_path):
        X_train, X_test, y_train, y_test, features, le, scaler = load_and_preprocess(test_db)
        models = load_trained_models(trained_models_dir)

        # Perfect results dictionary
        perfect_results = {
            name: {
                "y_pred": y_test.copy(),
                "accuracy": 1.0,
                "confusion_matrix": __import__("numpy").eye(len(le.classes_), dtype=int),
                "report_str": "",
            }
            for name in models
        }

        out = str(tmp_path)
        misclassification_analysis(perfect_results, le, out)
        # No PDF generated
        assert not os.path.exists(os.path.join(out, "misclassification_pairs.pdf"))


class TestParseArgs:
    """Tests for the parse_args function, which parses command-line arguments for the model evaluation script."""
    def test_defaults(self):
        with patch("sys.argv", ["prog"]):
            args = parse_args()

        assert args.db is not None
        assert args.models_dir is not None
        assert args.output is not None

    def test_custom_db(self):
        with patch("sys.argv", ["prog", "--db", "custom.db"]):
            args = parse_args()

        assert args.db == "custom.db"

    def test_custom_models_dir(self):
        with patch("sys.argv", ["prog", "--models-dir", "models/"]):
            args = parse_args()

        assert args.models_dir == "models/"

    def test_custom_output(self):
        with patch("sys.argv", ["prog", "--output", "out/"]):
            args = parse_args()

        assert args.output == "out/"

    def test_all_custom_args(self):
        with patch("sys.argv", [
            "prog",
            "--db", "my.db",
            "--models-dir", "models/",
            "--output", "out/"
        ]):
            args = parse_args()

        assert args.db == "my.db"
        assert args.models_dir == "models/"
        assert args.output == "out/"
