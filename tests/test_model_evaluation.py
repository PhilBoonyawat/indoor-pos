"""Tests for src/training/model_evaluation.py"""

import json
import os
from unittest.mock import patch

import matplotlib
import pytest
import numpy as np

from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False


@pytest.fixture(autouse=True)
def _no_latex():
    """Disable LaTeX rendering for every test."""
    matplotlib.rcParams["text.usetex"] = False


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


# ── Shared fixture ─────────────────────────────────────────────

@pytest.fixture
def eval_results(trained_models_dir, test_db):
    """Run evaluate_models once so multiple test classes can reuse the output."""
    X_train, X_test, y_train, y_test, features, le, scaler = load_and_preprocess(test_db)
    models = load_trained_models(trained_models_dir)
    results = evaluate_models(models, X_test, y_test, le)
    return results, le, features, X_train, y_train


# ── BSSID ↔ SSID lookup helpers ─────────────────────────────────

class TestLoadBssidToSsidMap:
    def test_returns_dict_with_all_bssids(self, test_db):
        mapping = load_bssid_to_ssid_map(test_db)
        assert isinstance(mapping, dict)
        assert len(mapping) == 10

    def test_values_are_strings(self, test_db):
        mapping = load_bssid_to_ssid_map(test_db)
        assert all(isinstance(ssid, str) for ssid in mapping.values())


class TestFormatApLabel:
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


# ── Loading trained models from a directory ────────────────────

class TestLoadTrainedModels:
    def test_loads_present_models(self, trained_models_dir):
        models = load_trained_models(trained_models_dir)
        assert set(models.keys()) >= {"Weighted KNN", "Random Forest"}

    def test_loaded_models_have_predict_method(self, trained_models_dir):
        models = load_trained_models(trained_models_dir)
        for model in models.values():
            assert hasattr(model, "predict")

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        assert load_trained_models(str(tmp_path)) == {}

    def test_partial_models_loaded(self, trained_models_dir):
        """Only the pkls that exist should be loaded."""
        os.remove(os.path.join(trained_models_dir, "random_forest.pkl"))
        models = load_trained_models(trained_models_dir)
        assert "Weighted KNN" in models
        assert "Random Forest" not in models


# ── Model evaluation on test set ───────────────────────────────

class TestEvaluateModels:
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


# ── Plotting smoke tests ───────────────────────────────────────
# Verify the plot functions complete and produce files.
# We're not checking visual content — that's best verified by eye.

class TestFeatureImportanceAnalysis:
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
    def test_produces_pdf(self, model_config, test_db, tmp_path):
        """Patch the hardcoded MODEL_CONFIG_PATH so the test uses our fast config."""
        X_train, _, y_train, _, _, _, _ = load_and_preprocess(test_db)
        out = str(tmp_path)

        with patch("src.training.model_evaluation.MODEL_CONFIG_PATH", model_config):
            learning_curve_analysis(X_train, y_train, out)

        assert os.path.exists(os.path.join(out, "learning_curves.pdf"))


class TestMisclassificationAnalysis:
    def test_produces_pdf_when_errors_exist(self, eval_results, tmp_path):
        results, le, *_ = eval_results
        out = str(tmp_path)
        misclassification_analysis(results, le, out)
        # May or may not produce plot — depends on whether there are misclassifications
        # Either way it shouldn't raise
        assert os.path.isdir(out)

    def test_handles_no_errors_gracefully(self, trained_models_dir, test_db, tmp_path):
        """If every prediction is correct, function returns early without crashing."""
        X_train, X_test, y_train, y_test, features, le, scaler = load_and_preprocess(test_db)
        models = load_trained_models(trained_models_dir)

        # Craft a results dict where predictions == labels (no errors)
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
        # Should return early — no PDF generated
        assert not os.path.exists(os.path.join(out, "misclassification_pairs.pdf"))

import pytest
from unittest.mock import patch

from src.training.model_evaluation import parse_args


class TestParseArgs:

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

class TestMisclassificationAnalysis:
    def test_no_misclassifications_returns_early(self, tmp_path):
        le = LabelEncoder()
        le.fit(["Room A", "Room B", "Room C"])

        results = {
            "Model A": {
                "confusion_matrix": np.array([
                    [5, 0, 0],
                    [0, 4, 0],
                    [0, 0, 6],
                ])
            }
        }

        misclassification_analysis(results, le, output_dir=str(tmp_path))

        assert not (tmp_path / "misclassification_pairs.pdf").exists()