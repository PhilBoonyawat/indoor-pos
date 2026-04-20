"""Tests for src/training/train.py"""

import json
import os
from unittest.mock import patch

import joblib
import pytest


from src.training.preprocess import load_and_preprocess_temporal  # noqa: E402
from src.training.train import (  # noqa: E402
    DATABASE_PATH,
    MODEL_OUTPUT_DIRECTORY,
    export_models,
    get_models,
    parse_args,
    plot_comparison,
    plot_leakage_comparison,
    temporal_validation,
    train_and_evaluate,
)

@pytest.fixture
def temporal_data(test_db):
    """
    Preprocessed data using the time-ordered split (leakage detection).
    
    Args:
        test_db: Fixture providing path to test database

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, features, label_encoder, scaler) where:
            - X_train, X_test: Feature matrices for training and testing
            - y_train, y_test: Label vectors for training and testing
            - features: List of feature names (BSSIDs)
            - label_encoder: Fitted LabelEncoder instance
            - scaler: Fitted StandardScaler instance
    """
    return load_and_preprocess_temporal(test_db)


@pytest.fixture
def train_results(quick_models, training_data):
    """
    Trains every model once -> shared by TestTrainAndEvaluate, TestExportModels,
    and TestPlotComparison so we don't re-fit models for each test.

    Args:
        quick_models: Fixture providing the model instances to train
        training_data: Fixture providing preprocessed training and testing data

    Returns:
        Dictionary of {model_name: result_dict} where result_dict contains:
            - cv_scores: list of cross-validation scores
            - cv_mean: mean CV score
            - cv_std: std dev of CV scores
            - test_accuracy: accuracy on the test set
            - test_f1: F1 score on the test set
            - confusion_matrix: confusion matrix on the test set
            - report: classification report dict
            - report_str: classification report as a string
            - train_time: time taken to train the model
            - predict_time: time taken to predict on the test set
            - y_pred: predicted labels for the test set
            - model: the fitted model instance
            - description: human-readable description of the model configuration
    """
    X_train, X_test, y_train, y_test, _, le, _ = training_data
    return train_and_evaluate(
        quick_models, X_train, X_test, y_train, y_test, le, cv_folds=3
    )


@pytest.fixture
def temporal_results(quick_models, temporal_data):
    """
    Run temporal_validation once per test class that needs it.

    Args:
        quick_models: Fixture providing the model instances to validate
        temporal_data: Fixture providing preprocessed data with time-ordered split

    Returns:
        Dictionary of {model_name: result_dict} where result_dict contains:
            - test_accuracy: accuracy on the temporal test set
            - test_f1: F1 score on the temporal test set
            - confusion_matrix: confusion matrix on the temporal test set
            - y_pred: predicted labels for the temporal test set    
    """
    X_train, X_test, y_train, y_test, _, le, _ = temporal_data
    return temporal_validation(quick_models, X_train, X_test, y_train, y_test, le)


@pytest.fixture
def export_dir(train_results, training_data, tmp_path):
    """
    Run export_models once into a temporary directory and yield
    (directory_path, best_model_name). Shared across TestExportModels tests.

    Args:
        train_results: Fixture providing the results of training all models
        training_data: Fixture providing preprocessed training and testing data
        tmp_path: pytest fixture providing a temporary directory path   

    Returns:
        Tuple of (export_directory_path, best_model_name) where:
            - export_directory_path: Path to the directory where models and metadata were exported      
            - best_model_name: Name of the model with the highest F1 score, as determined by export_models
    """
    _, _, _, _, features, le, scaler = training_data
    out = str(tmp_path / "export")
    best = export_models(train_results, le, scaler, features, out)
    return out, best


class TestGetModels:
    """Tests for the get_models function, which loads model configurations and creates model instances."""

    def test_loads_from_provided_config(self, model_config):
        models = get_models(model_config)
        assert "Weighted KNN" in models
        assert "Random Forest" in models

class TestTrainAndEvaluate:
    """Tests for the train_and_evaluate function, which trains each model and evaluates it on the test set, returning a comprehensive results dictionary."""

    def test_returns_entry_for_every_model(self, train_results, quick_models):
        assert set(train_results.keys()) == set(quick_models.keys())

    def test_result_has_all_expected_keys(self, train_results):
        expected = {
            "cv_scores", "cv_mean", "cv_std",
            "test_accuracy", "test_f1",
            "confusion_matrix", "report", "report_str",
            "train_time", "predict_time",
            "y_pred", "model", "description",
        }
        for result in train_results.values():
            assert expected.issubset(result.keys())

    def test_accuracy_better_than_random_chance(self, train_results):
        """4 rooms -> chance is ~25%. Models should comfortably beat that."""
        for result in train_results.values():
            assert result["test_accuracy"] > 0.25

    def test_cv_scores_match_folds_parameter(self, train_results):
        for result in train_results.values():
            assert len(result["cv_scores"]) == 3

    def test_confusion_matrix_is_square(self, train_results):
        for result in train_results.values():
            cm = result["confusion_matrix"]
            assert cm.shape == (4, 4)

    def test_times_are_non_negative(self, train_results):
        for result in train_results.values():
            assert result["train_time"] >= 0
            assert result["predict_time"] >= 0

    def test_model_is_fitted(self, train_results, training_data):
        """A fitted model can predict on new data without raising."""
        _, X_test, *_ = training_data
        for result in train_results.values():
            preds = result["model"].predict(X_test[:1])
            assert len(preds) == 1


class TestExportModels:
    """Tests for the export_models function, which saves the trained models and metadata to disk and returns the name of the best model."""

    def test_saves_one_pkl_per_model(self, train_results, export_dir):
        out, _ = export_dir
        for name in train_results:
            safe = name.lower().replace(" ", "_")
            assert os.path.exists(os.path.join(out, f"{safe}.pkl"))

    def test_saves_all_metadata_files(self, export_dir):
        out, _ = export_dir
        for fname in ("label_encoder.pkl", "scaler.pkl",
                      "feature_names.json", "comparison_summary.json"):
            assert os.path.exists(os.path.join(out, fname))

    def test_feature_names_json_matches_features(self, export_dir, training_data):
        out, _ = export_dir
        _, _, _, _, features, _, _ = training_data
        with open(os.path.join(out, "feature_names.json")) as f:
            assert json.load(f) == features

    def test_returns_best_model_name(self, export_dir, train_results):
        _, best = export_dir
        assert best in train_results

    def test_summary_json_records_best_model(self, export_dir, train_results):
        out, _ = export_dir
        with open(os.path.join(out, "comparison_summary.json")) as f:
            summary = json.load(f)
        assert "_best_model" in summary
        assert "_best_accuracy" in summary
        assert summary["_best_model"] in train_results

    def test_best_model_has_highest_f1(self, export_dir, train_results):
        _, best = export_dir
        actual_best = max(train_results.keys(),
                          key=lambda n: train_results[n]["test_f1"])
        assert best == actual_best

    def test_saved_models_are_loadable_and_predict(self, export_dir, train_results, training_data):
        out, _ = export_dir
        _, X_test, *_ = training_data
        for name in train_results:
            safe = name.lower().replace(" ", "_")
            loaded = joblib.load(os.path.join(out, f"{safe}.pkl"))
            preds = loaded.predict(X_test[:3])
            assert len(preds) == 3

    def test_creates_output_directory_if_missing(self, train_results, training_data, tmp_path):
        """export_models should create the output dir if it doesn't exist.
        Not using `export_dir` fixture because that pre-creates the dir."""
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "nonexistent")
        export_models(train_results, le, scaler, features, out)
        assert os.path.isdir(out)


class TestTemporalValidation:
    """Validates on a temporally-ordered split to detect data leakage."""

    def test_returns_entry_for_every_model(self, temporal_results, quick_models):
        assert set(temporal_results.keys()) == set(quick_models.keys())

    def test_result_has_expected_keys(self, temporal_results):
        expected = {"test_accuracy", "test_f1", "confusion_matrix", "y_pred"}
        for result in temporal_results.values():
            assert expected.issubset(result.keys())

    def test_accuracy_and_f1_in_valid_range(self, temporal_results):
        for result in temporal_results.values():
            assert 0.0 <= result["test_accuracy"] <= 1.0
            assert 0.0 <= result["test_f1"] <= 1.0


class TestPlotComparison:
    """Tests for the plot_comparison function, which generates PDF plots comparing model performance."""
    def test_creates_all_four_plots(self, train_results, training_data, tmp_path):
        _, _, _, _, _, le, _ = training_data
        plot_comparison(train_results, le, str(tmp_path))
        for fname in ("model_comparison.pdf", "confusion_matrices.pdf",
                      "timing_comparison.pdf", "cv_boxplot.pdf"):
            assert os.path.exists(os.path.join(tmp_path, fname))

    def test_pdf_files_non_empty(self, train_results, training_data, tmp_path):
        _, _, _, _, _, le, _ = training_data
        plot_comparison(train_results, le, str(tmp_path))
        for fname in os.listdir(tmp_path):
            if fname.endswith(".pdf"):
                assert os.path.getsize(tmp_path / fname) > 0


class TestPlotLeakageComparison:
    """Tests for the plot_leakage_comparison function, which generates PDF plots comparing standard vs temporal validation results to detect leakage."""
    def test_creates_leakage_plots(
        self, train_results, temporal_results, training_data, tmp_path
    ):
        _, _, _, _, _, le, _ = training_data
        out = str(tmp_path)
        plot_leakage_comparison(train_results, temporal_results, le, out)
        assert os.path.exists(os.path.join(out, "leakage_test.pdf"))
        assert os.path.exists(os.path.join(out, "temporal_confusion_matrices.pdf"))


class TestParseArgs:
    """Tests for the parse_args function, which parses command-line arguments for the training script."""

    def test_defaults(self):
        with patch("sys.argv", ["prog"]):
            args = parse_args()
        assert args.db == DATABASE_PATH
        assert args.output == MODEL_OUTPUT_DIRECTORY
        assert args.test_size == 0.2
        assert args.cv_folds == 5
        assert args.min_detection_rate == 0.05

    def test_custom_db(self):
        with patch("sys.argv", ["prog", "--db", "custom.db"]):
            assert parse_args().db == "custom.db"

    def test_custom_output(self):
        with patch("sys.argv", ["prog", "--output", "out/"]):
            assert parse_args().output == "out/"

    def test_custom_test_size(self):
        with patch("sys.argv", ["prog", "--test-size", "0.3"]):
            assert parse_args().test_size == 0.3

    def test_custom_cv_folds(self):
        with patch("sys.argv", ["prog", "--cv-folds", "3"]):
            assert parse_args().cv_folds == 3

    def test_custom_min_detection_rate(self):
        with patch("sys.argv", ["prog", "--min-detection-rate", "0.1"]):
            assert parse_args().min_detection_rate == 0.1

    def test_all_custom_args(self):
        argv = [
            "prog",
            "--db", "my.db",
            "--output", "out/",
            "--test-size", "0.3",
            "--cv-folds", "3",
            "--min-detection-rate", "0.1",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        assert args.db == "my.db"
        assert args.output == "out/"
        assert args.test_size == 0.3
        assert args.cv_folds == 3
        assert args.min_detection_rate == 0.1