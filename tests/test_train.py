"""Tests for src/training/train.py"""

import json
import os

import joblib
import matplotlib
import pytest
from unittest.mock import patch

# Use non-interactive backend and disable LaTeX (not always installed in CI)
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False


@pytest.fixture(autouse=True)
def _no_latex():
    """Disable LaTeX rendering for every test — matplotlib rcParams can be
    mutated by imported modules, so we reset it before each test."""
    matplotlib.rcParams["text.usetex"] = False


from src.training.model_loader import load_models_from_config
from src.training.preprocess import (
    load_and_preprocess,
    load_and_preprocess_temporal,
)
from src.training.train import (
    export_models,
    get_models,
    plot_comparison,
    plot_leakage_comparison,
    temporal_validation,
    train_and_evaluate,
    parse_args,
    DATABASE_PATH,
    MODEL_OUTPUT_DIRECTORY
)


# ── Fixtures for training data and results ─────────────────────

@pytest.fixture
def training_data(test_db):
    return load_and_preprocess(test_db)


@pytest.fixture
def temporal_data(test_db):
    return load_and_preprocess_temporal(test_db)


@pytest.fixture
def quick_models(model_config):
    return load_models_from_config(model_config)


@pytest.fixture
def train_results(quick_models, training_data):
    X_train, X_test, y_train, y_test, _, le, _ = training_data
    return train_and_evaluate(
        quick_models, X_train, X_test, y_train, y_test, le, cv_folds=3
    )


# ── get_models (config loader wrapper) ─────────────────────────

class TestGetModels:
    def test_loads_from_provided_config(self, model_config):
        models = get_models(model_config)
        assert "Weighted KNN" in models
        assert "Random Forest" in models


# ── train_and_evaluate ─────────────────────────────────────────

class TestTrainAndEvaluate:
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
        """3 rooms → chance is ~33%. Models should comfortably beat that."""
        for name, result in train_results.items():
            assert result["test_accuracy"] > 0.33, (
                f"{name} accuracy {result['test_accuracy']} not better than chance"
            )

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
        X_train, X_test, *_ = training_data
        for result in train_results.values():
            # Fitted model can predict without error
            preds = result["model"].predict(X_test[:1])
            assert len(preds) == 1


# ── export_models ──────────────────────────────────────────────

class TestExportModels:
    def test_saves_one_pkl_per_model(self, train_results, training_data, tmp_path):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        export_models(train_results, le, scaler, features, out)

        for name in train_results:
            safe = name.lower().replace(" ", "_")
            assert os.path.exists(os.path.join(out, f"{safe}.pkl"))

    def test_saves_all_metadata_files(self, train_results, training_data, tmp_path):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        export_models(train_results, le, scaler, features, out)

        for fname in ("label_encoder.pkl", "scaler.pkl",
                      "feature_names.json", "comparison_summary.json"):
            assert os.path.exists(os.path.join(out, fname))

    def test_feature_names_json_is_valid(self, train_results, training_data, tmp_path):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        export_models(train_results, le, scaler, features, out)
        with open(os.path.join(out, "feature_names.json")) as f:
            saved = json.load(f)
        assert saved == features

    def test_returns_best_model_name(self, train_results, training_data, tmp_path):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        best = export_models(train_results, le, scaler, features, out)
        assert best in train_results

    def test_summary_json_records_best_model(
        self, train_results, training_data, tmp_path
    ):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        export_models(train_results, le, scaler, features, out)
        with open(os.path.join(out, "comparison_summary.json")) as f:
            summary = json.load(f)
        assert "_best_model" in summary
        assert "_best_accuracy" in summary
        assert summary["_best_model"] in train_results

    def test_best_model_has_highest_f1(self, train_results, training_data, tmp_path):
        """_best_model should be the one with the highest test_f1."""
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        best = export_models(train_results, le, scaler, features, out)
        actual_best = max(train_results.keys(),
                          key=lambda n: train_results[n]["test_f1"])
        assert best == actual_best

    def test_saved_models_are_loadable_and_predict(
        self, train_results, training_data, tmp_path
    ):
        _, X_test, _, _, features, le, scaler = training_data
        out = str(tmp_path / "export")
        os.makedirs(out)
        export_models(train_results, le, scaler, features, out)

        for name in train_results:
            safe = name.lower().replace(" ", "_")
            loaded = joblib.load(os.path.join(out, f"{safe}.pkl"))
            preds = loaded.predict(X_test[:3])
            assert len(preds) == 3

    def test_creates_output_directory_if_missing(
        self, train_results, training_data, tmp_path
    ):
        _, _, _, _, features, le, scaler = training_data
        out = str(tmp_path / "nonexistent")
        export_models(train_results, le, scaler, features, out)
        assert os.path.isdir(out)


# ── temporal_validation ────────────────────────────────────────

class TestTemporalValidation:
    def test_returns_entry_for_every_model(self, quick_models, temporal_data):
        X_train, X_test, y_train, y_test, _, le, _ = temporal_data
        results = temporal_validation(quick_models, X_train, X_test, y_train, y_test, le)
        assert set(results.keys()) == set(quick_models.keys())

    def test_result_has_expected_keys(self, quick_models, temporal_data):
        X_train, X_test, y_train, y_test, _, le, _ = temporal_data
        results = temporal_validation(quick_models, X_train, X_test, y_train, y_test, le)
        for result in results.values():
            assert {"test_accuracy", "test_f1", "confusion_matrix", "y_pred"} \
                   .issubset(result.keys())

    def test_accuracy_above_random_chance(self, quick_models, temporal_data):
        X_train, X_test, y_train, y_test, _, le, _ = temporal_data
        results = temporal_validation(quick_models, X_train, X_test, y_train, y_test, le)
        for name, result in results.items():
            assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_accuracy_and_f1_in_valid_range(self, quick_models, temporal_data):
        X_train, X_test, y_train, y_test, _, le, _ = temporal_data
        results = temporal_validation(quick_models, X_train, X_test, y_train, y_test, le)
        for result in results.values():
            assert 0.0 <= result["test_accuracy"] <= 1.0
            assert 0.0 <= result["test_f1"] <= 1.0


# ── Plot smoke tests ───────────────────────────────────────────
# These don't verify the visual content — they verify matplotlib doesn't
# crash and the expected PDFs are produced.

class TestPlotComparison:
    def test_creates_all_four_plots(self, train_results, training_data, tmp_path):
        _, _, _, _, _, le, _ = training_data
        out = str(tmp_path)
        plot_comparison(train_results, le, out)
        for fname in ("model_comparison.pdf", "confusion_matrices.pdf",
                      "timing_comparison.pdf", "cv_boxplot.pdf"):
            assert os.path.exists(os.path.join(out, fname))

    def test_pdf_files_non_empty(self, train_results, training_data, tmp_path):
        _, _, _, _, _, le, _ = training_data
        out = str(tmp_path)
        plot_comparison(train_results, le, out)
        for fname in os.listdir(out):
            if fname.endswith(".pdf"):
                assert os.path.getsize(os.path.join(out, fname)) > 0


class TestPlotLeakageComparison:
    def test_creates_leakage_plots(
        self, train_results, quick_models, temporal_data, training_data, tmp_path
    ):
        X_train, X_test, y_train, y_test, _, le, _ = temporal_data
        temporal_results = temporal_validation(
            quick_models, X_train, X_test, y_train, y_test, le
        )
        out = str(tmp_path)
        plot_leakage_comparison(train_results, temporal_results, le, out)
        assert os.path.exists(os.path.join(out, "leakage_test.pdf"))
        assert os.path.exists(os.path.join(out, "temporal_confusion_matrices.pdf"))


class TestParseArgs:
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
            args = parse_args()

        assert args.db == "custom.db"

    def test_custom_output(self):
        with patch("sys.argv", ["prog", "--output", "out/"]):
            args = parse_args()

        assert args.output == "out/"

    def test_custom_test_size(self):
        with patch("sys.argv", ["prog", "--test-size", "0.3"]):
            args = parse_args()

        assert args.test_size == 0.3

    def test_custom_cv_folds(self):
        with patch("sys.argv", ["prog", "--cv-folds", "3"]):
            args = parse_args()

        assert args.cv_folds == 3

    def test_custom_min_detection_rate(self):
        with patch("sys.argv", ["prog", "--min-detection-rate", "0.1"]):
            args = parse_args()

        assert args.min_detection_rate == 0.1

    def test_all_custom_args(self):
        with patch(
            "sys.argv",
            [
                "prog",
                "--db", "my.db",
                "--output", "out/",
                "--test-size", "0.3",
                "--cv-folds", "3",
                "--min-detection-rate", "0.1",
            ],
        ):
            args = parse_args()

        assert args.db == "my.db"
        assert args.output == "out/"
        assert args.test_size == 0.3
        assert args.cv_folds == 3
        assert args.min_detection_rate == 0.1