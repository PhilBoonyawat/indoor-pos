"""Tests for src/training/data_efficiency_analysis.py"""

import os
import pytest
from unittest.mock import patch

from src.training.data_efficiency_analysis import (  
    get_models,
    plot_bar_comparison,
    plot_degradation_curves,
    plot_per_room_degradation,
    run_experiment,
    parse_args,
    DATABASE_DEFAULT_PATH,
    OUTPUT_DIR
)
from src.training.preprocess import load_and_preprocess 


@pytest.fixture
def experiment_data(test_db):
    """
    Training/test data and label encoder ready for run_experiment.

    Args:
        test_db: Fixture providing path to test database.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_encoder)
    """
    X_train, X_test, y_train, y_test, _, label_encoder, _ = load_and_preprocess(test_db)
    return X_train, X_test, y_train, y_test, label_encoder


@pytest.fixture
def experiment_results(experiment_data, model_config):
    """
    Run the experiment once; reuse results across plot tests.

    Args:
        experiment_data: Fixture providing preprocessed training/test data and label encoder.
        model_config: Fixture providing path to model_configs.json with test hyperparameters.

    Returns:
        Tuple of (experiment_results, fractions, label_encoder) where:
        - experiment_results is the dict output from run_experiment
        - fractions is the list of training data fractions used in the experiment
        - label_encoder is the LabelEncoder instance used to decode class labels    
    """
    X_train, X_test, y_train, y_test, le = experiment_data
    fractions = [0.25, 0.5, 1.0]
    return run_experiment(X_train, X_test, y_train, y_test, le, fractions,
                          config_path=model_config), fractions, le


class TestGetModels:
    """Tests for the get_models function, which reads model configurations and returns instantiated model objects."""
    def test_returns_all_four_models(self, model_config):
        models = get_models(model_config)
        assert set(models.keys()) == {"Weighted KNN", "Random Forest", "SVM", "MLP"}

    def test_returns_bare_instances_not_tuples(self, model_config):
        """
        Unlike load_models_from_config which returns (model, desc),
        get_models unwraps them to bare model instances.
        """
        models = get_models(model_config)
        for model in models.values():
            # bare sklearn estimators have a .fit method
            assert hasattr(model, "fit")
            # should NOT be tuples
            assert not isinstance(model, tuple)

class TestRunExperiment:
    """
    Tests for the run_experiment function, which trains and 
    evaluates models on different fractions of the training data and returns performance metrics.
    """
    def test_returns_entry_per_model(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[1.0], config_path=model_config)
        assert set(results.keys()) == {"Weighted KNN", "Random Forest", "SVM", "MLP"}

    def test_each_fraction_has_entry(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        fractions = [0.25, 0.5, 1.0]
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=fractions, config_path=model_config)
        for model_results in results.values():
            assert set(model_results.keys()) == set(fractions)

    def test_result_has_expected_fields(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[0.5], config_path=model_config)
        for model_results in results.values():
            r = model_results[0.5]
            assert set(r.keys()) == {"accuracy", "f1", "n_train", "report"}

    def test_n_train_reflects_fraction(self, experiment_data, model_config):
        """For fraction 0.5 with 48 training samples, n_train should be ~24."""
        X_train, X_test, y_train, y_test, le = experiment_data
        total = len(X_train)
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[0.5], config_path=model_config)
        for model_results in results.values():
            n = model_results[0.5]["n_train"]
            assert abs(n - total * 0.5) <= 2

    def test_full_fraction_uses_all_training_data(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[1.0], config_path=model_config)
        for model_results in results.values():
            assert model_results[1.0]["n_train"] == len(X_train)

    def test_accuracy_and_f1_in_valid_range(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[1.0], config_path=model_config)
        for model_results in results.values():
            r = model_results[1.0]
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["f1"] <= 1.0

    def test_classification_report_includes_rooms(self, experiment_data, model_config):
        X_train, X_test, y_train, y_test, le = experiment_data
        results = run_experiment(X_train, X_test, y_train, y_test, le,
                                 fractions=[1.0], config_path=model_config)
        rooms = set(le.classes_)
        for model_results in results.values():
            report = model_results[1.0]["report"]
            assert rooms.issubset(set(report.keys()))


class TestPlotDegradationCurves:
    """Tests PDF is produced and non-empty."""
    
    def test_produces_pdf(self, experiment_results, tmp_path):
        results, fractions, _ = experiment_results
        plot_degradation_curves(results, fractions, str(tmp_path))
        assert os.path.exists(tmp_path / "data_efficiency_curves.pdf")

    def test_pdf_is_non_empty(self, experiment_results, tmp_path):
        results, fractions, _ = experiment_results
        plot_degradation_curves(results, fractions, str(tmp_path))
        assert (tmp_path / "data_efficiency_curves.pdf").stat().st_size > 0


class TestPlotBarComparison:
    """Tests PDF is produced."""
    def test_produces_pdf(self, experiment_results, tmp_path):
        results, _, _ = experiment_results
        plot_bar_comparison(results, str(tmp_path))
        assert os.path.exists(tmp_path / "data_efficiency_bars.pdf")


class TestPlotPerRoomDegradation:
    """Tests PDF is produced."""
    def test_produces_pdf(self, experiment_results, tmp_path):
        results, _, le = experiment_results
        plot_per_room_degradation(results, le, str(tmp_path))
        assert os.path.exists(tmp_path / "data_efficiency_per_room.pdf")

class TestParseArgs:
    """Tests for the parse_args function, which parses command-line arguments for the data efficiency analysis script."""
    def test_defaults(self):
        with patch("sys.argv", ["prog"]):
            args = parse_args()

        assert args.db == DATABASE_DEFAULT_PATH
        assert args.output == OUTPUT_DIR

    def test_custom_db(self):
        with patch("sys.argv", ["prog", "--db", "custom.db"]):
            args = parse_args()

        assert args.db == "custom.db"
        assert args.output == OUTPUT_DIR  

    def test_custom_output(self):
        with patch("sys.argv", ["prog", "--output", "out/"]):
            args = parse_args()

        assert args.db == DATABASE_DEFAULT_PATH  
        assert args.output == "out/"

    def test_all_custom_args(self):
        with patch(
            "sys.argv",
            [
                "prog",
                "--db", "my.db",
                "--output", "out/",
            ],
        ):
            args = parse_args()

        assert args.db == "my.db"
        assert args.output == "out/"