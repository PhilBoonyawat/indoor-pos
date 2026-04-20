"""Tests for src/training/hyperparameter_tuning.py"""

import os

import numpy as np
import pytest

from src.training.hyperparameter_tuning import (  # noqa: E402
    get_param_grids,
    plot_knn_lines,
    plot_mlp_grid,
    plot_rf_lines,
    plot_svm_lines,
    run_grid_search,
)
from src.training.model_loader import load_models_from_config  # noqa: E402
from src.training.preprocess import load_and_preprocess  # noqa: E402


class TestGetParamGrids:
    """Tests for the get_param_grids function, which returns the hyperparameter grids for each model."""
    def test_has_all_four_model_keys(self):
        grids = get_param_grids()
        assert set(grids.keys()) == {"Weighted KNN", "Random Forest", "SVM", "MLP"}

    def test_each_grid_is_non_empty_dict(self):
        grids = get_param_grids()
        for name, grid in grids.items():
            assert isinstance(grid, dict)
            assert len(grid) > 0, f"{name} grid is empty"

    def test_knn_grid_has_expected_hyperparameters(self):
        grid = get_param_grids()["Weighted KNN"]
        assert "n_neighbors" in grid
        assert "weights" in grid
        assert "metric" in grid

    def test_rf_grid_has_expected_hyperparameters(self):
        grid = get_param_grids()["Random Forest"]
        assert "n_estimators" in grid
        assert "max_depth" in grid

    def test_svm_grid_has_expected_hyperparameters(self):
        grid = get_param_grids()["SVM"]
        assert "C" in grid
        assert "gamma" in grid

    def test_mlp_grid_has_expected_hyperparameters(self):
        grid = get_param_grids()["MLP"]
        assert "hidden_layer_sizes" in grid
        assert "alpha" in grid


class TestRunGridSearch:
    """Tests for the run_grid_search function, which performs GridSearchCV for each model and returns the best results."""

    @pytest.fixture
    def minimal_grids(self):
        """
        Single-combination grids to keep the test fast.
        
        Returns:
            dict of {model_name: param_grid_dict} where each param_grid_dict has one value per hyperparameter.
        """
        return {
            "Weighted KNN": {"n_neighbors": [3]},
            "Random Forest": {"n_estimators": [10]},
            "SVM": {"C": [1]},
            "MLP": {"alpha": [1e-4]},
        }

    def test_returns_entry_per_model(self, test_db, model_config, minimal_grids):
        X_train, _, y_train, _, _, _, _ = load_and_preprocess(test_db)
        models = load_models_from_config(model_config)
        results = run_grid_search(models, minimal_grids, X_train, y_train)
        assert set(results.keys()) == set(models.keys())

    def test_result_has_expected_keys(self, test_db, model_config, minimal_grids):
        X_train, _, y_train, _, _, _, _ = load_and_preprocess(test_db)
        models = load_models_from_config(model_config)
        results = run_grid_search(models, minimal_grids, X_train, y_train)

        for r in results.values():
            assert set(r.keys()) == {"best_model", "best_params", "best_score", "cv_results"}

    def test_best_score_in_valid_range(self, test_db, model_config, minimal_grids):
        X_train, _, y_train, _, _, _, _ = load_and_preprocess(test_db)
        models = load_models_from_config(model_config)
        results = run_grid_search(models, minimal_grids, X_train, y_train)

        for r in results.values():
            assert 0.0 <= r["best_score"] <= 1.0

    def test_best_model_is_fitted(self, test_db, model_config, minimal_grids):
        """best_model must be usable for prediction after fitting."""
        X_train, X_test, y_train, _, _, _, _ = load_and_preprocess(test_db)
        models = load_models_from_config(model_config)
        results = run_grid_search(models, minimal_grids, X_train, y_train)

        for r in results.values():
            preds = r["best_model"].predict(X_test[:1])
            assert len(preds) == 1


def _make_fake_knn_cv_results():
    """
    Produces cv_results for: metric × weights × n_neighbors.

    Returns:
        dict of arrays mimicking GridSearchCV cv_results_ structure, with keys like 'param_metric', 'param_weights', 'param_n_neighbors', and 'mean_test_score'.
    """
    rows = []
    for metric in ["manhattan", "euclidean"]:
        for weight in ["uniform", "distance"]:
            for n in [3, 5, 7]:
                rows.append({
                    "param_metric": metric,
                    "param_weights": weight,
                    "param_n_neighbors": n,
                    "mean_test_score": 0.8 + 0.05 * (n / 10),
                })
    # Convert to the list-of-arrays format GridSearchCV uses
    return {
        "param_metric": np.array([r["param_metric"] for r in rows]),
        "param_weights": np.array([r["param_weights"] for r in rows]),
        "param_n_neighbors": np.array([r["param_n_neighbors"] for r in rows]),
        "mean_test_score": np.array([r["mean_test_score"] for r in rows]),
    }


def _make_fake_rf_cv_results():
    """
    Produces cv_results for: max_features × max_depth × n_estimators.

    Returns:
        dict of arrays mimicking GridSearchCV cv_results_ structure, with keys like 'param_max_features', 'param_max_depth', 'param_n_estimators', and 'mean_test_score'.
    """
    
    rows = []
    for max_feat in ["sqrt", "log2"]:
        for depth in [None, 10, 20]:
            for trees in [50, 100]:
                rows.append({
                    "param_max_features": max_feat,
                    "param_max_depth": depth,
                    "param_n_estimators": trees,
                    "param_min_samples_split": 5,
                    "param_min_samples_leaf": 1,
                    "mean_test_score": 0.85,
                })
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _make_fake_svm_cv_results():
    """
    Produces cv_results for: C × gamma. 

    Returns:
        dict of arrays mimicking GridSearchCV cv_results_ structure, with keys like 'param_C', 'param_gamma', and 'mean_test_score'.
    """
    
    rows = []
    for C in [0.1, 1, 10]:
        for gamma in ["scale", 0.01, 0.1]:
            rows.append({
                "param_C": C,
                "param_gamma": gamma,
                "mean_test_score": 0.9,
            })
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def _make_fake_mlp_cv_results():
    """
    Produces cv_results for: hidden_layer_sizes × alpha × learning_rate_init.

    Returns:
        dict of arrays mimicking GridSearchCV cv_results_ structure, with keys like 'param_hidden_layer_sizes', 'param_alpha', 'param_learning_rate_init', and 'mean_test_score'.   
    """
    
    rows = []
    for arch in [(64,), (128, 64)]:
        for alpha in [1e-5, 1e-4]:
            for lr in [1e-4, 1e-3]:
                rows.append({
                    "param_hidden_layer_sizes": arch,
                    "param_alpha": alpha,
                    "param_learning_rate_init": lr,
                    "mean_test_score": 0.88,
                })
    return {k: np.array([r[k] for r in rows], dtype=object) for k in rows[0]}


class TestPlotKNN:
    """Tests PDF is produced."""
    def test_produces_pdf(self, tmp_path):
        results = {"Weighted KNN": {"cv_results": _make_fake_knn_cv_results()}}
        plot_knn_lines(results, str(tmp_path))
        assert os.path.exists(tmp_path / "knn_grid_search.pdf")


class TestPlotRF:
    """Tests PDF is produced."""
    def test_produces_pdf(self, tmp_path):
        results = {"Random Forest": {"cv_results": _make_fake_rf_cv_results()}}
        plot_rf_lines(results, str(tmp_path))
        assert os.path.exists(tmp_path / "rf_grid_search.pdf")


class TestPlotSVM:
    """Tests PDF is produced."""
    def test_produces_pdf(self, tmp_path):
        results = {"SVM": {"cv_results": _make_fake_svm_cv_results()}}
        plot_svm_lines(results, str(tmp_path))
        assert os.path.exists(tmp_path / "svm_grid_search.pdf")


class TestPlotMLP:
    """Tests PDF is produced."""
    def test_produces_pdf(self, tmp_path):
        results = {"MLP": {"cv_results": _make_fake_mlp_cv_results()}}
        plot_mlp_grid(results, str(tmp_path))
        assert os.path.exists(tmp_path / "mlp_grid_search.pdf")