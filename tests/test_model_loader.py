"""Tests for src/training/model_loader.py"""

import json

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.training.model_loader import load_models_from_config, save_best_params


# ── Loading config ─────────────────────────────────────────────

class TestLoadModelsFromConfig:
    def test_loads_all_four_models(self, model_config):
        models = load_models_from_config(model_config)
        assert set(models.keys()) == {"Weighted KNN", "Random Forest", "SVM", "MLP"}

    def test_returns_model_instance_and_description(self, model_config):
        models = load_models_from_config(model_config)
        for name, (model, desc) in models.items():
            assert model is not None
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_instantiates_correct_classes(self, model_config):
        models = load_models_from_config(model_config)
        assert isinstance(models["Weighted KNN"][0], KNeighborsClassifier)
        assert isinstance(models["Random Forest"][0], RandomForestClassifier)
        assert isinstance(models["SVM"][0], SVC)
        assert isinstance(models["MLP"][0], MLPClassifier)

    def test_knn_hyperparameters_applied(self, model_config):
        knn, _ = load_models_from_config(model_config)["Weighted KNN"]
        assert knn.n_neighbors == 3
        assert knn.weights == "distance"
        assert knn.metric == "manhattan"

    def test_rf_hyperparameters_applied(self, model_config):
        rf, _ = load_models_from_config(model_config)["Random Forest"]
        assert rf.n_estimators == 10
        assert rf.max_features == "log2"

    def test_svm_probability_enabled(self, model_config):
        svm, _ = load_models_from_config(model_config)["SVM"]
        assert svm.probability is True

    def test_mlp_list_converted_to_tuple(self, model_config):
        """JSON stores layers as a list; sklearn requires a tuple."""
        mlp, _ = load_models_from_config(model_config)["MLP"]
        assert isinstance(mlp.hidden_layer_sizes, tuple)
        assert mlp.hidden_layer_sizes == (64, 32)

    def test_missing_config_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_models_from_config("/no/such/file.json")

    def test_unknown_model_class_is_skipped(self, tmp_path):
        """Unknown class names should be skipped without raising."""
        config = {
            "Weighted KNN": {
                "model": "KNeighborsClassifier",
                "params": {"n_neighbors": 3},
            },
            "Bogus": {"model": "NonexistentClass", "params": {}},
        }
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump(config, f)

        models = load_models_from_config(path)
        assert "Weighted KNN" in models
        assert "Bogus" not in models

    def test_description_excludes_internal_params(self, model_config):
        """random_state, n_jobs, probability, verbose are excluded from desc string."""
        _, desc = load_models_from_config(model_config)["Random Forest"]
        assert "random_state" not in desc
        assert "n_jobs" not in desc

    def test_description_includes_real_hyperparameters(self, model_config):
        _, desc = load_models_from_config(model_config)["Weighted KNN"]
        assert "n_neighbors" in desc
        assert "weights" in desc


# ── Saving best params ─────────────────────────────────────────

class TestSaveBestParams:
    def test_updates_tuned_params(self, model_config):
        save_best_params(
            {"Weighted KNN": {"best_params": {"n_neighbors": 7, "weights": "uniform"}}},
            model_config,
        )
        with open(model_config) as f:
            updated = json.load(f)
        assert updated["Weighted KNN"]["params"]["n_neighbors"] == 7
        assert updated["Weighted KNN"]["params"]["weights"] == "uniform"

    def test_preserves_untuned_params(self, model_config):
        save_best_params(
            {"Weighted KNN": {"best_params": {"n_neighbors": 5}}},
            model_config,
        )
        with open(model_config) as f:
            updated = json.load(f)
        # metric wasn't in best_params — should still equal "manhattan"
        assert updated["Weighted KNN"]["params"]["metric"] == "manhattan"

    def test_tuple_is_converted_to_list_for_json(self, model_config):
        """JSON can't serialise tuples — save must convert."""
        save_best_params(
            {"MLP": {"best_params": {"hidden_layer_sizes": (256, 128)}}},
            model_config,
        )
        with open(model_config) as f:
            updated = json.load(f)
        assert updated["MLP"]["params"]["hidden_layer_sizes"] == [256, 128]

    def test_models_not_in_config_are_skipped(self, model_config):
        """Should log a warning but not raise."""
        save_best_params(
            {"Nonexistent Model": {"best_params": {"foo": "bar"}}},
            model_config,
        )
        # Success = no exception

    def test_other_models_unchanged(self, model_config):
        save_best_params(
            {"Weighted KNN": {"best_params": {"n_neighbors": 99}}},
            model_config,
        )
        with open(model_config) as f:
            updated = json.load(f)
        # SVM config should be untouched
        assert updated["SVM"]["params"]["C"] == 10

    def test_multiple_models_updated_at_once(self, model_config):
        save_best_params(
            {
                "Weighted KNN": {"best_params": {"n_neighbors": 5}},
                "SVM": {"best_params": {"C": 100}},
            },
            model_config,
        )
        with open(model_config) as f:
            updated = json.load(f)
        assert updated["Weighted KNN"]["params"]["n_neighbors"] == 5
        assert updated["SVM"]["params"]["C"] == 100