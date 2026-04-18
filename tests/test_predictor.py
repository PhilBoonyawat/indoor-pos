"""Tests for src/app/predictor.py"""

import os

import joblib
import numpy as np
import pytest

from src.app.predictor import Predictor


# ── Init / loading ──────────────────────────────────────────────

class TestPredictorInit:
    """Loading models and metadata from a directory."""

    def test_loads_both_models(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert "Weighted KNN" in p.models
        assert "Random Forest" in p.models

    def test_defaults_to_random_forest(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert p.active_model == "Random Forest"

    def test_loads_label_encoder_with_all_classes(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert p.label_encoder is not None
        assert len(p.label_encoder.classes_) == 4

    def test_loads_scaler(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert p.scaler is not None

    def test_loads_feature_names(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert len(p.feature_names) == 10

    def test_loads_room_positions(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert len(p.room_positions) == 4

    def test_no_models_dir_means_demo_only(self):
        p = Predictor(models_dir=None)
        assert p.models == {}
        assert p.active_model is None

    def test_nonexistent_dir_means_demo_only(self):
        p = Predictor(models_dir="/no/such/path")
        assert p.models == {}

    def test_prefers_random_forest_when_present(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert p.active_model == "Random Forest"

    def test_falls_back_to_first_model_when_rf_missing(self, trained_models_dir):
        """If random_forest.pkl is deleted, Predictor should pick another model."""
        os.remove(os.path.join(trained_models_dir, "random_forest.pkl"))
        p = Predictor(models_dir=trained_models_dir)
        assert p.active_model == "Weighted KNN"

    def test_corrupted_pkl_file_is_skipped(self, trained_models_dir):
        """Corrupt .pkl must not crash init — it's logged and skipped."""
        rf_path = os.path.join(trained_models_dir, "random_forest.pkl")
        with open(rf_path, "w") as f:
            f.write("not a real pickle")

        p = Predictor(models_dir=trained_models_dir)
        assert "Random Forest" not in p.models
        assert p.active_model == "Weighted KNN"

    def test_missing_metadata_files_handled_gracefully(self, trained_models_dir):
        """Predictor loads what it can and skips missing files without raising."""
        os.remove(os.path.join(trained_models_dir, "label_encoder.pkl"))
        p = Predictor(models_dir=trained_models_dir)
        assert p.label_encoder is None


# ── Model switching ─────────────────────────────────────────────

class TestPredictorModelSwitching:
    def test_get_available_models_lists_all(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert set(p.get_available_models()) == {"Weighted KNN", "Random Forest"}

    def test_switch_to_valid_model(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        assert p.set_active_model("Weighted KNN") is True
        assert p.active_model == "Weighted KNN"

    def test_switch_to_unknown_model_returns_false(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        original = p.active_model
        assert p.set_active_model("Nonexistent") is False
        assert p.active_model == original  # unchanged


# ── Prediction ─────────────────────────────────────────────────

class TestLivePrediction:
    def test_returns_all_expected_keys(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        expected = {"room", "confidence", "model_used", "position_x",
                    "position_y", "aps_detected", "mode"}
        assert expected.issubset(result.keys())

    def test_predicted_room_is_known_class(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        assert result["room"] in {"(S) 7.01", "(S) 7.02", "(S) 7.03", "(S) 7.06"}

    def test_confidence_is_valid_probability(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_mode_is_live_when_model_loaded(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        assert result["mode"] == "live"

    def test_model_used_matches_active_model(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        p.set_active_model("Weighted KNN")
        result = p.predict(sample_fingerprint)
        assert result["model_used"] == "Weighted KNN"

    def test_aps_detected_matches_input_size(self, trained_models_dir, sample_fingerprint):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        assert result["aps_detected"] == len(sample_fingerprint)

    def test_handles_sparse_fingerprint(self, trained_models_dir):
        """Only two known APs in the fingerprint — model must still predict."""
        p = Predictor(models_dir=trained_models_dir)
        sparse = {"aa:bb:cc:dd:ee:00": -55, "aa:bb:cc:dd:ee:01": -70}
        result = p.predict(sparse)
        assert "room" in result

    def test_handles_empty_fingerprint(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict({})
        assert result["aps_detected"] == 0

    def test_ignores_unknown_bssids(self, trained_models_dir):
        """BSSIDs not in feature_names are silently dropped."""
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict({"zz:zz:zz:zz:zz:01": -50})
        assert "room" in result

    def test_position_returned_for_predicted_room(
        self, trained_models_dir, sample_fingerprint
    ):
        p = Predictor(models_dir=trained_models_dir)
        result = p.predict(sample_fingerprint)
        assert isinstance(result["position_x"], int)
        assert isinstance(result["position_y"], int)

    def test_knn_uses_fallback_confidence_path(
        self, trained_models_dir, sample_fingerprint
    ):
        """KNN's predict_proba works — confidence should still be valid."""
        p = Predictor(models_dir=trained_models_dir)
        p.set_active_model("Weighted KNN")
        result = p.predict(sample_fingerprint)
        assert 0.0 <= result["confidence"] <= 1.0


class TestDemoPrediction:
    """When no models are loaded, predict() falls back to random demo output."""

    def test_demo_mode_flag(self):
        p = Predictor(models_dir=None)
        assert p.predict({})["mode"] == "demo"

    def test_demo_model_used_label(self):
        p = Predictor(models_dir=None)
        assert p.predict({})["model_used"] == "Demo Mode"

    def test_demo_confidence_in_expected_range(self):
        """Demo mode uses uniform(0.85, 0.99)."""
        p = Predictor(models_dir=None)
        for _ in range(10):
            result = p.predict({})
            assert 0.85 <= result["confidence"] <= 0.99

    def test_demo_returns_one_of_predefined_rooms(self):
        p = Predictor(models_dir=None)
        valid_rooms = {
            "(S) 7.01", "(S) 7.02", "(S) 7.03",
            "(S) 7.04", "(S) 7.05", "(S) 7.06",
        }
        for _ in range(20):
            assert p.predict({})["room"] in valid_rooms

    def test_demo_aps_detected_is_zero(self):
        p = Predictor(models_dir=None)
        assert p.predict({})["aps_detected"] == 0


class TestFingerprintToVector:
    def test_vector_length_matches_feature_names(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        vector = p._fingerprint_to_vector({})
        assert len(vector) == len(p.feature_names)

    def test_missing_aps_default_to_minus_100(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        vector = p._fingerprint_to_vector({})
        assert np.all(vector == -100)

    def test_known_ap_value_preserved(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        bssid = p.feature_names[0]
        vector = p._fingerprint_to_vector({bssid: -42})
        assert vector[0] == -42
        assert np.all(vector[1:] == -100)

    def test_returns_numpy_array_of_floats(self, trained_models_dir):
        p = Predictor(models_dir=trained_models_dir)
        vector = p._fingerprint_to_vector({})
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float64