"""Tests for src/training/preprocess.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.training.preprocess import (
    filter_low_variance_aps,
    load_and_preprocess,
    load_and_preprocess_temporal,
    load_fingerprints_from_db,
    normalise_rssi,
)


class TestLoadFingerprints:
    """Pivoting raw SQL rows into a (scans × BSSIDs) matrix."""

    def test_returns_dataframe_and_labels(self, test_db):
        fingerprints, labels = load_fingerprints_from_db(test_db)
        assert isinstance(fingerprints, pd.DataFrame)
        assert isinstance(labels, pd.Series)

    def test_shape_matches_test_db(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        assert fingerprints.shape == (160, 10)

    def test_labels_align_with_rows(self, test_db):
        fingerprints, labels = load_fingerprints_from_db(test_db)
        assert len(labels) == len(fingerprints)
        assert list(labels.index) == list(fingerprints.index)

    def test_all_rooms_present(self, test_db):
        _, labels = load_fingerprints_from_db(test_db)
        assert set(labels.values) == {"(S) 7.01", "(S) 7.02", "(S) 7.03", "(S) 7.06"}

    def test_rssi_values_in_valid_range(self, test_db):
        """RSSI values must be between -100 (floor) and 0 (ceiling)."""
        fingerprints, _ = load_fingerprints_from_db(test_db)
        assert (fingerprints >= -100).all().all()
        assert (fingerprints <= 0).all().all()

    def test_missing_aps_filled_with_minus_100(self, test_db):
        """Any AP that wasn't detected in a scan gets -100 (not NaN)."""
        fingerprints, _ = load_fingerprints_from_db(test_db)
        assert not fingerprints.isna().any().any()

    def test_empty_db_raises(self, empty_db):
        with pytest.raises(ValueError, match="No data found"):
            load_fingerprints_from_db(empty_db)

    def test_nonexistent_db_raises(self):
        with pytest.raises(Exception):
            load_fingerprints_from_db("/no/such/path.db")


class TestFilterLowVarianceAPs:
    """Removes APs that are rarely seen."""

    def test_keeps_all_if_all_common(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        filtered = filter_low_variance_aps(fingerprints, min_detection_rate=0.05)
        assert filtered.shape[1] == fingerprints.shape[1]

    def test_removes_rare_aps(self):
        """AP detected in 1% of scans should be removed at 5% threshold."""
        data = pd.DataFrame(np.full((100, 3), -100.0),
                            columns=["ap1", "ap2", "ap3"])
        data.loc[:, "ap1"] = -60.0       # 100% detection
        data.loc[:9, "ap2"] = -70.0      # 10% detection
        data.loc[0, "ap3"] = -80.0       # 1% detection

        filtered = filter_low_variance_aps(data, min_detection_rate=0.05)
        assert set(filtered.columns) == {"ap1", "ap2"}

    def test_threshold_of_zero_keeps_everything(self):
        data = pd.DataFrame([[-100.0, -100.0]], columns=["a", "b"])
        data.loc[0, "a"] = -50.0
        filtered = filter_low_variance_aps(data, min_detection_rate=0.0)
        assert set(filtered.columns) == {"a", "b"}

    def test_threshold_above_one_removes_all(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        filtered = filter_low_variance_aps(fingerprints, min_detection_rate=1.01)
        assert filtered.shape[1] == 0


class TestNormaliseRSSI:
    """MinMax scaling RSSI to [0, 1]."""

    def test_values_in_zero_to_one(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        normalised, _ = normalise_rssi(fingerprints)
        # Allow tiny float tolerance
        assert normalised.min().min() >= -1e-9
        assert normalised.max().max() <= 1.0 + 1e-9

    def test_returns_fitted_scaler(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        _, scaler = normalise_rssi(fingerprints)
        assert isinstance(scaler, MinMaxScaler)
        # Scaler should be fitted — data_min_ exists
        assert hasattr(scaler, "data_min_")

    def test_preserves_shape_and_index(self, test_db):
        fingerprints, _ = load_fingerprints_from_db(test_db)
        normalised, _ = normalise_rssi(fingerprints)
        assert normalised.shape == fingerprints.shape
        assert list(normalised.columns) == list(fingerprints.columns)
        assert list(normalised.index) == list(fingerprints.index)

    def test_scaler_is_reusable_on_new_data(self, test_db):
        """The returned scaler must be able to transform fresh data."""
        fingerprints, _ = load_fingerprints_from_db(test_db)
        _, scaler = normalise_rssi(fingerprints)
        new_data = fingerprints.head(3)
        transformed = scaler.transform(new_data)
        assert transformed.shape == new_data.shape


class TestLoadAndPreprocess:
    """End-to-end: DB → ready-to-train arrays."""

    def test_returns_seven_items(self, test_db):
        result = load_and_preprocess(test_db)
        assert len(result) == 7

    def test_train_test_sum_equals_total(self, test_db):
        X_train, X_test, *_ = load_and_preprocess(test_db)
        assert X_train.shape[0] + X_test.shape[0] == 160

    def test_feature_dimension_consistent(self, test_db):
        X_train, X_test, *_ = load_and_preprocess(test_db)
        assert X_train.shape[1] == X_test.shape[1]

    def test_labels_count_matches_rows(self, test_db):
        X_train, X_test, y_train, y_test, *_ = load_and_preprocess(test_db)
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_feature_names_match_column_count(self, test_db):
        X_train, _, _, _, feature_names, _, _ = load_and_preprocess(test_db)
        assert len(feature_names) == X_train.shape[1]

    def test_label_encoder_covers_all_rooms(self, test_db):
        _, _, _, _, _, le, _ = load_and_preprocess(test_db)
        assert set(le.classes_) == {"(S) 7.01", "(S) 7.02", "(S) 7.03", "(S) 7.06"}

    def test_stratified_split_includes_every_room(self, test_db):
        _, _, y_train, y_test, _, le, _ = load_and_preprocess(test_db)
        train_rooms = set(le.inverse_transform(y_train))
        test_rooms = set(le.inverse_transform(y_test))
        assert train_rooms == test_rooms == set(le.classes_)

    def test_custom_test_size_respected(self, test_db):
        X_train, X_test, *_ = load_and_preprocess(test_db, test_size=0.3)
        total = X_train.shape[0] + X_test.shape[0]
        assert abs(X_test.shape[0] / total - 0.3) < 0.1

    def test_output_is_normalised(self, test_db):
        X_train, X_test, *_ = load_and_preprocess(test_db)
        assert X_train.min() >= -1e-9
        assert X_train.max() <= 1.0 + 1e-9

    def test_reproducibility_with_same_seed(self, test_db):
        """Same random_state must produce the same split."""
        out1 = load_and_preprocess(test_db, random_state=7)
        out2 = load_and_preprocess(test_db, random_state=7)
        np.testing.assert_array_equal(out1[2], out2[2])  # y_train


class TestLoadAndPreprocessTemporal:
    """Time-ordered split (train on earlier scans, test on later)."""

    def test_returns_seven_items(self, test_db):
        assert len(load_and_preprocess_temporal(test_db)) == 7

    def test_no_data_leakage_in_sample_count(self, test_db):
        X_train, X_test, *_ = load_and_preprocess_temporal(test_db)
        assert X_train.shape[0] + X_test.shape[0] == 160

    def test_default_split_ratio_close_to_20_percent(self, test_db):
        X_train, X_test, *_ = load_and_preprocess_temporal(test_db, test_ratio=0.2)
        total = X_train.shape[0] + X_test.shape[0]
        # Per-room integer rounding makes this approximate
        assert 0.1 < X_test.shape[0] / total < 0.4

    def test_custom_test_ratio_respected(self, test_db):
        X_train, X_test, *_ = load_and_preprocess_temporal(test_db, test_ratio=0.5)
        total = X_train.shape[0] + X_test.shape[0]
        assert 0.4 < X_test.shape[0] / total < 0.6

    def test_every_room_appears_in_train_and_test(self, test_db):
        _, _, y_train, y_test, _, le, _ = load_and_preprocess_temporal(test_db)
        assert set(le.inverse_transform(y_train)) == set(le.classes_)
        assert set(le.inverse_transform(y_test)) == set(le.classes_)