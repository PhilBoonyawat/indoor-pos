"""
Shared test fixtures for the Indoor Position Tracker test suite.

Organisation:
- Constants at the top: TEST_ROOMS, TEST_BSSIDS, SCANS_PER_ROOM
- Factory fixtures: make_scan_data (callable — tests pass args)
- Data fixtures: test_db, empty_db, sample_fingerprint
- File/dir fixtures: trained_models_dir, model_config
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# ── Shared constants ────────────────────────────────────────────────

TEST_ROOMS = ["(S) 7.01", "(S) 7.02", "(S) 7.03", "(S) 7.06"]
TEST_BSSIDS = [f"aa:bb:cc:dd:ee:{i:02x}" for i in range(10)]
SCANS_PER_ROOM = 40


# ── Factory fixtures — return callables for flexible data creation ──

@pytest.fixture
def make_scan_data():
    """
    Factory fixture: returns a function that builds scan_data lists.
    Tests can call it with different args instead of hardcoding data.
    """
    def _make(n_aps=5, location="(S) 7.01", orientation="N",
              timestamp="2024-01-01T10:00:00"):
        return [
            {
                "ssid": f"Network-{i}",
                "bssid": f"aa:bb:cc:dd:ee:{i:02x}",
                "rssi": -60 + i,
                "noise": -95,
                "channel": 6,
                "timestamp": timestamp,
                "location": location,
                "latitude": 51.511,
                "longitude": -0.116,
                "orientation": orientation,
            }
            for i in range(n_aps)
        ]
    return _make


# ── Database fixtures ──────────────────────────────────────────────

def _create_schema(conn):
    """Create all WiFi scan tables. Shared between test_db and empty_db."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE scan_metadata (
            scan_id TEXT PRIMARY KEY,
            timestamp TEXT,
            location TEXT,
            latitude REAL,
            longitude REAL,
            orientation TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE ssid (
            bssid TEXT PRIMARY KEY,
            ssid TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE wifi_scan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT REFERENCES scan_metadata(scan_id),
            bssid TEXT REFERENCES ssid(bssid),
            rssi INTEGER,
            noise INTEGER,
            channel INTEGER
        )
    """)


@pytest.fixture
def test_db(tmp_path):
    """
    SQLite database populated with 60 scans (20 per room, 3 rooms).
    Each room has a distinct RSSI signature so ML models can learn it.
    """
    db_path = str(tmp_path / "wifi_scans.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _create_schema(conn)

    cur = conn.cursor()
    for bssid in TEST_BSSIDS:
        cur.execute(
            "INSERT INTO ssid (bssid, ssid) VALUES (?, ?)",
            (bssid, f"TestAP-{bssid[-2:]}")
        )

    rng = np.random.RandomState(42)
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    for room_idx, room in enumerate(TEST_ROOMS):
        for scan_num in range(SCANS_PER_ROOM):
            scan_id = str(uuid.uuid4())
            ts = (base_time + timedelta(
                minutes=scan_num * 5 + room_idx * 100
            )).isoformat()

            cur.execute("""
                INSERT INTO scan_metadata
                (scan_id, timestamp, location, latitude, longitude, orientation)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (scan_id, ts, room, 51.511, -0.116, "N"))

            for ap_idx, bssid in enumerate(TEST_BSSIDS):
                base_rssi = -80 + (room_idx * 5) + (ap_idx * (room_idx - 1))
                rssi = int(np.clip(base_rssi + rng.randint(-5, 6), -100, -20))
                cur.execute("""
                    INSERT INTO wifi_scan
                    (scan_id, bssid, rssi, noise, channel)
                    VALUES (?, ?, ?, ?, ?)
                """, (scan_id, bssid, rssi, -95, 6))

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_db(tmp_path):
    """Empty database with schema only, no rows."""
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    _create_schema(conn)
    conn.commit()
    conn.close()
    return db_path


# ── Model fixtures ─────────────────────────────────────────────────

@pytest.fixture
def trained_models_dir(tmp_path, test_db):
    """Directory containing two trained models plus all metadata files."""
    from src.training.preprocess import load_and_preprocess

    models_dir = str(tmp_path / "models")
    os.makedirs(models_dir)

    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = \
        load_and_preprocess(test_db)

    knn = KNeighborsClassifier(n_neighbors=3, weights="distance")
    knn.fit(X_train, y_train)
    joblib.dump(knn, os.path.join(models_dir, "weighted_knn.pkl"))

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))

    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    room_positions = {
        room: {"x": i * 100, "y": 50}
        for i, room in enumerate(TEST_ROOMS)
    }
    with open(os.path.join(models_dir, "room_positions.json"), "w") as f:
        json.dump(room_positions, f)

    return models_dir


@pytest.fixture
def model_config(tmp_path):
    """A model_configs.json file with fast-training hyperparameters."""
    config = {
        "Weighted KNN": {
            "model": "KNeighborsClassifier",
            "params": {
                "n_neighbors": 3,
                "weights": "distance",
                "metric": "manhattan",
                "n_jobs": -1,
            },
        },
        "Random Forest": {
            "model": "RandomForestClassifier",
            "params": {
                "n_estimators": 10,
                "max_features": "log2",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "SVM": {
            "model": "SVC",
            "params": {
                "kernel": "rbf",
                "C": 10,
                "gamma": "scale",
                "random_state": 42,
                "probability": True,
            },
        },
        "MLP": {
            "model": "MLPClassifier",
            "params": {
                "hidden_layer_sizes": [64, 32],
                "alpha": 0.0001,
                "max_iter": 100,
                "early_stopping": True,
                "random_state": 42,
            },
        },
    }
    config_path = str(tmp_path / "model_configs.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ── Prediction fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_fingerprint():
    """A realistic WiFi fingerprint — one RSSI reading per known AP."""
    return {bssid: -60 + i * 3 for i, bssid in enumerate(TEST_BSSIDS)}