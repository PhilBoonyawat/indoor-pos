"""Tests for src/app/app.py — FastAPI endpoints."""

import os
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client(trained_models_dir, test_db):
    """
    Set up a TestClient with a real predictor but a mocked scanner.
    The mocked scanner stops the background thread from running in tests.
    """
    # Ensure static dir exists — app.py mounts it on import
    static_dir = os.path.join(
        os.path.dirname(__file__), "..", "src", "app", "static"
    )
    os.makedirs(static_dir, exist_ok=True)
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("<html><body>test</body></html>")

    # Configure via env vars — app.py reads these on import
    os.environ["MODELS_DIR"] = trained_models_dir
    os.environ["DB_PATH"] = test_db
    os.environ["SCAN_INTERVAL"] = "60"

    # Force reimport so env vars take effect
    for mod_name in ("src.app.app", "src.app.predictor", "src.app.scanner_thread"):
        sys.modules.pop(mod_name, None)

    import src.app.app as app_module

    # Stop the real scanner and swap in a mock
    app_module.scanner.stop()
    mock = MagicMock()
    mock.get_current_position.return_value = {
        "room": "(S) 7.02",
        "confidence": 0.95,
        "model_used": "Random Forest",
        "position_x": 100,
        "position_y": 50,
        "aps_detected": 8,
        "mode": "live",
        "timestamp": "2024-01-01T10:00:00",
    }
    mock.get_history.return_value = [
        {
            "room": "(S) 7.01",
            "confidence": 0.92,
            "model_used": "Random Forest",
            "position_x": 0,
            "position_y": 50,
            "aps_detected": 7,
            "mode": "live",
            "timestamp": "2024-01-01T09:59:00",
        }
    ]
    mock.is_running.return_value = True
    mock.get_scan_mode.return_value = "live"
    mock.interval = 3
    app_module.scanner = mock

    from fastapi.testclient import TestClient
    yield TestClient(app_module.app)

    for mod_name in ("src.app.app", "src.app.predictor", "src.app.scanner_thread"):
        sys.modules.pop(mod_name, None)


class TestPositionEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/position").status_code == 200

    def test_returns_expected_fields(self, client):
        data = client.get("/api/position").json()
        assert data["room"] == "(S) 7.02"
        assert data["confidence"] == 0.95
        assert data["mode"] == "live"

    def test_has_coordinate_fields(self, client):
        data = client.get("/api/position").json()
        assert "position_x" in data
        assert "position_y" in data


class TestScanHistoryEndpoint:
    def test_returns_list(self, client):
        data = client.get("/api/scan-history").json()
        assert isinstance(data, list)

    def test_entries_have_timestamp(self, client):
        data = client.get("/api/scan-history").json()
        for entry in data:
            assert "timestamp" in entry


class TestStatusEndpoint:
    def test_returns_all_fields(self, client):
        data = client.get("/api/status").json()
        assert data["scanning"] is True
        assert data["scan_mode"] == "live"
        assert data["scan_interval"] == 3
        assert "active_model" in data
        assert "available_models" in data


class TestModelEndpoints:
    def test_get_models_lists_available(self, client):
        data = client.get("/api/models").json()
        assert "active" in data
        assert isinstance(data["available"], list)

    def test_switch_to_valid_model(self, client):
        resp = client.post("/api/model", json={"model_name": "Weighted KNN"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_switch_to_invalid_model_returns_error(self, client):
        resp = client.post("/api/model", json={"model_name": "Nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "available" in data


class TestRoomPositionsEndpoint:
    def test_returns_mapping(self, client):
        data = client.get("/api/room-positions").json()
        assert isinstance(data, dict)

    def test_positions_have_x_and_y(self, client):
        data = client.get("/api/room-positions").json()
        for room, pos in data.items():
            assert "x" in pos
            assert "y" in pos


class TestRootRoute:
    def test_root_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200