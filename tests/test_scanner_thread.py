""" Tests for src/app/scanner_thread.py"""
import time
from unittest.mock import patch, MagicMock

import pytest
#TODO

from src.app.predictor import Predictor
from src.app.scanner_thread import ScannerThread

@pytest.fixture
def demo_predictor():
    return Predictor(models_dir=None)


@pytest.fixture
def live_predictor(trained_models_dir):
    return Predictor(models_dir=trained_models_dir)


@pytest.fixture
def scanner(demo_predictor, test_db):
    s = ScannerThread(predictor=demo_predictor, interval=1, db_path=test_db)
    yield s
    if s.is_running():
        s.stop()


@pytest.fixture
def live_scanner(live_predictor, test_db):
    s = ScannerThread(predictor=live_predictor, interval=1, db_path=test_db)
    yield s
    if s.is_running():
        s.stop()


class TestScannerInit:
    def test_default_position_is_unknown(self, scanner):
        pos = scanner.get_current_position()
        assert pos["room"] == "Unknown"
        assert pos["mode"] == "initialising"

    def test_not_running_by_default(self, scanner):
        assert scanner.is_running() is False

    def test_interval_stored(self, scanner):
        assert scanner.interval == 1

    def test_history_starts_empty(self, scanner):
        assert scanner.get_history() == []


class TestScannerStartStop:
    def test_start_sets_running_flag(self, scanner):
        scanner.start()
        assert scanner.is_running() is True

    def test_stop_clears_running_flag(self, scanner):
        scanner.start()
        scanner.stop()
        assert scanner.is_running() is False

    def test_stop_without_start_does_not_raise(self, scanner):
        scanner.stop()


# ── Background scanning ────────────────────────────────────────

class TestScannerBackgroundLoop:
    def test_produces_predictions_over_time(self, scanner):
        scanner.start()
        time.sleep(2.5)
        scanner.stop()
        pos = scanner.get_current_position()
        assert pos["mode"] in ("demo", "live")
        assert pos["room"] != "Unknown"

    def test_history_accumulates(self, scanner):
        scanner.start()
        time.sleep(3.5)
        scanner.stop()
        assert len(scanner.get_history()) >= 2

    def test_history_limit_applied(self, scanner):
        scanner.start()
        time.sleep(4)
        scanner.stop()
        assert len(scanner.get_history(limit=2)) <= 2

    def test_history_entries_have_timestamps(self, scanner):
        scanner.start()
        time.sleep(2.5)
        scanner.stop()
        for entry in scanner.get_history():
            assert entry["timestamp"] is not None


# ── NEW: deeper scan loop logic ────────────────────────────────

class TestScannerAdvancedLoop:
    def test_reuses_last_live_fingerprint_when_scan_fails(self, live_scanner):
        live_scanner._last_fingerprint = {"aa:bb:cc:dd:ee:00": -55}
        live_scanner._last_mode = "live"

        with patch.object(live_scanner, "_try_live_scan", return_value=None), \
             patch.object(live_scanner.predictor, "predict", return_value={
                 "room": "(S)7.01",
                 "confidence": 0.9,
                 "model_used": "RF",
                 "position_x": 0,
                 "position_y": 0,
                 "aps_detected": 1,
             }), \
             patch("src.app.scanner_thread.time.sleep", side_effect=lambda _: setattr(live_scanner, "_running", False)):

            live_scanner._running = True
            live_scanner._scan_loop()

        pos = live_scanner.get_current_position()
        assert pos["mode"] == "live"

    def test_falls_back_to_demo_when_not_enough_known_aps(self, live_scanner):
        weak_scan = {"unknown_ap": -70}

        with patch.object(live_scanner, "_try_live_scan", return_value=weak_scan), \
             patch.object(live_scanner, "_load_demo_scans"), \
             patch.object(live_scanner, "_get_demo_fingerprint", return_value={"demo": -60}), \
             patch.object(live_scanner.predictor, "predict", return_value={
                 "room": "(S)7.02",
                 "confidence": 0.8,
                 "model_used": "RF",
                 "position_x": 0,
                 "position_y": 0,
                 "aps_detected": 1,
             }), \
             patch("src.app.scanner_thread.time.sleep", side_effect=lambda _: setattr(live_scanner, "_running", False)):

            live_scanner._running = True
            live_scanner._scan_loop()

        pos = live_scanner.get_current_position()
        assert pos["mode"] == "demo"

    def test_history_trimmed_to_100(self, live_scanner):
        fake_prediction = {
            "room": "(S)7.01",
            "confidence": 0.9,
            "model_used": "RF",
            "position_x": 0,
            "position_y": 0,
            "aps_detected": 5,
        }

        counter = {"n": 0}

        def fake_sleep(_):
            counter["n"] += 1
            if counter["n"] >= 105:
                live_scanner._running = False

        with patch.object(
            live_scanner,
            "_try_live_scan",
            return_value={"aa:bb:cc:dd:ee:00": -50},
        ), patch.object(
            live_scanner.predictor,
            "predict",
            return_value=fake_prediction,
        ), patch(
            "src.app.scanner_thread.time.sleep",
            side_effect=fake_sleep,
        ):
            live_scanner._running = True
            live_scanner._scan_loop()

        assert len(live_scanner.get_history(200)) == 100


# ── NEW: macOS scan logic (mocked) ─────────────────────────────

class TestMacOSScan:
    def test_returns_cached_scan_on_resource_busy(self, live_scanner):
        live_scanner._last_live_scan = {"aa:bb:cc:dd:ee:00": -50}

        fake_iface = MagicMock()
        fake_iface.scanForNetworksWithName_error_.return_value = ([], "Resource busy")

        fake_client = MagicMock()
        fake_client.interface.return_value = fake_iface

        with patch("src.app.scanner_thread.CWWiFiClient.sharedWiFiClient", return_value=fake_client):
            result = live_scanner._scan_macos()

        assert result == {"aa:bb:cc:dd:ee:00": -50}

    def test_builds_fingerprint_and_caches(self, live_scanner):
        scan1 = MagicMock()
        scan1.bssid.return_value = "AA:BB:CC:DD:EE:01"
        scan1.rssiValue.return_value = -55

        scan2 = MagicMock()
        scan2.bssid.return_value = "AA:BB:CC:DD:EE:02"
        scan2.rssiValue.return_value = -60

        fake_iface = MagicMock()
        fake_iface.scanForNetworksWithName_error_.return_value = ([scan1, scan2], None)

        fake_client = MagicMock()
        fake_client.interface.return_value = fake_iface

        with patch("src.app.scanner_thread.CWWiFiClient.sharedWiFiClient", return_value=fake_client):
            result = live_scanner._scan_macos()

        assert result == {
            "aa:bb:cc:dd:ee:01": -55,
            "aa:bb:cc:dd:ee:02": -60,
        }
        assert live_scanner._last_live_scan == result