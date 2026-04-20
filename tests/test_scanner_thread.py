"""Tests for src/app/scanner_thread.py"""

import time
from unittest.mock import MagicMock, patch


class TestScannerInit:
    """Tests for the initialization of the ScannerThread class, which manages background Wi-Fi scanning and prediction."""
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

    def test_initial_scan_mode_is_demo(self, scanner):
        """get_scan_mode returns the default _last_mode set in __init__."""
        assert scanner.get_scan_mode() == "demo"


class TestScannerStartStop:
    """Tests for starting and stopping the ScannerThread."""
    def test_start_sets_running_flag(self, scanner):
        scanner.start()
        assert scanner.is_running() is True

    def test_stop_clears_running_flag(self, scanner):
        scanner.start()
        scanner.stop()
        assert scanner.is_running() is False

    def test_stop_without_start_does_not_raise(self, scanner):
        scanner.stop()


class TestScannerBackgroundLoop:
    """Tests for the real-thread background scanning loop."""
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


class TestScannerAdvancedLoop:
    """Tests for scan-loop fallback paths — live failures, weak scans, history trimming."""

    def test_reuses_last_live_fingerprint_when_scan_fails(self, live_scanner):
        live_scanner._last_fingerprint = {"aa:bb:cc:dd:ee:00": -55}
        live_scanner._last_mode = "live"

        with patch.object(live_scanner, "_try_live_scan", return_value=None), \
             patch.object(live_scanner.predictor, "predict", return_value={
                 "room": "(S)7.01", "confidence": 0.9, "model_used": "RF",
                 "position_x": 0, "position_y": 0, "aps_detected": 1,
             }), \
             patch("src.app.scanner_thread.time.sleep",
                   side_effect=lambda _: setattr(live_scanner, "_running", False)):
            live_scanner._running = True
            live_scanner._scan_loop()

        assert live_scanner.get_current_position()["mode"] == "live"

    def test_falls_back_to_demo_when_not_enough_known_aps(self, live_scanner):
        weak_scan = {"unknown_ap": -70}

        with patch.object(live_scanner, "_try_live_scan", return_value=weak_scan), \
             patch.object(live_scanner, "_load_demo_scans"), \
             patch.object(live_scanner, "_get_demo_fingerprint", return_value={"demo": -60}), \
             patch.object(live_scanner.predictor, "predict", return_value={
                 "room": "(S)7.02", "confidence": 0.8, "model_used": "RF",
                 "position_x": 0, "position_y": 0, "aps_detected": 1,
             }), \
             patch("src.app.scanner_thread.time.sleep",
                   side_effect=lambda _: setattr(live_scanner, "_running", False)):
            live_scanner._running = True
            live_scanner._scan_loop()

        assert live_scanner.get_current_position()["mode"] == "demo"

    def test_falls_through_to_demo_when_scan_fails_and_no_cache(self, live_scanner):
        """When live scan returns None AND there's no cached _last_fingerprint,
        fingerprint stays None, so we enter the demo fallback block."""
        live_scanner._last_fingerprint = None
        live_scanner._last_mode = "demo"

        with patch.object(live_scanner, "_try_live_scan", return_value=None), \
             patch.object(live_scanner, "_load_demo_scans"), \
             patch.object(live_scanner, "_get_demo_fingerprint",
                          return_value={"aa:bb:cc:dd:ee:00": -50}), \
             patch.object(live_scanner.predictor, "predict", return_value={
                 "room": "(S)7.03", "confidence": 0.7, "model_used": "RF",
                 "position_x": 0, "position_y": 0, "aps_detected": 1,
             }), \
             patch("src.app.scanner_thread.time.sleep",
                   side_effect=lambda _: setattr(live_scanner, "_running", False)):
            live_scanner._running = True
            live_scanner._scan_loop()

        assert live_scanner.get_current_position()["mode"] == "demo"
        assert live_scanner._demo_loaded is True

    def test_history_trimmed_to_100(self, live_scanner):
        fake_prediction = {
            "room": "(S)7.01", "confidence": 0.9, "model_used": "RF",
            "position_x": 0, "position_y": 0, "aps_detected": 5,
        }

        counter = {"n": 0}
        def fake_sleep(_):
            counter["n"] += 1
            if counter["n"] >= 105:
                live_scanner._running = False

        with patch.object(live_scanner, "_try_live_scan",
                          return_value={"aa:bb:cc:dd:ee:00": -50}), \
             patch.object(live_scanner.predictor, "predict",
                          return_value=fake_prediction), \
             patch("src.app.scanner_thread.time.sleep", side_effect=fake_sleep):
            live_scanner._running = True
            live_scanner._scan_loop()

        assert len(live_scanner.get_history(200)) == 100


class TestLiveScanRouting:
    """Tests for _try_live_scan — platform routing and exception handling."""

    def test_non_darwin_platform_returns_none(self, live_scanner):
        with patch("src.app.scanner_thread.platform.system", return_value="Linux"):
            assert live_scanner._try_live_scan() is None

    def test_exception_in_macos_scan_returns_none(self, live_scanner):
        with patch("src.app.scanner_thread.platform.system", return_value="Darwin"), \
             patch.object(live_scanner, "_scan_macos",
                          side_effect=RuntimeError("simulated failure")):
            assert live_scanner._try_live_scan() is None

    def test_macos_scan_result_passed_through(self, live_scanner):
        fake_result = {"aa:bb:cc:dd:ee:00": -50}
        with patch("src.app.scanner_thread.platform.system", return_value="Darwin"), \
             patch.object(live_scanner, "_scan_macos", return_value=fake_result):
            assert live_scanner._try_live_scan() == fake_result


class TestMacOSScan:
    """Tests for _scan_macos CoreWLAN interaction."""

    def _install_corewlan_mock(self, scans, error=None):
        """
        Build a mock CWWiFiClient whose shared instance returns an
        interface that yields the given (scans, error) tuple.
        
        Args:
            scans: List of scan results to return from scanForNetworksWithName_error_
            error: Optional error string to return from scanForNetworksWithName_error_

        Returns:
            A context manager that patches CWWiFiClient with the configured mock.
        """
        fake_iface = MagicMock()
        fake_iface.scanForNetworksWithName_error_.return_value = (scans, error)
        fake_client = MagicMock()
        fake_client.interface.return_value = fake_iface

        mock_cwwificlient_cls = MagicMock()
        mock_cwwificlient_cls.sharedWiFiClient.return_value = fake_client

        return patch(
            "src.app.scanner_thread.CWWiFiClient",
            mock_cwwificlient_cls,
            create=True,
        )

    def test_returns_cached_scan_on_resource_busy(self, live_scanner):
        live_scanner._last_live_scan = {"aa:bb:cc:dd:ee:00": -50}
        with self._install_corewlan_mock([], error="Resource busy"):
            result = live_scanner._scan_macos()
        assert result == {"aa:bb:cc:dd:ee:00": -50}

    def test_non_busy_error_returns_none(self, live_scanner):
        with self._install_corewlan_mock([], error="Some other error"):
            assert live_scanner._scan_macos() is None

    def test_empty_scan_results_returns_none(self, live_scanner):
        with self._install_corewlan_mock([], error=None):
            assert live_scanner._scan_macos() is None


class TestDemoMode:
    """Tests for demo-mode behaviour: loading scans and rotation."""

    def test_get_demo_fingerprint_empty_when_no_scans(self, scanner):
        scanner._demo_scans = []
        assert scanner._get_demo_fingerprint() == {}

    def test_get_demo_fingerprint_rotates_through_scans(self, scanner):
        scanner._demo_scans = [
            {"ap1": -50},
            {"ap2": -60},
            {"ap3": -70},
        ]
        scanner._demo_index = 0

        assert scanner._get_demo_fingerprint() == {"ap1": -50}
        assert scanner._get_demo_fingerprint() == {"ap2": -60}
        assert scanner._get_demo_fingerprint() == {"ap3": -70}
        assert scanner._get_demo_fingerprint() == {"ap1": -50}

    def test_load_demo_scans_with_missing_db_path(self, scanner):
        scanner.db_path = None
        scanner._demo_scans = []
        scanner._load_demo_scans()
        assert scanner._demo_scans == []

    def test_load_demo_scans_with_nonexistent_db(self, scanner, tmp_path):
        scanner.db_path = str(tmp_path / "does_not_exist.db")
        scanner._demo_scans = []
        scanner._load_demo_scans()
        assert scanner._demo_scans == []

    def test_load_demo_scans_populates_from_real_db(self, scanner, test_db):
        scanner.db_path = test_db
        scanner._demo_scans = []
        scanner._load_demo_scans()
        assert len(scanner._demo_scans) >= 1
        for fp in scanner._demo_scans:
            assert all(isinstance(k, str) for k in fp.keys())
            assert all(isinstance(v, int) for v in fp.values())

    def test_load_demo_scans_handles_db_error(self, scanner, tmp_path):
        bad_db = tmp_path / "not_a_real.db"
        bad_db.write_text("this is not sqlite content")
        scanner.db_path = str(bad_db)
        scanner._demo_scans = []
        scanner._load_demo_scans()
        assert scanner._demo_scans == []