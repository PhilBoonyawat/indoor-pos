"""Tests for src/data_collection/data_collector_service.py"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Mock macOS-only modules BEFORE importing the module under test ──
# Using direct assignment (not setdefault) so even on macOS — where the real
# CoreWLAN is importable — we still use our mock. Without this, the tests
# that patch CWWiFiClient wouldn't work because the real module would be in
# sys.modules already.

_mock_corewlan = MagicMock()
_mock_location_service = MagicMock()
_mock_location_service.retrieve_current_location = MagicMock(
    return_value=[51.511, -0.116]
)

sys.modules["CoreWLAN"] = _mock_corewlan
sys.modules["data_collection"] = MagicMock()
sys.modules["data_collection.location_service"] = _mock_location_service
sys.modules["data_collection.db_service"] = MagicMock()

from src.data_collection.data_collector_service import (  # noqa: E402
    parse_args,
    scan_for_networks,
    write_data_to_db,
)


class TestParseArgs:
    def test_parses_both_args_short(self):
        with patch("sys.argv", ["prog", "-l", "(S) 7.01", "-f", "N"]):
            args = parse_args()
        assert args.location == "(S) 7.01"
        assert args.orientation == "N"

    def test_parses_both_args_long(self):
        with patch("sys.argv", ["prog", "--location", "Room A", "--orientation", "S"]):
            args = parse_args()
        assert args.location == "Room A"
        assert args.orientation == "S"

    def test_no_args_returns_none_values(self):
        with patch("sys.argv", ["prog"]):
            args = parse_args()
        assert args.location is None
        assert args.orientation is None


class TestWriteDataToDBValidation:
    """Input validation happens before any scanning — test that first."""

    def test_none_location_raises(self):
        with pytest.raises(ValueError, match="Location is required"):
            write_data_to_db(None, "N")

    def test_empty_location_raises(self):
        with pytest.raises(ValueError, match="Location is required"):
            write_data_to_db("", "N")

    def test_none_orientation_raises(self):
        with pytest.raises(ValueError, match="Orientation is required"):
            write_data_to_db("(S) 7.01", None)

    def test_empty_orientation_raises(self):
        with pytest.raises(ValueError, match="Orientation is required"):
            write_data_to_db("(S) 7.01", "")


class TestWriteDataToDBHappyPath:
    """Covers the successful scan → init_db → store path (lines 59-65)."""

    def test_calls_scan_init_and_store_in_order(self):
        fake_scan_data = [{"bssid": "aa:bb", "ssid": "net", "rssi": -60}]
        fake_conn = MagicMock()

        with patch("src.data_collection.data_collector_service.scan_for_networks",
                   return_value=fake_scan_data) as mock_scan, \
             patch("src.data_collection.data_collector_service.init_db",
                   return_value=fake_conn) as mock_init, \
             patch("src.data_collection.data_collector_service.store_raw_scan") as mock_store:
            write_data_to_db("(S) 7.01", "N")

        mock_scan.assert_called_once_with("(S) 7.01", "N")
        mock_init.assert_called_once()
        mock_store.assert_called_once_with(fake_conn, fake_scan_data)
        fake_conn.close.assert_called_once()

    def test_closes_connection_even_if_store_fails(self):
        """Connection must be closed (try/finally) even on store_raw_scan error."""
        fake_conn = MagicMock()

        with patch("src.data_collection.data_collector_service.scan_for_networks",
                   return_value=[{"x": 1}]), \
             patch("src.data_collection.data_collector_service.init_db",
                   return_value=fake_conn), \
             patch("src.data_collection.data_collector_service.store_raw_scan",
                   side_effect=RuntimeError("DB write failed")):
            with pytest.raises(RuntimeError, match="DB write failed"):
                write_data_to_db("(S) 7.01", "N")

        fake_conn.close.assert_called_once()


class TestScanForNetworks:
    """scan_for_networks is heavily mocked — we only verify shape and logic."""

    def _make_mock_scan(self, bssid, ssid="Net", rssi=-60, channel=6, noise=-95):
        """Create a mock CoreWLAN scan result object."""
        scan = MagicMock()
        scan.bssid.return_value = bssid
        scan.ssid.return_value = ssid
        scan.rssiValue.return_value = rssi
        scan.noiseMeasurement.return_value = noise
        scan.wlanChannel.return_value.channelNumber.return_value = channel
        return scan

    def test_builds_scan_data_entries(self):
        scans = [
            self._make_mock_scan("aa:bb:cc:dd:ee:01", rssi=-55),
            self._make_mock_scan("aa:bb:cc:dd:ee:02", rssi=-70),
        ]

        with patch("src.data_collection.data_collector_service.CWWiFiClient") as mock_client:
            iface = mock_client.sharedWiFiClient.return_value.interface.return_value
            iface.scanForNetworksWithName_error_.return_value = (scans, None)

            result = scan_for_networks("(S) 7.01", "N")

        assert len(result) == 2
        assert result[0]["bssid"] == "aa:bb:cc:dd:ee:01"
        assert result[0]["rssi"] == -55
        assert result[0]["location"] == "(S) 7.01"
        assert result[0]["orientation"] == "N"

    def test_deduplicates_bssids_in_single_scan(self):
        """If CoreWLAN returns the same BSSID twice, keep only one."""
        scans = [
            self._make_mock_scan("aa:bb:cc:dd:ee:01"),
            self._make_mock_scan("aa:bb:cc:dd:ee:01"),  # duplicate
            self._make_mock_scan("aa:bb:cc:dd:ee:02"),
        ]
        with patch("src.data_collection.data_collector_service.CWWiFiClient") as mock_client:
            iface = mock_client.sharedWiFiClient.return_value.interface.return_value
            iface.scanForNetworksWithName_error_.return_value = (scans, None)

            result = scan_for_networks("Room", "N")

        bssids = [r["bssid"] for r in result]
        assert len(bssids) == 2
        assert len(set(bssids)) == 2

    def test_scan_error_raises_runtime_error(self):
        with patch("src.data_collection.data_collector_service.CWWiFiClient") as mock_client:
            iface = mock_client.sharedWiFiClient.return_value.interface.return_value
            iface.scanForNetworksWithName_error_.return_value = ([], "Scan failed")

            with pytest.raises(RuntimeError, match="Wi-Fi scan failed"):
                scan_for_networks("Room", "N")

    def test_uses_current_coordinates_when_available(self):
        scans = [self._make_mock_scan("aa:bb:cc:dd:ee:01")]

        with patch("src.data_collection.data_collector_service.retrieve_current_location",
                   return_value=[40.7, -74.0]), \
             patch("src.data_collection.data_collector_service.CWWiFiClient") as mock_client:
            iface = mock_client.sharedWiFiClient.return_value.interface.return_value
            iface.scanForNetworksWithName_error_.return_value = (scans, None)

            result = scan_for_networks("Room", "N")

        assert result[0]["latitude"] == 40.7
        assert result[0]["longitude"] == -74.0

    def test_handles_no_location_gracefully(self):
        """If retrieve_current_location returns None, lat/lon fields become None."""
        scans = [self._make_mock_scan("aa:bb:cc:dd:ee:01")]

        with patch("src.data_collection.data_collector_service.retrieve_current_location",
                   return_value=None), \
             patch("src.data_collection.data_collector_service.CWWiFiClient") as mock_client:
            iface = mock_client.sharedWiFiClient.return_value.interface.return_value
            iface.scanForNetworksWithName_error_.return_value = (scans, None)

            result = scan_for_networks("Room", "N")

        assert result[0]["latitude"] is None
        assert result[0]["longitude"] is None