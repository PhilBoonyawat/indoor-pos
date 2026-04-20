"""Tests for src/data_collection/location_service.py"""

import sys
from unittest.mock import MagicMock, patch

sys.modules["CoreLocation"] = MagicMock()
sys.modules["Foundation"] = MagicMock()
sys.modules["objc"] = MagicMock()

from src.data_collection import location_service  # noqa: E402


# Status codes from Apple's CoreLocation framework, used in boolean logic and integer-to-string mapping in the app
location_service.CoreLocation.kCLAuthorizationStatusAuthorizedWhenInUse = 4
location_service.CoreLocation.kCLAuthorizationStatusAuthorizedAlways = 3
location_service.CoreLocation.kCLAuthorizationStatusNotDetermined = 0


def _install_manager(status_code, monkeypatch):
    """Replace CLLocationManager.alloc().init() with a mock manager whose
    authorizationStatus() returns the given code."""
    manager = MagicMock()
    manager.authorizationStatus.return_value = status_code

    alloc_chain = MagicMock()
    alloc_chain.init.return_value = manager

    cls = MagicMock()
    cls.alloc.return_value = alloc_chain
    monkeypatch.setattr(location_service.CoreLocation, "CLLocationManager", cls)
    return manager


class TestCheckLocationPermission:
    """Tests the integer-to-string mapping. Covers the only reachable path
    through this function."""

    def test_returns_granted_for_authorized_when_in_use(self, monkeypatch):
        _install_manager(4, monkeypatch)
        assert location_service.check_location_permission() == "granted"

    def test_returns_granted_for_authorized_always(self, monkeypatch):
        _install_manager(3, monkeypatch)
        assert location_service.check_location_permission() == "granted"

    def test_returns_not_determined_when_unprompted(self, monkeypatch):
        _install_manager(0, monkeypatch)
        assert location_service.check_location_permission() == "not_determined"

    def test_returns_denied_for_explicit_denial(self, monkeypatch):
        _install_manager(2, monkeypatch)
        assert location_service.check_location_permission() == "denied"

    def test_returns_denied_for_restricted(self, monkeypatch):
        _install_manager(1, monkeypatch)
        assert location_service.check_location_permission() == "denied"


class TestWaitForLocationPermission:
    """Tests the polling loop."""
    def test_returns_true_immediately_if_already_granted(self):
        with patch.object(location_service, "check_location_permission",
                          return_value="granted"), \
             patch.object(location_service.time, "sleep") as mock_sleep:
            assert location_service.wait_for_location_permission(30, 1) is True
        mock_sleep.assert_not_called()

    def test_returns_false_on_timeout(self):
        """Permission never arrives -> must eventually time out."""
        with patch.object(location_service, "check_location_permission",
                          return_value="not_determined"), \
             patch.object(location_service.time, "sleep"):
            assert location_service.wait_for_location_permission(6, 2) is False

    def test_returns_true_when_permission_arrives_mid_poll(self):
        """User toggles permission on during the wait —> function should return True on the next poll."""
        responses = iter(["not_determined", "not_determined", "granted"])
        with patch.object(location_service, "check_location_permission",
                          side_effect=lambda: next(responses)), \
             patch.object(location_service.time, "sleep"):
            assert location_service.wait_for_location_permission(30, 1) is True

    def test_polls_at_configured_interval(self):
        with patch.object(location_service, "check_location_permission",
                          return_value="denied"), \
             patch.object(location_service.time, "sleep") as mock_sleep:
            location_service.wait_for_location_permission(9, 3)
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 3