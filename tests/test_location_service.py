"""Tests for src/data_collection/location_service.py
#TODO
================================================================================
COVERAGE JUSTIFICATION — WHY THIS FILE SITS AT ~35%
================================================================================

This module is an integration layer between Python and macOS's CoreLocation
framework. Most of its code paths can only execute when three conditions are
ALL true simultaneously:

  1. Running on real macOS with PyObjC installed (not mocked).
  2. A real `CLLocationManager` object allocated by the Obj-C runtime.
  3. The macOS event loop actively dispatching delegate callbacks.

In a unit-test environment at least one of those is always missing, so
the following pieces cannot be exercised in isolation:

──────────────────────────────────────────────────────────────────
UNTESTABLE — `LocationDelegate` class (lines 31-98)
──────────────────────────────────────────────────────────────────
The class subclasses `Foundation.NSObject`. PyObjC binds this class to the
Objective-C runtime at class-definition time, which rewrites method names
using Obj-C selectors (e.g. `locationManagerDidChangeAuthorization_`
becomes a selector, not a Python attribute). As a result:

  • On Linux CI, `Foundation.NSObject` is a `MagicMock`, so class-body
    execution silently produces a MagicMock subclass — no real methods get
    attached. The test fails with "AttributeError: Mock object has no
    attribute 'locationManagerDidChangeAuthorization_'".

  • On macOS with real PyObjC, the methods exist but attribute lookup
    doesn't follow normal Python rules. `LocationDelegate.methodName_`
    accessed as an unbound function behaves differently from how Python
    would normally expect.

  • Even if we could access the methods, they're designed to be INVOKED
    by the OS as delegate callbacks via the run loop. Calling them
    manually with fake `manager` and `locations` arguments tests our
    mocks, not the actual Obj-C integration.

The `LocationDelegate.init` (lines 31-39) cannot be tested at all —
it calls `objc.super(LocationDelegate, self).init()` which is the
Obj-C init chain. With `objc` mocked this returns a MagicMock; with
real `objc` it can only be called on a freshly alloc'd Obj-C instance.

──────────────────────────────────────────────────────────────────
UNTESTABLE — `trigger_permission_prompt` (lines 130-142)
──────────────────────────────────────────────────────────────────
Calls `Foundation.CFRunLoopRunInMode(kCFRunLoopDefaultMode, 5.0, False)`
which blocks the thread for 5 real seconds waiting for OS events. In a
unit test there ARE no OS events — the function just wastes 5 seconds
doing nothing. Mocking the run-loop call is possible but then we're
testing what happens when the system never responds, which is the
error path — not the actual behaviour we care about.

──────────────────────────────────────────────────────────────────
UNTESTABLE — `retrieve_current_location` (lines 177-207)
──────────────────────────────────────────────────────────────────
Same problem, 15-second timeout. The happy path requires real GPS
hardware to fire a delegate callback that populates
`delegate.coordinates`, and the run loop to dispatch that callback to
our delegate. Neither can be simulated in isolation.

──────────────────────────────────────────────────────────────────
TESTED — pure logic that doesn't need PyObjC or the run loop
──────────────────────────────────────────────────────────────────
`check_location_permission()` — a 6-line function that wraps
`manager.authorizationStatus()` and maps the integer to a string. We
mock the manager and verify every status code maps correctly. 100%
covered.

`wait_for_location_permission()` — a polling loop that calls
`check_location_permission` and `time.sleep`. Mocked both and
verified: immediate-grant, timeout, mid-poll grant, and interval
timing. 100% covered.

──────────────────────────────────────────────────────────────────
VERIFICATION STRATEGY FOR THE UNTESTED PARTS
──────────────────────────────────────────────────────────────────
The untested code paths are verified by running the app end-to-end
on macOS:

  1. `python run.py` on a fresh Mac triggers `trigger_permission_prompt`
     and the System Settings registration flow. Verified manually.

  2. `python run.py` on a Mac with permission already granted exercises
     `retrieve_current_location` and attaches real GPS coordinates to
     each scan. Verified by checking the DB rows after data collection.

  3. The `LocationDelegate` callback methods fire during both flows.
     Their `print()` statements ("[Location] Permission granted.",
     "[Location] Got coordinates: ...") confirm they executed.

The ~35% number is an honest ceiling given the framework boundary.
Pushing it higher would require either (a) running against real macOS
in CI, or (b) testing our own mocks, neither of which provides
additional confidence in correctness.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── Mock PyObjC frameworks so collection works on any OS ──────────
sys.modules["CoreLocation"] = MagicMock()
sys.modules["Foundation"] = MagicMock()
sys.modules["objc"] = MagicMock()

from src.data_collection import location_service  # noqa: E402


# Pin the status-code constants. Real macOS uses these integer values, and
# the module compares `status == CoreLocation.k...` so we must set them.
location_service.CoreLocation.kCLAuthorizationStatusAuthorizedWhenInUse = 4
location_service.CoreLocation.kCLAuthorizationStatusAuthorizedAlways = 3
location_service.CoreLocation.kCLAuthorizationStatusNotDetermined = 0


def _install_manager(status_code, monkeypatch):
    """Replace `CLLocationManager.alloc().init()` with a mock manager whose
    `authorizationStatus()` returns the given code."""
    manager = MagicMock()
    manager.authorizationStatus.return_value = status_code

    alloc_chain = MagicMock()
    alloc_chain.init.return_value = manager

    cls = MagicMock()
    cls.alloc.return_value = alloc_chain
    monkeypatch.setattr(location_service.CoreLocation, "CLLocationManager", cls)
    return manager


# ── check_location_permission ──────────────────────────────────

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
        """Status 2 in CoreLocation is "Denied"."""
        _install_manager(2, monkeypatch)
        assert location_service.check_location_permission() == "denied"

    def test_returns_denied_for_restricted(self, monkeypatch):
        """Status 1 is "Restricted" (e.g. parental controls). Also treated
        as denied by the app."""
        _install_manager(1, monkeypatch)
        assert location_service.check_location_permission() == "denied"


# ── wait_for_location_permission ───────────────────────────────

class TestWaitForLocationPermission:
    """Tests the polling loop. Covers every branch: immediate return,
    timeout, mid-poll completion, and interval usage."""

    def test_returns_true_immediately_if_already_granted(self):
        with patch.object(location_service, "check_location_permission",
                          return_value="granted"), \
             patch.object(location_service.time, "sleep") as mock_sleep:
            assert location_service.wait_for_location_permission(30, 1) is True
        mock_sleep.assert_not_called()

    def test_returns_false_on_timeout(self):
        """Permission never arrives — wait must eventually time out."""
        with patch.object(location_service, "check_location_permission",
                          return_value="not_determined"), \
             patch.object(location_service.time, "sleep"):
            assert location_service.wait_for_location_permission(6, 2) is False

    def test_returns_true_when_permission_arrives_mid_poll(self):
        """User toggles permission on during the wait — function should
        return True on the next poll."""
        responses = iter(["not_determined", "not_determined", "granted"])
        with patch.object(location_service, "check_location_permission",
                          side_effect=lambda: next(responses)), \
             patch.object(location_service.time, "sleep"):
            assert location_service.wait_for_location_permission(30, 1) is True

    def test_polls_at_configured_interval(self):
        """Every `time.sleep` call should use the `poll_interval` value."""
        with patch.object(location_service, "check_location_permission",
                          return_value="denied"), \
             patch.object(location_service.time, "sleep") as mock_sleep:
            location_service.wait_for_location_permission(9, 3)
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 3