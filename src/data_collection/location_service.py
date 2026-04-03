# TODO: review this file
"""
location_service.py — Handles retrieval of GPS coordinates on macOS using CoreLocation. 

Designed to be called by data_collector_service.py during Wi-Fi scans to attach location data to each scan record.

Usage:
    from data_collection.location_service import retrieve_current_location
    coords = retrieve_current_location()
"""

import CoreLocation
import Foundation
import objc
import time

class LocationDelegate(Foundation.NSObject):
    def init(self):
        self = objc.super(LocationDelegate, self).init()
        if self is None:
            return None

        self.coordinates = None
        self.error = None
        self._authorized = False
        self._authorization_resolved = False
        return self

    def locationManagerDidChangeAuthorization_(self, manager):
        """
        Called when authorization status changes (including after the user responds to the prompt).

        Args:
            manager: The CLLocationManager instance that triggered the authorization change.
        """
        status = manager.authorizationStatus()

        if status == CoreLocation.kCLAuthorizationStatusNotDetermined:
            return

        self._authorization_resolved = True

        if status in (
            CoreLocation.kCLAuthorizationStatusAuthorizedAlways,
            CoreLocation.kCLAuthorizationStatusAuthorizedWhenInUse,
        ):
            self._authorized = True
            print("[Location] Permission granted.")
            manager.startUpdatingLocation()
        else:
            self._authorized = False
            self.error = "Location permission denied."
            print(f"[Location] {self.error}")
            Foundation.CFRunLoopStop(Foundation.CFRunLoopGetCurrent())

    def locationManager_didUpdateLocations_(self, manager, locations):
        """
        Called when new location data is available.

        Args:
            manager: The CLLocationManager instance that triggered the update.
            locations: An array of CLLocation objects, with the most recent location last.
        """
        
        loc = locations[-1]
        coord = loc.coordinate()
        self.coordinates = [coord.latitude, coord.longitude]
        print(f"[Location] Got coordinates: {coord.latitude}, {coord.longitude}")

        manager.stopUpdatingLocation()
        Foundation.CFRunLoopStop(Foundation.CFRunLoopGetCurrent())

    def locationManager_didFailWithError_(self, manager, error):
        """
        Called when there is an error retrieving location data.

        Args:
            manager: The CLLocationManager instance that triggered the error.
            error: An NSError object describing the error that occurred.
        """
        
        self.coordinates = None
        self.error = error.localizedDescription()
        print(f"[Location] Error: {self.error}")
        manager.stopUpdatingLocation()
        Foundation.CFRunLoopStop(Foundation.CFRunLoopGetCurrent())

def check_location_permission():
    """
    Check the current location authorization status.

    Returns:
        'granted' if permission is granted,
        'not_determined' if the user has not yet been prompted,
        'denied' if permission is denied.
    """
    manager = CoreLocation.CLLocationManager.alloc().init()
    status = manager.authorizationStatus()

    if status in (
        CoreLocation.kCLAuthorizationStatusAuthorizedAlways,
        CoreLocation.kCLAuthorizationStatusAuthorizedWhenInUse,
    ):
        return "granted"
    elif status == CoreLocation.kCLAuthorizationStatusNotDetermined:
        return "not_determined"
    else:
        return "denied"


def trigger_permission_prompt():
    """
    Attempt to trigger the macOS location permission prompt.

    In a CLI context this may not show a dialog, but it registers Python
    (or the terminal app) in System Settings > Location Services, which the user can then toggle it on manually.
    """
    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = LocationDelegate.alloc().init()
    manager.setDelegate_(delegate)

    print("[Location] Triggering location request to register with macOS...")
    manager.requestWhenInUseAuthorization()

    # Give macOS a moment to process and potentially show a dialog
    Foundation.CFRunLoopRunInMode(
        Foundation.kCFRunLoopDefaultMode, 5.0, False
    )

    return check_location_permission()


def wait_for_location_permission(timeout=120, poll_interval=3):
    """
    Poll until location permission is granted or timeout is reached.

    This is designed for the CLI workflow where the user needs to manually
    toggle the permission in System Settings after we've registered Python.

    Args:
        timeout: max seconds to wait (default: 120)
        poll_interval: seconds between checks (default: 3)

    Returns:
        True if permission was granted, False if timed out
    """
    elapsed = 0
    while elapsed < timeout:
        status = check_location_permission()
        if status == "granted":
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval

    return False


def retrieve_current_location():
    """
    Fetch current GPS coordinates. Assumes permission has already been granted.

    Returns:
        List of [latitude, longitude]
    """
    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = LocationDelegate.alloc().init()

    manager.setDelegate_(delegate)

    status = manager.authorizationStatus()

    if status == CoreLocation.kCLAuthorizationStatusNotDetermined:
        print("[Location] Requesting permission...")
        manager.requestWhenInUseAuthorization()
    elif status in (
        CoreLocation.kCLAuthorizationStatusAuthorizedAlways,
        CoreLocation.kCLAuthorizationStatusAuthorizedWhenInUse,
    ):
        manager.startUpdatingLocation()
    else:
        raise RuntimeError(
            "Location permission denied. Enable in System Settings > Privacy & Security > Location Services."
        )

    Foundation.CFRunLoopRunInMode(
        Foundation.kCFRunLoopDefaultMode, 15.0, False
    )

    if delegate.error:
        raise RuntimeError(f"Location error: {delegate.error}")

    if delegate.coordinates is None:
        raise RuntimeError("Location unavailable (timed out)")

    return delegate.coordinates