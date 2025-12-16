import AppKit
import CoreLocation
import Foundation
import objc

class LocationDelegate(Foundation.NSObject):
    def __init__(self):
        objc.super(LocationDelegate, self).init()
        self.coordinates = None

    def locationManager_didUpdateLocations_(self, manager, locations):
        loc = locations[-1]
        coord = loc.coordinate()
        self.coordinates = [coord.latitude, coord.longitude]
        print(coord.latitude, coord.longitude)

        # Stop after first update if you want "current location"
        manager.stopUpdatingLocation()
        AppKit.NSApp.terminate_(None)

    def locationManager_didFailWithError_(self, manager, error):
        print(f"Error: {error.localizedDescription()}")
        self.coordinates = None
        AppKit.NSApp.terminate_(None)


def retrieve_current_location():
    app = AppKit.NSApplication.sharedApplication()

    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = LocationDelegate.alloc().init()

    manager.setDelegate_(delegate)

    # this should prompt permission access if in infoplist
    manager.requestWhenInUseAuthorization()
    manager.startUpdatingLocation()

    app.run()

    return delegate.coordinates

