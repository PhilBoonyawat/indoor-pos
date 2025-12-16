import AppKit
import CoreLocation
import Foundation

class LocationDelegate(Foundation.NSObject):
    def locationManager_didUpdateLocations_(self, manager, locations):
        loc = locations[-1]
        coord = loc.coordinate()
        print(coord.latitude, coord.longitude)

        # Stop after first update if you want "current location"
        manager.stopUpdatingLocation()
        AppKit.NSApp.terminate_(None)

    def locationManager_didFailWithError_(self, manager, error):
        print(error.localizedDescription())
        AppKit.NSApp.terminate_(None)

def retrieve_current_location():
    app = AppKit.NSApplication.sharedApplication()

    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = LocationDelegate.alloc().init()

    manager.setDelegate_(delegate)

    # this should prompt permission access if in infoplist
    manager.requestWhenInUseAuthorization()
    manager.startUpdatingLocation()

    # AppKit.NSApp.run()
    app.run()
