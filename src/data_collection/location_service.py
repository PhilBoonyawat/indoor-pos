import CoreLocation
import Foundation
import objc

class LocationDelegate(Foundation.NSObject):
    def init(self):
        self = objc.super(LocationDelegate, self).init()
        if self is None:
            return None

        self.coordinates = None
        self.error = None
        return self

    def locationManager_didUpdateLocations_(self, manager, locations):
        loc = locations[-1]
        coord = loc.coordinate()
        self.coordinates = [coord.latitude, coord.longitude]
        print(coord.latitude, coord.longitude)

        # Stop after first update if you want "current location"
        manager.stopUpdatingLocation()
        Foundation.CFRunLoopStop(Foundation.CFRunLoopGetCurrent())


    def locationManager_didFailWithError_(self, manager, error):
        self.coordinates = None
        self.error = error.localizedDescription()
        manager.stopUpdatingLocation()
        Foundation.CFRunLoopStop(Foundation.CFRunLoopGetCurrent())

        
def retrieve_current_location():
    manager = CoreLocation.CLLocationManager.alloc().init()
    delegate = LocationDelegate.alloc().init()

    manager.setDelegate_(delegate)

    # this should prompt permission access if in infoplist
    manager.requestWhenInUseAuthorization()
    manager.startUpdatingLocation()

    Foundation.CFRunLoopRun()

    if delegate.error:
        raise RuntimeError(f"Location error: {delegate.error}")

    if delegate.coordinates is None:
        raise RuntimeError("Location unavailable")

    return delegate.coordinates