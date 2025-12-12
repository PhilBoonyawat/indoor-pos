# from CoreLocation import CLLocationManager
# from Foundation import NSObject


# class LocationDelegate(NSObject):

#     def locationManager_didUpdateLocations_(self, manager, locations):
#         location = locations[-1]   
#         coord = location.coordinate()
#         print(f"Latitude: {coord.latitude}, Longitude: {coord.longitude}")

#     def locationManager_didFailWithError_(self, manager, error):
#         print("Location failed:", error.localizedDescription())

    

# def retrieve_current_location():
#     location_manager = CLLocationManager.alloc().init()
#     location_manager_delegate = LocationDelegate.alloc().init()
#     location_manager.setDelegate_(location_manager_delegate)
#     print(location_manager.locationServicesEnabled())
#     location_manager.startUpdatingLocation()

# if __name__ == "__main__":
#     retrieve_current_location()


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

    # Keep references alive
    retrieve_current_location.manager = manager
    retrieve_current_location.delegate = delegate

    manager.setDelegate_(delegate)
    manager.requestWhenInUseAuthorization()
    manager.startUpdatingLocation()

    AppKit.NSApp.run()

if __name__ == "__main__":
    retrieve_current_location()