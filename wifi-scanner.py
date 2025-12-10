# from CoreWLAN import CWWiFiClient
# import objc
# from Foundation import NSObject, NSRunLoop, NSDate

# # Load CoreLocation bundle
# objc.loadBundle('CoreLocation', globals(), bundle_path='/System/Library/Frameworks/CoreLocation.framework')

# # Get the CLLocationManager class
# CLLocationManager = objc.lookUpClass('CLLocationManager')

# # Authorization status constants
# CL_AUTH_STATUS_AUTHORIZED_WHEN_IN_USE = 4

# # Define delegate
# class Delegate(NSObject):
#     def init(self):
#         self = objc.super(Delegate, self).init()
#         self.locationManager = CLLocationManager.alloc().init()
#         self.locationManager.setDelegate_(self)
#         self.locationManager.requestWhenInUseAuthorization()
#         return self

#     def locationManager_didChangeAuthorizationStatus_(self, manager, status):
#         print("Authorization status changed:", status)
#         if status == CL_AUTH_STATUS_AUTHORIZED_WHEN_IN_USE:
#             print("✅ Location access granted.")
#         else:
#             print("❌ Location access not granted.")
#         import sys
#         sys.exit(0)

# # Trigger prompt
# delegate = Delegate.alloc().init()

# # Keep script alive until user responds
# while True:
#     NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.5))



# client = CWWiFiClient.sharedWiFiClient()
# iface = client.interface()
# scan, err = iface.scanForNetworksWithName_error_(None, None)

# for n in scan:
#     print(
#         "SSID:", n.ssid(),
#         "BSSID:", n.bssid(),
#         "RSSI:", n.rssiValue(),
#         "Noise:", n.noiseMeasurement(),
#         "Channel:", n.wlanChannel().channelNumber()
#     )



from CoreWLAN import CWWiFiClient

# Now scan Wi-Fi networks
client = CWWiFiClient.sharedWiFiClient()
iface = client.interface()
scan, err = iface.scanForNetworksWithName_error_(None, None)

if err:
    print("Scan error:", err)
else:
    for n in scan:
        print(
            "SSID:", n.ssid(),
            "BSSID:", n.bssid(),
            "RSSI:", n.rssiValue(),
            "Noise:", n.noiseMeasurement(),
            "Channel:", n.wlanChannel().channelNumber()
        )
