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
