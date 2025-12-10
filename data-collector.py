from CoreWLAN import CWWiFiClient
from datetime import datetime
import sys

def scan_for_networks():
    client = CWWiFiClient.sharedWiFiClient()
    wifi_iface = client.interface()
    print(wifi_iface.interfaceName)
    scans, scan_err = wifi_iface.scanForNetworksWithName_error_(None, None)

    if scan_err:
        sys.exit("Scan error: " + str(scan_err))
    else:
        for scan in scans:
            now = datetime.now().isoformat()
            print(
                "SSID:", scan.ssid(),
                "BSSID:", scan.bssid(),
                "RSSI:", scan.rssiValue(),
                "Noise:", scan.noiseMeasurement(),
                "Channel:", scan.wlanChannel().channelNumber(),
                "Current Time:", now
            )


if __name__ == "__main__":
    scan_for_networks()


