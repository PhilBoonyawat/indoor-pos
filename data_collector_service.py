from CoreWLAN import CWWiFiClient
from datetime import datetime
from location_service import retrieve_current_location
import sys
import csv
import os
import argparse


def scan_for_networks(location, orientation):
    currentCoords = retrieve_current_location()

    if (currentCoords is None):
        raise Exception("Unable to retrieve current coordinates.")
    
    latitude, longitude = currentCoords
    
    client = CWWiFiClient.sharedWiFiClient()
    wifi_iface = client.interface()
    print(wifi_iface.interfaceName)
    scans, scan_err = wifi_iface.scanForNetworksWithName_error_(None, None)

    if scan_err:
        raise RuntimeError(f"Wi-Fi scan failed: {scan_err}")
    
    scan_data = []
    bssid_set = set()
    counter = 0
    for scan in scans:
        now = datetime.now().isoformat()
        # slow
        if scan.bssid() in bssid_set:
            print("duplication: " + str(scan.bssid()))
            continue
        bssid_set.add(scan.bssid())
        counter += 1
        print(
            "SSID:", scan.ssid(),
            "BSSID:", scan.bssid(),
            "RSSI:", scan.rssiValue(),
            "Noise:", scan.noiseMeasurement(),
            "Channel:", scan.wlanChannel().channelNumber(),
            "Current Time:", now,
            "latitude: ", latitude,
            "longitude: ", longitude
        )
        scan_data.append({
            'ssid': scan.ssid(),
            'bssid': scan.bssid(),
            'rssi': scan.rssiValue(),
            'noise': scan.noiseMeasurement(),
            'channel': scan.wlanChannel().channelNumber(),
            'timestamp': now,
            'location': location,
            'latitude': latitude,
            'longitude': longitude,
            'orientation': orientation
        })
    print(len(bssid_set))
    print("Total scanned networks: " + str(counter))
    return scan_data
    
def parse_args():
    p = argparse.ArgumentParser(description="Collect Wi‑Fi scan data to CSV")
    p.add_argument("-l", "--location", help="Location in the building in which you are in")
    p.add_argument("-o", "--out", default="test.csv", help="Output CSV file (default: wifi_scans.csv)")
    p.add_argument("-f", "--orientation", help="Please enter the direction you are facing (orientation)")
    return p.parse_args()

def write_data_to_csv():
    args = parse_args()

    location = args.location
    if not location:
        try:
            location = input("Enter your current location: ").strip()
        except EOFError:
            location = ""
    if not location:
        print("Location is required.", file=sys.stderr)
        return 3
    
    orientation = args.orientation
    if not orientation:
        try:
            orientation = input("Enter your orientation: ").strip()
        except EOFError:
            orientation = ""
    if not orientation:
        print("Orientation is required.", file=sys.stderr)
        return 4
    
    output_file = args.out
    if not output_file:
        try:
            output_file = input("Enter output CSV file path: ").strip()
        except EOFError:
            output_file = "test.csv"
    if not output_file:
        print("Output file path is required.", file=sys.stderr)
        return 5
    
    try:
        scan_data = scan_for_networks(location, orientation)
    except Exception as e:
        print(f"Error during Wi-Fi scan: {e}", file=sys.stderr)
        sys.exit(1)

    file_exists = os.path.exists(output_file)
       
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['scan_id', 'ssid', 'bssid', 'rssi', 'noise', 'channel', 'timestamp', 'location', 'latitude', 'longitude', 'orientation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(scan_data)


if __name__ == "__main__":
    write_data_to_csv()


