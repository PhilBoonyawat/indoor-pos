# TODO: review this file
from CoreWLAN import CWWiFiClient
from datetime import datetime
from data_collection.location_service import retrieve_current_location
import sys
import argparse
from data_collection.db_service import init_db, store_raw_scan

def scan_for_networks(location, orientation):
    currentCoords = retrieve_current_location()
    latitude, longitude = currentCoords if currentCoords else (None, None)
    
    client = CWWiFiClient.sharedWiFiClient()
    wifi_iface = client.interface()
    scans, scan_err = wifi_iface.scanForNetworksWithName_error_(None, None)

    if scan_err:
        raise RuntimeError(f"Wi-Fi scan failed: {scan_err}")
    
    scan_data = []
    bssid_set = set()
    counter = 0
    now = datetime.now().isoformat()
    for scan in scans:
        # avoid replicating the same BSSID multiple times in one scan
        if scan.bssid() in bssid_set:
            continue
        bssid_set.add(scan.bssid())
        counter += 1
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
    print(f"Total scanned networks: {counter}")
    return scan_data
    
def parse_args():
    p = argparse.ArgumentParser(description="Collect Wi-Fi scan data")
    p.add_argument("-l", "--location", help="Location in the building in which you are in")
    p.add_argument("-f", "--orientation", help="Please enter the direction you are facing (orientation)")
    return p.parse_args()

def write_data_to_db(location, orientation):

    if not location:
        raise ValueError("Location is required")

    if not orientation:
        raise ValueError("Orientation is required")

    scan_data = scan_for_networks(location, orientation)

    conn = init_db()
    try:
        store_raw_scan(conn, scan_data)
    finally:
        conn.close()

def main():
    args = parse_args()
    
    location = args.location or input("Enter your current location: ").strip()
    orientation = args.orientation or input("Enter your orientation: ").strip()

    try:
        write_data_to_db(location, orientation)
    except ValueError as e:
        print(f"[INPUT ERROR] {e}", file=sys.stderr)
        sys.exit(2)

    except RuntimeError as e:
        print(f"[SCAN ERROR] {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()

