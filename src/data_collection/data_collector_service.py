from CoreWLAN import CWWiFiClient
from datetime import datetime
from datacollection.location_service import retrieve_current_location
import sys
import csv
import argparse
from datacollection.db_service import init_db, store_raw_scan
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "wifi_scans.csv"


def scan_for_networks(location, orientation):
    currentCoords = retrieve_current_location()
    latitude, longitude = currentCoords
    
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
        # slow
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
    print(len(bssid_set))
    print("Total scanned networks: " + str(counter))
    return scan_data
    
def parse_args():
    p = argparse.ArgumentParser(description="Collect Wi‑Fi scan data to CSV")
    p.add_argument("-l", "--location", help="Location in the building in which you are in")
    p.add_argument("-o", "--out", default=DEFAULT_CSV, help="Output CSV file (default: wifi_scans.csv)")
    p.add_argument("-f", "--orientation", help="Please enter the direction you are facing (orientation)")
    return p.parse_args()

def write_data_to_csv(location, orientation, output_file = DEFAULT_CSV):

    if not location:
        raise ValueError("Location is required")

    if not orientation:
        raise ValueError("Orientation is required")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    scan_data = scan_for_networks(location, orientation)

    file_exists = output_file.exists()

    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = [
            'scan_id','ssid','bssid','rssi',
            'noise','channel','timestamp',
            'location','latitude','longitude','orientation'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(scan_data)


def write_data_to_db(location, orientation):

    if not location:
        raise ValueError("Location is required")

    if not orientation:
        raise ValueError("Orientation is required")

    scan_data = scan_for_networks(location, orientation)

    conn = init_db()
    store_raw_scan(conn, scan_data)
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

