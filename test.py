from CoreWLAN import CWWiFiClient
from datetime import datetime
import argparse
import csv
import os
import sys

CSV_FIELDS = ["ssid", "bssid", "rssi", "noise", "channel", "time_utc", "location"]

def write_rows(csv_path, rows):
    first_write = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if first_write:
            writer.writeheader()
        writer.writerows(rows)

def scan_networks(location):
    client = CWWiFiClient.sharedWiFiClient()
    iface = client.interface()
    if iface is None:
        print("No Wi‑Fi interface found", file=sys.stderr)
        return 1

    scan, err = iface.scanForNetworksWithName_error_(None, None)
    if err is not None:
        print("Scan error:", err, file=sys.stderr)
        return 2

    time_utc = datetime.now().time().isoformat(timespec="seconds")
    rows = []
    for n in scan:
        rows.append({
            "ssid": n.ssid() or "<hidden>",
            "bssid": n.bssid() or "",
            "rssi": n.rssiValue(),
            "noise": n.noiseMeasurement(),
            "channel": n.wlanChannel().channelNumber(),
            "time_utc": time_utc,
            "location": location,
        })

    write_rows("wifi_scans.csv", rows)
    print(f"Wrote {len(rows)} rows to wifi_scans.csv")
    return 0

def parse_args():
    p = argparse.ArgumentParser(description="Collect Wi‑Fi scan data to CSV")
    p.add_argument("-l", "--location", help="Location label you will enter (required if not prompted)")
    p.add_argument("-o", "--out", default="wifi_scans.csv", help="Output CSV file (default: wifi_scans.csv)")
    return p.parse_args()

def main():
    args = parse_args()
    location = args.location
    if not location:
        try:
            location = input("Enter location label: ").strip()
        except EOFError:
            location = ""
    if not location:
        print("Location is required.", file=sys.stderr)
        return 3

    # Use requested output file name
    global write_rows
    def write_rows(csv_path, rows):
        first_write = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if first_write:
                writer.writeheader()
            writer.writerows(rows)

    # scan and save
    client = CWWiFiClient.sharedWiFiClient()
    iface = client.interface()
    if iface is None:
        print("No Wi‑Fi interface found", file=sys.stderr)
        return 1

    scan, err = iface.scanForNetworksWithName_error_(None, None)
    if err is not None:
        print("Scan error:", err, file=sys.stderr)
        return 2

    time_utc = datetime.utcnow().time().isoformat(timespec="seconds")
    rows = []
    for n in scan:
        rows.append({
            "ssid": n.ssid() or "<hidden>",
            "bssid": n.bssid() or "",
            "rssi": n.rssiValue(),
            "noise": n.noiseMeasurement(),
            "channel": n.wlanChannel().channelNumber(),
            "time_utc": time_utc,
            "location": location,
        })

    write_rows(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
