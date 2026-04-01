# TODO: review this file
import sqlite3
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = PROJECT_ROOT / "data" / "raw" / "wifi_scans.db"

def init_db(db_path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scan_metadata (
        scan_id TEXT PRIMARY KEY,
        timestamp TEXT,
        location TEXT,
        latitude REAL,
        longitude REAL,
        orientation TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ssid (
        bssid TEXT PRIMARY KEY,
        ssid TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wifi_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT REFERENCES scan_metadata(scan_id),
        bssid TEXT REFERENCES ssid(bssid),
        rssi INTEGER,
        noise INTEGER,
        channel INTEGER
    );
    """)

    conn.commit()
    return conn


def store_raw_scan(conn, scan_data):
    cur = conn.cursor()
    scan_id = str(uuid.uuid4())

    if not scan_data:
        raise ValueError("No scan data provided.")


    # 1. Insert scan metadata (same for all rows in this scan)
    first = scan_data[0]
    cur.execute("""
        INSERT INTO scan_metadata (scan_id, timestamp, location, latitude, longitude, orientation)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (scan_id, first["timestamp"], first["location"],
          first["latitude"], first["longitude"], first["orientation"]))

    # 2. Insert unique BSSIDs into ssid table
    cur.executemany("""
        INSERT OR IGNORE INTO ssid (bssid, ssid)
        VALUES (?, ?)
    """, [(r["bssid"], r["ssid"]) for r in scan_data])


    # 3. Insert wifi scan results
    cur.executemany("""
        INSERT INTO wifi_scan (scan_id, bssid, rssi, noise, channel)
        VALUES (?, ?, ?, ?, ?)
    """, [(scan_id, r["bssid"], r["rssi"], r["noise"], r["channel"]) for r in scan_data])

    conn.commit()
    return scan_id

