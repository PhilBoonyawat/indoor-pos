import sqlite3
import uuid

def init_db(db_path="wifi_scans.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS wifi_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT,
        ssid TEXT,
        bssid TEXT,
        rssi INTEGER,
        noise INTEGER,
        channel INTEGER,
        timestamp TEXT,
        location TEXT,
        latitude REAL,
        longitude REAL,
        orientation TEXT
    );
    """)

    conn.commit()
    return conn


def store_raw_scan(conn, scan_data):
    cur = conn.cursor()

    scan_id = str(uuid.uuid4())

    rows = [
        (
            scan_id,
            row["ssid"],
            row["bssid"],
            row["rssi"],
            row["noise"],
            row["channel"],
            row["timestamp"],
            row["location"],
            row["latitude"],
            row["longitude"],
            row["orientation"]
        )
        for row in scan_data
    ]

    cur.executemany("""
        INSERT INTO wifi_raw
        (scan_id, ssid, bssid, rssi, noise, channel, timestamp,
         location, latitude, longitude, orientation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    return scan_id

