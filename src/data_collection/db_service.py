"""
db_service.py — Initialize SQLite database and provide functions to store raw Wi-Fi scan data, including metadata and unique SSIDs.

This script is designed to be called by data_collector_service.py, which performs the actual Wi-Fi scanning and GPS retrieval. The db_service is responsible for ensuring the database schema is set up correctly and for inserting new scan records in a structured way.

Usage:
    from data_collection.db_service import init_db, store_raw_scan
"""

import sqlite3
import uuid
import os

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)

def init_db(db_path = DB_PATH):
    """
    Initialize the SQLite database with the required tables if they don't already exist. 
    The database will have three tables:
        - scan_metadata: Stores metadata for each Wi-Fi scan, including location, GPS coordinates, and orientation.
        - ssid: A lookup table to store unique BSSID-SSID pairs to avoid redundancy.
        - wifi_scan: Stores the actual Wi-Fi scan results, referencing the scan_metadata and ssid tables.

    Args:
        db_path: Path to the SQLite database file. Defaults to data/raw/wifi_scans.db.
    
    Returns:
        A sqlite3.Connection object connected to the initialized database.
    """
    
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
    """
    Store raw Wi-Fi scan data in the database. 

    Args:
        conn: A sqlite3.Connection object connected to the initialized database.
        scan_data: A list of dictionaries, where each dictionary contains the following keys:
            - ssid: The SSID of the Wi-Fi network.
            - bssid: The BSSID (MAC address) of the Wi-Fi network.
            - rssi: The signal strength (RSSI) of the Wi-Fi network.
            - noise: The noise measurement of the Wi-Fi network.
            - channel: The Wi-Fi channel number.
            - timestamp: The timestamp of the scan in ISO format.
            - location: The user-provided location string.
            - latitude: The GPS latitude (can be null if permission not granted).
            - longitude: The GPS longitude (can be null if permission not granted).
            - orientation: The user-provided orientation string.

    Returns:
        The scan_id of the inserted scan record, which can be used for reference.
    """
    
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

