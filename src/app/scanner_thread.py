"""
scanner_thread.py — Handles scanning of Wi-Fi scans.

Creates a background thread that periodically scans WiFi networks and runs position predictions. 
Falls back to demo mode if scanning fails or user is outside the building or location where access points are not recognised.

Usage:
    from scanner_thread import ScannerThread
    scanner = ScannerThread(predictor=my_predictor, interval=3, db_path="path/to/db")
    scanner.start()
"""

import threading
import time
import subprocess
from CoreWLAN import CWWiFiClient
import platform
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'wifi_scans.db')

class ScannerThread:
    """
    Background thread that periodically scans Wi-Fi networks
    and runs position predictions.

    Live Mode: Performs real Wi-Fi scans and uses the predictor to estimate position. 
    Demo Mode: Uses pre-recorded scans from the database for demonstration when live scanning fails (e.g. permissions, outside building, access points not recognised).
    """

    def __init__(self, predictor, interval=3, db_path=DB_PATH):
        """
        Initializes the ScannerThread with a predictor, scan interval, and optional database path for demo mode.

        Args:
            predictor: An instance of the Predictor class used to make position predictions from Wi-Fi scans.
            interval: Time in seconds between each scan (default = 3).
            db_path: Optional path to the SQLite database file for loading demo scans. If not provided, demo mode will not have real scans available.
        """
        
        self.predictor = predictor
        self.interval = interval
        self.db_path = db_path
        self._current_position = {
            "room": "Unknown",
            "confidence": 0.0,
            "model_used": "None",
            "position_x": 0,
            "position_y": 0,
            "aps_detected": 0,
            "mode": "initialising",
            "timestamp": None
        }
        self._history = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._demo_scans = []
        self._demo_index = 0
        self._demo_loaded = False
        self._last_mode = "demo"
        self._last_fingerprint = None

    def start(self):
        """
        Start the background scanning thread.
        """
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print(f"[Scanner] Started scanning every {self.interval}s")

    def stop(self):
        """
        Stop the background scanning thread gracefully.
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[Scanner] Stopped")

    def is_running(self):
        """
        Check if the scanner thread is currently running.

        Returns:
            bool: True if the scanner is running, False otherwise.
        """
        return self._running

    def get_scan_mode(self):
        """
        Returns current scan mode.

        Returns:
            str: "live" if using live Wi-Fi scans, "demo" if using pre-recorded demo scans, or "initialising" if not yet determined.
        """
        return self._last_mode

    def get_current_position(self):
        """
        Get the latest predicted position in a thread-safe way.

        Returns:
            dict: A dictionary containing the latest position prediction, including room, confidence, 
                  model used, coordinates, number of APs detected, mode, and timestamp.
        """
        
        with self._lock:
            return self._current_position.copy()

    def get_history(self, limit=20):
        """
        Get recent scan history for trail visualization in a thread-safe way.

        Args:
            limit: Maximum number of recent scans to return (default = 20)

        Returns:         
            list: A list of dictionaries containing recent position predictions, ordered from oldest to newest, limited to the specified number. 
                  Each dictionary includes room, confidence, model used, coordinates, number of APs detected, mode, and timestamp.
        """
        
        with self._lock:
            return list(self._history[-limit:])

    def _scan_loop(self):
        """
        Main scanning loop with smart fallback.
        """
        while self._running:
            try:
                fingerprint = None
                mode = self._last_mode 

                live_scan = self._try_live_scan()

                if live_scan is not None:
                    known_aps = sum(1 for ap in live_scan if ap in self.predictor.feature_names)

                    if known_aps >= 3:
                        # Use live scan since we have enough known APs for a reliable prediction
                        fingerprint = live_scan
                        mode = "live"
                        self._last_mode = "live"
                        self._last_fingerprint = live_scan
                        print(f"[Scanner] Live scan — {len(live_scan)} APs total, {known_aps} known")
                    else:
                        print(f"[Scanner] Outside known area — {len(live_scan)} APs found, only {known_aps} known.")
                else:
                    # Scan failed (busy/permissions) — reuse last live scan if available
                    if self._last_fingerprint is not None and self._last_mode == "live":
                        fingerprint = self._last_fingerprint
                        mode = "live"

                # Fall back to demo only if we have no usable scan at all
                if fingerprint is None:
                    if not self._demo_loaded:
                        self._load_demo_scans()
                        self._demo_loaded = True
                    fingerprint = self._get_demo_fingerprint()
                    mode = "demo"
                    self._last_mode = "demo"

                if fingerprint is not None:
                    prediction = self.predictor.predict(fingerprint)
                    prediction["mode"] = mode
                    prediction["timestamp"] = datetime.now().isoformat()

                    with self._lock:
                        self._current_position = prediction
                        self._history.append(prediction)
                        if len(self._history) > 100:
                            self._history = self._history[-100:]

            except Exception as e:
                print(f"[Scanner] Error: {e}")

            # Wait for the specified interval before the next scan
            time.sleep(self.interval)

    def _try_live_scan(self):
        """
        Attempt a live WiFi scan. Returns fingerprint dict or None.

        Returns:
            dict or None: A dictionary mapping BSSID to RSSI if scan successful, or None
        """
        system = platform.system()
        try:
            if system == "Darwin":
                return self._scan_macos()
            else:
                return None
        except Exception as e:
            return None

    def _scan_macos(self):
        """
        Scan Wi-Fi on macOS using CoreWLAN framework (same as data collection).

        Returns:
            dict or None: A dictionary mapping BSSID to RSSI if scan successful, or None if scan failed (e.g. permissions, busy)
        """
        
        client = CWWiFiClient.sharedWiFiClient()
        wifi_iface = client.interface()
        scans, scan_err = wifi_iface.scanForNetworksWithName_error_(None, None)

        if scan_err:
            # Use the last successful scan instead of failing
            if "Resource busy" in str(scan_err) and hasattr(self, '_last_live_scan'):
                print("[Scanner] Resource busy — using cached scan")
                return self._last_live_scan
            print(f"[Scanner] CoreWLAN error: {scan_err}")
            return None

        fingerprint = {}
        for scan in scans:
            bssid = scan.bssid()
            if bssid and bssid not in fingerprint:
                fingerprint[bssid.lower()] = scan.rssiValue()

        if fingerprint:
            self._last_live_scan = fingerprint  # cache for busy errors

        return fingerprint if fingerprint else None

    def _load_demo_scans(self):
        """
        Load real scans from database for realistic demo playback.
        """
        if not self.db_path or not os.path.exists(self.db_path):
            print("[Scanner] No database found for demo mode — using random predictions")
            return

        try:
            conn = sqlite3.connect(self.db_path)

            # Get one scan per room for demo rotation
            scan_ids = conn.execute("""
                SELECT scan_id, location FROM scan_metadata
                GROUP BY location
                ORDER BY location
            """).fetchall()

            for scan_id, location in scan_ids:
                rows = conn.execute("""
                    SELECT bssid, rssi FROM wifi_scan WHERE scan_id = ?
                """, (scan_id,)).fetchall()

                fingerprint = {bssid: rssi for bssid, rssi in rows}
                self._demo_scans.append(fingerprint)

            conn.close()
            print(f"[Scanner] Loaded {len(self._demo_scans)} demo scans from database")

        except Exception as e:
            print(f"[Scanner] Failed to load demo scans: {e}")

    def _get_demo_fingerprint(self):
        """
        Return next demo scan in rotation.
        
        Returns:
            dict or None: A dictionary mapping BSSID to RSSI for the next demo scan, or None if no demo scans available.
        """
        if not self._demo_scans:
            return {}

        fingerprint = self._demo_scans[self._demo_index]
        self._demo_index = (self._demo_index + 1) % len(self._demo_scans)
        return fingerprint