import threading
import time
import subprocess
import platform
import re
import sqlite3
import json
import os
from datetime import datetime


class ScannerThread:
    """
    Background thread that periodically scans WiFi networks
    and runs position predictions.
    Falls back to demo mode if scanning fails or user is outside the building.
    """

    def __init__(self, predictor, interval=3, db_path=None):
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
        """Start the background scanning thread."""
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        print(f"[Scanner] Started scanning every {self.interval}s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[Scanner] Stopped")

    def is_running(self):
        return self._running

    def get_scan_mode(self):
        """Returns current scan mode."""
        return self._last_mode

    def get_current_position(self):
        with self._lock:
            return self._current_position.copy()

    def get_history(self, limit=20):
        with self._lock:
            return list(self._history[-limit:])

    def _scan_loop(self):
        """Main scanning loop with smart fallback."""
        while self._running:
            try:
                fingerprint = None
                mode = self._last_mode  # persist mode from last successful scan

                # 1. Try live WiFi scan
                live_scan = self._try_live_scan()

                if live_scan is not None:
                    # 2. Check if we're in the building (do we see known APs?)
                    known_aps = sum(1 for ap in live_scan if ap in self.predictor.feature_names)

                    if known_aps >= 3:
                        # We're in the building — use live scan
                        fingerprint = live_scan
                        mode = "live"
                        self._last_mode = "live"
                        self._last_fingerprint = live_scan
                        print(f"[Scanner] Live scan — {len(live_scan)} APs total, {known_aps} known")
                    else:
                        print(f"[Scanner] Outside building — {len(live_scan)} APs found, only {known_aps} known.")
                else:
                    # Scan failed (busy/permissions) — reuse last live scan if available
                    if self._last_fingerprint is not None and self._last_mode == "live":
                        fingerprint = self._last_fingerprint
                        mode = "live"

                # 3. Fall back to demo only if we have no usable scan at all
                if fingerprint is None:
                    if not self._demo_loaded:
                        self._load_demo_scans()
                        self._demo_loaded = True
                    fingerprint = self._get_demo_fingerprint()
                    mode = "demo"
                    self._last_mode = "demo"

                # 4. Run prediction
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

            time.sleep(self.interval)

    def _try_live_scan(self):
        """Attempt a live WiFi scan. Returns fingerprint dict or None."""
        system = platform.system()
        try:
            if system == "Darwin":
                return self._scan_macos()
            elif system == "Linux":
                return self._scan_linux()
            elif system == "Windows":
                return self._scan_windows()
            else:
                return None
        except Exception as e:
            return None

    def _scan_macos(self):
        """Scan WiFi on macOS using CoreWLAN framework (same as data collection)."""
        try:
            from CoreWLAN import CWWiFiClient

            client = CWWiFiClient.sharedWiFiClient()
            wifi_iface = client.interface()
            scans, scan_err = wifi_iface.scanForNetworksWithName_error_(None, None)

            if scan_err:
                # "Resource busy" is common when scanning too fast
                # Use the last successful scan instead of failing
                if "Resource busy" in str(scan_err) and hasattr(self, '_last_live_scan'):
                    return self._last_live_scan
                return None

            fingerprint = {}
            for scan in scans:
                bssid = scan.bssid()
                if bssid and bssid not in fingerprint:
                    fingerprint[bssid.lower()] = scan.rssiValue()

            if fingerprint:
                self._last_live_scan = fingerprint  # cache for busy errors

            return fingerprint if fingerprint else None

        except ImportError:
            print("[Scanner] CoreWLAN not available — install pyobjc-framework-CoreWLAN")
            return None

    def _scan_linux(self):
        """Scan WiFi on Linux using nmcli."""
        result = subprocess.run(
            ["nmcli", "-t", "-f", "BSSID,SIGNAL", "dev", "wifi", "list", "--rescan", "yes"],
            capture_output=True, text=True, timeout=10
        )

        fingerprint = {}
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    bssid = parts[0].strip().lower()
                    try:
                        signal_pct = int(parts[1].strip())
                        rssi = int(signal_pct / 2 - 100)
                        fingerprint[bssid] = rssi
                    except ValueError:
                        continue

        return fingerprint if fingerprint else None

    def _scan_windows(self):
        """Scan WiFi on Windows using netsh."""
        result = subprocess.run(
            ["netsh", "wlan", "show", "networks", "mode=bssid"],
            capture_output=True, text=True, timeout=10
        )

        fingerprint = {}
        current_bssid = None

        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("BSSID"):
                current_bssid = line.split(":", 1)[1].strip().lower()
            elif "Signal" in line and current_bssid:
                try:
                    signal_pct = int(line.split(":")[1].strip().replace("%", ""))
                    rssi = int(signal_pct / 2 - 100)
                    fingerprint[current_bssid] = rssi
                    current_bssid = None
                except ValueError:
                    continue

        return fingerprint if fingerprint else None

    def _load_demo_scans(self):
        """Load real scans from database for realistic demo playback."""
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
        """Return next demo scan in rotation."""
        if not self._demo_scans:
            return {}

        fingerprint = self._demo_scans[self._demo_index]
        self._demo_index = (self._demo_index + 1) % len(self._demo_scans)
        return fingerprint