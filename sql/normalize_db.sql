BEGIN TRANSACTION;

-- 1. Create the normalized tables
CREATE TABLE scan_metadata (
    scan_id TEXT PRIMARY KEY,
    timestamp TEXT,
    location TEXT,
    latitude REAL,
    longitude REAL,
    orientation TEXT
);

CREATE TABLE ssid (
    bssid TEXT PRIMARY KEY,
    ssid TEXT
);

CREATE TABLE wifi_scan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id TEXT REFERENCES scan_metadata(scan_id),
    bssid TEXT REFERENCES ssid(bssid),
    rssi INTEGER,
    noise INTEGER,
    channel INTEGER
);

-- 2. Populate scan_metadata (one row per unique scan)
INSERT OR IGNORE INTO scan_metadata (scan_id, timestamp, location, latitude, longitude, orientation)
SELECT DISTINCT scan_id, timestamp, location, latitude, longitude, orientation
FROM wifi_raw;

-- 3. Populate ssid (one row per unique bssid)
INSERT OR IGNORE INTO ssid (bssid, ssid)
SELECT DISTINCT bssid, ssid
FROM wifi_raw;

-- 4. Populate wifi_scan
INSERT INTO wifi_scan (scan_id, bssid, rssi, noise, channel)
SELECT scan_id, bssid, rssi, noise, channel
FROM wifi_raw;

-- 5. Drop the old table
DROP TABLE wifi_raw;

COMMIT;