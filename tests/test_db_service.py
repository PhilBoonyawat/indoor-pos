import sqlite3
import uuid
import pytest

from src.data_collection.db_service import init_db, store_raw_scan


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    connection = init_db(db_path)
    yield connection
    connection.close()


class TestDBService:
    def make_scan_data(self, n_aps=5, location="(S) 7.01"):
        return [
            {
                "ssid": f"Network-{i}",
                "bssid": f"aa:bb:cc:dd:ee:{i:02x}",
                "rssi": -60 + i,
                "noise": -95,
                "channel": 6,
                "timestamp": "2024-01-01T10:00:00",
                "location": location,
                "latitude": 51.511,
                "longitude": -0.116,
                "orientation": "75 N",
            }
            for i in range(n_aps)
        ]

    def test_creates_tables(self, conn):
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}

        assert "scan_metadata" in table_names
        assert "ssid" in table_names
        assert "wifi_scan" in table_names

    def test_stores_scan_metadata(self, conn):
        scan_id = store_raw_scan(conn, self.make_scan_data())
        row = conn.execute(
            "SELECT scan_id, location, orientation FROM scan_metadata WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()

        assert row is not None
        assert row[1] == "(S) 7.01"
        assert row[2] == "75 N"

    def test_returns_connection(self, conn):
        assert isinstance(conn, sqlite3.Connection)

    def test_foreign_keys_enabled(self, conn):
        fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk_status == 1

    def test_stores_wifi_readings(self, conn):
        scan_id = store_raw_scan(conn, self.make_scan_data(n_aps=5))
        count = conn.execute(
            "SELECT COUNT(*) FROM wifi_scan WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()[0]
        assert count == 5

    def test_stores_ssid_mapping(self, conn):
        store_raw_scan(conn, self.make_scan_data(n_aps=3))
        ssids = conn.execute("SELECT COUNT(*) FROM ssid").fetchone()[0]
        assert ssids == 3

    def test_duplicate_bssid_ignored_in_ssid_table(self, conn):
        store_raw_scan(conn, self.make_scan_data(n_aps=3))
        store_raw_scan(conn, self.make_scan_data(n_aps=3))
        ssids = conn.execute("SELECT COUNT(*) FROM ssid").fetchone()[0]
        assert ssids == 3

    def test_returns_valid_uuid(self, conn):
        scan_id = store_raw_scan(conn, self.make_scan_data())
        parsed = uuid.UUID(scan_id)
        assert str(parsed) == scan_id

    def test_empty_scan_data_raises(self, conn):
        with pytest.raises(ValueError, match="No scan data"):
            store_raw_scan(conn, [])

    def test_multiple_scans_different_ids(self, conn):
        id1 = store_raw_scan(conn, self.make_scan_data())
        id2 = store_raw_scan(conn, self.make_scan_data())
        assert id1 != id2