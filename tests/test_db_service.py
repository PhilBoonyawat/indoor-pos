"""Tests for src/data_collection/db_service.py"""
import sqlite3
import uuid

import pytest

from src.data_collection.db_service import init_db, store_raw_scan

@pytest.fixture
def conn(tmp_path):
    """
    Fresh in-file database connection per test (closed on teardown).

    Args:
        tmp_path: pytest fixture providing a temporary directory unique to the test.
    """
    db_path = tmp_path / "test.db"
    connection = init_db(db_path)
    yield connection
    connection.close()


class TestInitDB:
    """Verifies schema creation and connection setup."""

    def test_creates_all_tables(self, conn):
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert {"scan_metadata", "ssid", "wifi_scan"}.issubset(table_names)

    def test_is_idempotent(self, tmp_path):
        """Calling init_db on an existing database must not raise or duplicate."""
        db_path = tmp_path / "test.db"
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        count = conn2.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='scan_metadata'"
        ).fetchone()[0]
        assert count == 1
        conn2.close()

    def test_returns_sqlite_connection(self, conn):
        assert isinstance(conn, sqlite3.Connection)

    def test_foreign_keys_enabled(self, conn):
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_scan_metadata_has_expected_columns(self, conn):
        cols = [row[1] for row in conn.execute("PRAGMA table_info(scan_metadata)")]
        assert set(cols) == {
            "scan_id", "timestamp", "location",
            "latitude", "longitude", "orientation",
        }

    def test_wifi_scan_autoincrement_id(self, conn, make_scan_data):
        store_raw_scan(conn, make_scan_data(n_aps=2))
        store_raw_scan(conn, make_scan_data(n_aps=2))
        ids = [r[0] for r in conn.execute("SELECT id FROM wifi_scan")]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)


class TestStoreRawScan:
    """Verifies scan_data persistence into the three tables."""

    def test_stores_scan_metadata(self, conn, make_scan_data):
        scan_id = store_raw_scan(conn, make_scan_data())
        row = conn.execute(
            "SELECT location, orientation, latitude, longitude "
            "FROM scan_metadata WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()
        assert row == ("(S)7.01", "N", 51.511, -0.116)

    def test_stores_timestamp(self, conn, make_scan_data):
        scan_id = store_raw_scan(conn, make_scan_data(timestamp="2024-06-15T14:30:00"))
        ts = conn.execute(
            "SELECT timestamp FROM scan_metadata WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()[0]
        assert ts == "2024-06-15T14:30:00"

    def test_stores_correct_number_of_readings(self, conn, make_scan_data):
        scan_id = store_raw_scan(conn, make_scan_data(n_aps=7))
        count = conn.execute(
            "SELECT COUNT(*) FROM wifi_scan WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()[0]
        assert count == 7

    def test_stores_ssid_mapping(self, conn, make_scan_data):
        store_raw_scan(conn, make_scan_data(n_aps=3))
        ssids = conn.execute("SELECT COUNT(*) FROM ssid").fetchone()[0]
        assert ssids == 3

    def test_rssi_values_persisted_correctly(self, conn, make_scan_data):
        scan_id = store_raw_scan(conn, make_scan_data(n_aps=3))
        rssi_values = [
            r[0] for r in conn.execute(
                "SELECT rssi FROM wifi_scan WHERE scan_id = ? ORDER BY id",
                (scan_id,),
            )
        ]
        # make_scan_data sets rssi = -60 + i
        assert rssi_values == [-60, -59, -58]

    def test_duplicate_bssid_ignored_in_ssid_table(self, conn, make_scan_data):
        """Same BSSID across scans -> one ssid row, but two wifi_scan rows."""
        store_raw_scan(conn, make_scan_data(n_aps=3))
        store_raw_scan(conn, make_scan_data(n_aps=3))
        ssids = conn.execute("SELECT COUNT(*) FROM ssid").fetchone()[0]
        wifi_rows = conn.execute("SELECT COUNT(*) FROM wifi_scan").fetchone()[0]
        assert ssids == 3
        assert wifi_rows == 6

    def test_returns_valid_uuid(self, conn, make_scan_data):
        scan_id = store_raw_scan(conn, make_scan_data())
        parsed = uuid.UUID(scan_id)
        assert str(parsed) == scan_id

    def test_multiple_scans_get_unique_ids(self, conn, make_scan_data):
        ids = {store_raw_scan(conn, make_scan_data()) for _ in range(5)}
        assert len(ids) == 5

    def test_empty_scan_data_raises(self, conn):
        with pytest.raises(ValueError, match="No scan data"):
            store_raw_scan(conn, [])

    def test_different_locations_stored_separately(self, conn, make_scan_data):
        store_raw_scan(conn, make_scan_data(location="(S)7.01"))
        store_raw_scan(conn, make_scan_data(location="(S)7.02"))
        locations = [
            r[0] for r in conn.execute("SELECT location FROM scan_metadata")
        ]
        assert set(locations) == {"(S)7.01", "(S)7.02"}