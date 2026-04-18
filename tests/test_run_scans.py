"""Tests for src/data_collection/run_scans.py"""

from unittest.mock import patch

import pytest

from src.data_collection.run_scans import parse_args, run_scans


class TestParseArgs:
    def test_required_args(self):
        with patch("sys.argv", ["prog", "-l", "(S) 7.01", "-f", "N"]):
            args = parse_args()
        assert args.location == "(S) 7.01"
        assert args.orientation == "N"

    def test_defaults_for_optional_args(self):
        with patch("sys.argv", ["prog", "-l", "X", "-f", "Y"]):
            args = parse_args()
        assert args.num_scans == 20
        assert args.interval == 3

    def test_custom_num_scans_short(self):
        with patch("sys.argv", ["prog", "-l", "X", "-f", "Y", "-n", "5"]):
            args = parse_args()
        assert args.num_scans == 5

    def test_custom_num_scans_long(self):
        with patch("sys.argv", ["prog", "-l", "X", "-f", "Y", "--num-scans", "7"]):
            args = parse_args()
        assert args.num_scans == 7

    def test_custom_interval_short(self):
        with patch("sys.argv", ["prog", "-l", "X", "-f", "Y", "-i", "10"]):
            args = parse_args()
        assert args.interval == 10

    def test_missing_required_args_exits(self):
        with patch("sys.argv", ["prog"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_missing_location_exits(self):
        with patch("sys.argv", ["prog", "-f", "N"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_missing_orientation_exits(self):
        with patch("sys.argv", ["prog", "-l", "X"]):
            with pytest.raises(SystemExit):
                parse_args()


class TestRunScans:
    def test_invokes_subprocess_once_per_scan(self):
        with patch("src.data_collection.run_scans.subprocess.run") as mock_run, \
             patch("src.data_collection.run_scans.time.sleep"):
            run_scans("(S) 7.01", "N", num_scans=3, interval=1)
        assert mock_run.call_count == 3

    def test_passes_location_and_orientation_to_subprocess(self):
        with patch("src.data_collection.run_scans.subprocess.run") as mock_run, \
             patch("src.data_collection.run_scans.time.sleep"):
            run_scans("(S) 7.05", "196 S", num_scans=1, interval=1)
        call_args = mock_run.call_args[0][0]
        assert "(S) 7.05" in call_args
        assert "196 S" in call_args

    def test_sleeps_between_but_not_after_last(self):
        """3 scans → 2 sleeps; 1 scan → 0 sleeps."""
        with patch("src.data_collection.run_scans.subprocess.run"), \
             patch("src.data_collection.run_scans.time.sleep") as mock_sleep:
            run_scans("X", "Y", num_scans=3, interval=5)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(5)

    def test_single_scan_does_not_sleep(self):
        with patch("src.data_collection.run_scans.subprocess.run"), \
             patch("src.data_collection.run_scans.time.sleep") as mock_sleep:
            run_scans("X", "Y", num_scans=1, interval=5)
        assert mock_sleep.call_count == 0

    def test_zero_scans_calls_subprocess_zero_times(self):
        with patch("src.data_collection.run_scans.subprocess.run") as mock_run, \
             patch("src.data_collection.run_scans.time.sleep"):
            run_scans("X", "Y", num_scans=0, interval=1)
        assert mock_run.call_count == 0