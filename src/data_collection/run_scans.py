"""
run_scans.py — A simple script to run repeated Wi-Fi scans with user-provided location and orientation. 
Designed to be called from the command line, it will invoke data_collector_service.py as a subprocess for each scan, 
passing along the location and orientation. It also handles timing between scans and provides user feedback.

Usage:
    python3 src/data_collection/run_scans.py -l "(S)7.05" -f "196 S" -n 20 -i 3
"""

import time
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    """
    Parse command-line arguments for location, orientation, number of scans, and interval between scans.

    Returns:
        Namespace with 'location', 'orientation', 'num_scans', and 'interval' attributes.
    """
    
    parser = argparse.ArgumentParser(description="Run repeated Wi-Fi scans")
    parser.add_argument("-l", "--location", required=True, help="Room locatio in the building (EX: (S)7.05)")
    parser.add_argument("-f", "--orientation", required=True, help="Orientation you are facing (EX: 196 S or - if not recording)")
    parser.add_argument("-n", "--num-scans", type=int, default=20, help="Number of scans")
    parser.add_argument("-i", "--interval", type=int, default=3, help="Seconds between scans")
    return parser.parse_args()

def run_scans(location, orientation, num_scans, interval):
    """
    Run the specified number of Wi-Fi scans with the given location and orientation, 
    invoking data_collector_service.py as a subprocess for each scan.

    Subprocess is used to ensure that each scan is a fresh invocation of the data collection logic, 
    which can help with memory management and also allows for better isolation of each scan.

    Args:
        location: String representing the room/location in the building (EX. "(S)7.05")
        orientation: String representing the direction the user is facing (EX. "196 S")
        num_scans: Integer number of scans to perform
        interval: Integer number of seconds to wait between scans
    """
    for i in range(num_scans):
        print(f"Running Wi-Fi scan {i + 1}/{num_scans}")

        subprocess.run(
            [
                "python3",
                "-m",
                "data_collection.data_collector_service",
                "-l",
                location,
                "-f",
                orientation
            ],
            check=False,
            cwd=str(PROJECT_ROOT / "src")
        )

        print("Scan complete")

        if i < num_scans - 1:
            print(f"Sleeping for {interval} seconds...\n")
            time.sleep(interval)

    print("All scans completed.")

def main():
    """
    Main function to parse arguments and run the scans.
    """

    args = parse_args()
    run_scans(args.location, args.orientation, args.num_scans, args.interval)

if __name__ == "__main__":
    main()
