import time
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated Wi-Fi scans")
    parser.add_argument("-l", "--location", required=True, help="Room locatio in the building (EX: (S)7.05)")
    parser.add_argument("-f", "--orientation", required=True, help="Orientation you are facing")
    parser.add_argument("-n", "--num-scans", type=int, default=20, help="Number of scans")
    parser.add_argument("-i", "--interval", type=int, default=3, help="Seconds between scans")
    return parser.parse_args()

def run_scans(location, orientation, num_scans, interval):

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
    args = parse_args()
    run_scans(args.location, args.orientation, args.num_scans, args.interval)

if __name__ == "__main__":
    main()
