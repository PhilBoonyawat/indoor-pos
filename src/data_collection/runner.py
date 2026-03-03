import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCATION = "(S)7.04"
ORIENTATION = "164 S"

for i in range(50):
    print(f"Running Wi-Fi scan...\n{i}")
    subprocess.run(
        [
            "python3", "-m", "data_collection.data_collector_service",
            "-l", LOCATION,
            "-f", ORIENTATION
        ],
        check=False,
        cwd=str(PROJECT_ROOT / "src")
    )
    print("done")
    print("Sleeping for 3 seconds...\n")
    time.sleep(3)

print("YAYYYYYYY")