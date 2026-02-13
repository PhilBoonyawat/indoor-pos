import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCANNER_PATH = PROJECT_ROOT / "data_collection" / "data_collector_service.py"

LOCATION = "(S)7.03"
ORIENTATION = "162 S"

i = 0
while i < 10:
    print("Running Wi-Fi scan...")
    print(i)
    subprocess.run(
        [
            "python3",
            str(SCANNER_PATH),
            "-l", LOCATION,
            "-f", ORIENTATION
        ],
        check=False
    )
    i += 1
    print("done")

    print("Sleeping for 3 seconds...\n")
    time.sleep(3)
print("YAYYYYYYY")
