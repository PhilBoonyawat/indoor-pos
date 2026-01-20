import time
import subprocess

LOCATION = "(S)7.05"
ORIENTATION = "43 NE"

i = 0
while i < 10:
    print("Running Wi-Fi scan...")
    print(i)
    subprocess.run(
        [
            "python3",
            "data_collector_service.py",
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
