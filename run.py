import time
import subprocess

LOCATION = "(S)7.03"
ORIENTATION = "170 S"

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

    print("Sleeping for 5 seconds...\n")
    time.sleep(5)
print("YAYYYYYYY")
