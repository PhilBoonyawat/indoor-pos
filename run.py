"""
Indoor Position Tracker — Entry Point

Usage:
    python run.py                                    # Demo mode
    python run.py --models-dir models/               # With trained models
    python run.py --models-dir models/ --port 5000   # Custom port
    python run.py --skip-location                    # Skip location permission check

Open http://localhost:5000 in your browser.
API docs at http://localhost:5000/docs
"""

import argparse
import sys
import os
import webbrowser
import uvicorn


def preflight_location_permission():
    """
    On macOS, ensure Python is registered with Location Services and that
    permission has been granted. In a CLI context the system dialog often
    doesn't appear, so we:

    1. Import CoreLocation to trigger registration (makes Python appear in
       System Settings > Privacy & Security > Location Services).
    2. Check the current status.
    3. If not yet granted, print step-by-step instructions and wait for
       the user to toggle it on in System Settings.
    """
    try:
        from src.data_collection.location_service import (
            check_location_permission,
            trigger_permission_prompt,
            wait_for_location_permission,
        )
    except ImportError:
        print("[Preflight] Could not import location_service (pyobjc not installed?).")
        print("[Preflight] Continuing without location support.\n")
        return True

    status = check_location_permission()

    if status == "granted":
        print("[Preflight] Location permission already granted.\n")
        return True

    # Not yet determined — trigger the request so Python appears in System Settings
    if status == "not_determined":
        status = trigger_permission_prompt()
        if status == "granted":
            print("[Preflight] Location permission granted.\n")
            return True

    # Still not granted — guide the user through manual setup
    print()
    print("=" * 60)
    print("  LOCATION PERMISSION REQUIRED")
    print("=" * 60)
    print()
    print("  Python needs Location Services access for WiFi scanning.")
    print("  Since this is a command-line app, macOS may not show a")
    print("  permission dialog — you need to enable it manually:")
    print()
    print("  1. Open System Settings (Apple menu > System Settings)")
    print("  2. Go to Privacy & Security > Location Services")
    print("  3. Make sure Location Services is turned ON")
    print("  4. Find 'Python' (or your terminal app, e.g. 'Terminal',")
    print("     'iTerm') in the list and toggle it ON")
    print()
    print("=" * 60)
    print()

    answer = input("Waiting — press Enter once you've enabled it (or 'skip' to continue without location): ").strip().lower()

    if answer == "skip":
        print("[Preflight] Skipping — location data won't be available.\n")
        return True

    # Re-check after user says they've enabled it
    status = check_location_permission()
    if status == "granted":
        print("[Preflight] Location permission confirmed!\n")
        return True

    # Still not granted — poll briefly in case there's a delay
    print("[Preflight] Not detected yet — checking again for a few seconds...")
    if wait_for_location_permission(timeout=10, poll_interval=2):
        print("[Preflight] Location permission confirmed!\n")
        return True

    print("[Preflight] Location permission still not detected.")
    print("[Preflight] Continuing anyway — scans will work but without GPS coordinates.\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Indoor Position Tracker")
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory containing trained .pkl models (default: models/)"
    )
    parser.add_argument(
        "--db", type=str, default="data/raw/wifi_scans.db",
        help="Path to SQLite database for demo mode (default: data/raw/wifi_scans.db)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--interval", type=int, default=3,
        help="WiFi scan interval in seconds (default: 3)"
    )
    parser.add_argument(
        "--skip-location", action="store_true",
        help="Skip the location permission preflight check"
    )
    args = parser.parse_args()

    # Resolve paths relative to working directory
    models_dir = os.path.abspath(args.models_dir)
    db_path = os.path.abspath(args.db)

    # Set environment variables for app.py to pick up
    os.environ["MODELS_DIR"] = models_dir
    os.environ["DB_PATH"] = db_path
    os.environ["SCAN_INTERVAL"] = str(args.interval)

    # Add app directory to path
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "app")
    sys.path.insert(0, app_dir)

    # Location permission check (macOS only)
    if not args.skip_location and sys.platform == "darwin":
        preflight_location_permission()

    print("=" * 55)
    print("  Indoor Position Tracker — Bush House Level 7")
    print("=" * 55)
    print(f"  Models:    {models_dir}")
    print(f"  Database:  {db_path}")
    print(f"  Interval:  {args.interval}s")
    print(f"  App:       http://localhost:{args.port}")
    print(f"  API Docs:  http://localhost:{args.port}/docs")
    print("=" * 55)

    # Open browser after short delay
    import threading
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    # Start server
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()