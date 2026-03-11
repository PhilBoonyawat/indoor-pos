"""
Indoor Position Tracker — Entry Point

Usage:
    python run.py                                    # Demo mode
    python run.py --models-dir models/               # With trained models
    python run.py --models-dir models/ --port 5000   # Custom port

Open http://localhost:5000 in your browser.
API docs at http://localhost:5000/docs
"""

import argparse
import sys
import os
import webbrowser
import uvicorn


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