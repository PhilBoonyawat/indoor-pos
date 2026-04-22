# Indoor Positioning using Wi-Fi

A Wi-Fi fingerprint-based indoor positioning system for Bush House
Level 7 (S). Predicts the user's current room from live Wi-Fi scans
using one of four trained classifiers (Weighted KNN, Random Forest,
SVM, MLP), and displays the predicted position on an interactive
floor plan.

## Requirements

- macOS (tested on Sequoia 15.6)
- Python 3.14.0 or later
- Location Services permission for Python (see note below)

## Initial Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Positioning System

```bash
source venv/bin/activate
python3 run.py --models-dir models/ --port 8000
```

Then open `http://localhost:8000` in a browser. Interactive API docs are
served at `http://localhost:8000/docs`.

> **Note:** on first load, the backend may take a few seconds to start;
> a browser refresh is sometimes needed.

### macOS Location Permission

Live Wi-Fi scanning on macOS requires the Python interpreter to have
Location Services permission. On first run, CoreLocation will prompt
you to grant access. If the prompt does not appear, enable it manually:

**System Settings → Privacy & Security → Location Services → Python.**

If permission is not granted, the system will fall back to demo mode,
which replays stored fingerprints from the database.

## Collecting New Training Scans

To record a new batch of labelled fingerprints for a given room and
orientation:

```bash
source venv/bin/activate
python3 src/data_collection/run_scans.py -l "(S)7.03" -f N -n 20 -i 3
```

| Flag | Meaning | Default |
|---|---|---|
| `-l, --location`    | Room label (e.g. `(S)7.03`)      | required |
| `-f, --orientation` | Device orientation (`N`, `S`, etc.) | required |
| `-n, --num-scans`   | Number of scans to collect        | 20 |
| `-i, --interval`    | Seconds between scans             | 3 |

Scans are stored in `data/raw/wifi_scans.db`.

## Training Models

Run the four scripts in order. Each produces artefacts used by the
next.

```bash
python3 src/training/hyperparameter_tuning.py     
python3 src/training/train.py                      
python3 src/training/model_evaluation.py           
python3 src/training/data_efficiency_analysis.py   
```

## Room-Position Labeller

A browser-based tool for generating `room_positions.json` from a
floor-plan image. Click on the floor plan to mark each room's pixel
coordinates, then export as JSON.

```bash
source venv/bin/activate
cd src/app/static/room-labeller
python3 -m http.server 8001
```

Open `http://localhost:8001` in a browser and follow the on-screen
prompts. Move the exported JSON to `models/room_positions.json`.

## Running Tests

```bash
pytest --cov=src
```

The full suite comprises 268 tests and achieves 91.5% line-and-branch
coverage.

## Project Structure

```
src/
├── app/              # Online positioning system (FastAPI)
│   ├── app.py
│   ├── predictor.py
│   ├── scanner_thread.py
│   └── static/       # Frontend + room labeller
├── data_collection/  # Offline scan collection
└── training/         # Model training and evaluation
data/raw/             # SQLite database
models/               # Trained .pkl files + metadata
```