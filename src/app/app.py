from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from predictor import Predictor
from scanner_thread import ScannerThread


MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'wifi_scans.db'))
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", "3"))

# ── App Setup ────────────────────────────────────────────────

app = FastAPI(
    title="Indoor Position Tracker",
    description="Wi-Fi fingerprint-based indoor positioning for Bush House Level 7",
    version="1.0.0",
)

# Initialise predictor with all trained models
predictor = Predictor(models_dir=MODELS_DIR)

# Start background scanner
scanner = ScannerThread(predictor=predictor, interval=SCAN_INTERVAL, db_path=DB_PATH)
scanner.start()


# ── Response Models ──────────────────────────────────────────

class Position(BaseModel):
    room: str
    confidence: float
    model_used: str
    position_x: int = 0
    position_y: int = 0
    aps_detected: int = 0
    mode: str = "demo"
    timestamp: Optional[str] = None


class Status(BaseModel):
    scanning: bool
    scan_mode: str
    active_model: Optional[str]
    available_models: list[str]
    scan_interval: int


class ModelSwitch(BaseModel):
    model_name: str


# ── API Routes ───────────────────────────────────────────────

@app.get("/api/position", response_model=Position)
async def get_position():
    """
    Returns the latest predicted position.

    Returns:
        Position: The predicted position with room, confidence, model used, coordinates, and mode.
    """
    return scanner.get_current_position()


@app.get("/api/scan-history", response_model=list[Position])
async def get_scan_history():
    """
    Returns recent scan history for trail visualization.

    Returns:
        List[Position]: A list of recent predicted positions with metadata.
    """
    return scanner.get_history(limit=20)


@app.get("/api/status", response_model=Status)
async def get_status():
    """
    Returns system status info.

    Returns:
        Status: Current scanning status, active model, available models, and scan interval.
    """
    return {
        "scanning": scanner.is_running(),
        "scan_mode": scanner.get_scan_mode(),
        "active_model": predictor.active_model,
        "available_models": predictor.get_available_models(),
        "scan_interval": scanner.interval,
    }


@app.post("/api/model")
async def switch_model(body: ModelSwitch):
    """
    Switch the active prediction model.

    Args:
        body: ModelSwitch object containing the name of the model to activate.

    Returns:
        dict: Status of the model switch operation, including the new active model or error message.
    """
    success = predictor.set_active_model(body.model_name)
    if success:
        return {"status": "ok", "active_model": predictor.active_model}
    else:
        return {"status": "error", "message": f"Model '{body.model_name}' not found",
                "available": predictor.get_available_models()}


@app.get("/api/models")
async def get_models():
    """
    Returns list of available models.

    Returns:
        dict: Active model and list of available models.
    """
    return {
        "active": predictor.active_model,
        "available": predictor.get_available_models()
    }


@app.get("/api/room-positions")
async def get_room_positions():
    """
    Returns room positions for floor plan mapping.

    Returns:
        dict: Mapping of room names to their (x, y) coordinates for visualization on the floor plan.
    """
    return predictor.room_positions


# ── Static Files & Frontend ─────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def index():
    """
    Serve the frontend.
    
    Returns:
        FileResponse: The index.html file for the frontend application.
    """
    return FileResponse(os.path.join(static_dir, "index.html"))