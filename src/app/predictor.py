"""
predictor.py — Loads trained WiFi fingerprint models and predicts room from RSSI scans. Supports multiple models with runtime switching. Falls back to demo mode if no models are available.

Usage:
    1. Initialize predictor in app.py:
    from predictor import Predictor
    predictor = Predictor(models_dir="path/to/models")
    2. Get predictions:
    fingerprint = {"bssid1": -45, "bssid2": -80, ...}
    result = predictor.predict(fingerprint)
    result will be a dict with keys: room, confidence, model_used, position_x, position_y, aps_detected, mode
    3. Switch active model at runtime:
    predictor.set_active_model("Random Forest")  
"""

import joblib
import numpy as np
import json
import os
import random

import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

class Predictor:
    """
    Loads trained WiFi fingerprint models and predicts room from RSSI scans.
    Supports multiple models with runtime switching.
    Falls back to demo mode if no models are available.
    """

    def __init__(self, models_dir=MODELS_DIR):
        """
        Initialize the Predictor by loading models and metadata from the specified directory.

        Args:
            models_dir: Directory containing trained .pkl models and metadata files (label_encoder.pkl, scaler.pkl, feature_names.json, room_positions.json). 
                        If None or not found, runs in demo mode.
        """
        self.models = {}           # name -> trained model
        self.active_model = None   # currently selected model name
        self.label_encoder = None  # decodes predictions to room names
        self.scaler = None         # normalises RSSI values
        self.feature_names = []    # ordered list of BSSIDs
        self.room_positions = {}   # room name -> {x, y} on floor plan
        self.models_dir = models_dir

        if models_dir and os.path.exists(models_dir):
            self._load_all(models_dir)

    def _load_all(self, models_dir):
        """
        Load all models and metadata from the models directory.

        Args:
            models_dir: Directory containing trained .pkl models and metadata files.
        """

        # Load metadata first
        le_path = os.path.join(models_dir, 'label_encoder.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        features_path = os.path.join(models_dir, 'feature_names.json')
        positions_path = os.path.join(models_dir, 'room_positions.json')

        if os.path.exists(le_path):
            self.label_encoder = joblib.load(le_path)
            print(f"[Predictor] Label encoder loaded with classes: {self.label_encoder.classes_}")
            
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"[Predictor] Scaler loaded")

        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"[Predictor] Feature names loaded — {len(self.feature_names)} APs")

        if os.path.exists(positions_path):
            with open(positions_path, 'r') as f:
                self.room_positions = json.load(f)
            print(f"[Predictor] Room positions loaded — {len(self.room_positions)} rooms")

        # Load all .pkl model files
        model_files = {
            'weighted_knn': 'Weighted KNN',
            'random_forest': 'Random Forest',
            'svm': 'SVM',
            'mlp': 'MLP',
        }

        for filename, display_name in model_files.items():
            model_path = os.path.join(models_dir, f'{filename}.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[display_name] = joblib.load(model_path)
                    print(f"[Predictor] Loaded {display_name} from {filename}.pkl")
                except Exception as e:
                    print(f"[Predictor] Failed to load {display_name}: {e}")

        # Set default active model
        if self.models:
            # Random Forest
            if 'Random Forest' in self.models:
                self.active_model = 'Random Forest'
            else:
                self.active_model = list(self.models.keys())[0]
            print(f"[Predictor] Active model: {self.active_model}")
        else:
            print("[Predictor] No models loaded — running in demo mode")

    def get_available_models(self):
        """
        Return list of loaded model names.

        Returns:
            List of model names available for prediction.
        """
        return list(self.models.keys())

    def set_active_model(self, model_name):
        """
        Switch the active model.
        
        Args:            
            model_name: Name of the model to activate (must be in loaded models)

        Returns:
            bool: True if model switched successfully, False if model_name not found

        """
        if model_name in self.models:
            self.active_model = model_name
            print(f"[Predictor] Switched to {model_name}")
            return True
        return False

    def predict(self, fingerprint):
        """
        Predict room from a WiFi fingerprint.

        Args:
            fingerprint: dict of {bssid: rssi} values from a WiFi scan

        Returns:
            dict with room, confidence, model_used, and position
        """
        if self.active_model and self.active_model in self.models:
            return self._model_predict(fingerprint)
        else:
            return self._demo_predict()

    def _model_predict(self, fingerprint):
        """
        Run prediction using the active trained model.

        Args:
            fingerprint: dict of {bssid: rssi} values from a WiFi scan
        """
        model = self.models[self.active_model]

        # 1. Convert fingerprint to feature vector
        vector = self._fingerprint_to_vector(fingerprint)

        # 2. Normalise using the saved scaler (use DataFrame to preserve feature names)
        if self.scaler is not None:
            df = pd.DataFrame([vector], columns=self.feature_names)
            vector = self.scaler.transform(df)[0]

        # 3. Predict
        prediction = model.predict([vector])[0]

        # 4. Get confidence (probability of predicted class)
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba([vector])[0]
                confidence = float(max(proba))
            except Exception:
                confidence = 1.0
        else:
            confidence = 1.0

        # 5. Decode label
        room = "Unknown"
        if self.label_encoder is not None:
            room = self.label_encoder.inverse_transform([prediction])[0]

        # 6. Get floor plan position for this room
        position = self.room_positions.get(room, {"x": 0, "y": 0})

        return {
            "room": room,
            "confidence": round(confidence, 4),
            "model_used": self.active_model,
            "position_x": position.get("x", 0),
            "position_y": position.get("y", 0),
            "aps_detected": len(fingerprint),
            "mode": "live"
        }

    def _demo_predict(self):
        """
        Demo mode — returns simulated predictions for testing the UI.
        """
        rooms = ["(S) 7.01", "(S) 7.02", "(S) 7.03", "(S) 7.04", "(S) 7.05", "(S) 7.06"]
        room = random.choice(rooms)
        position = self.room_positions.get(room, {"x": 0, "y": 0})

        return {
            "room": room,
            "confidence": round(random.uniform(0.85, 0.99), 4),
            "model_used": "Demo Mode",
            "position_x": position.get("x", 0),
            "position_y": position.get("y", 0),
            "aps_detected": 0,
            "mode": "demo"
        }

    def _fingerprint_to_vector(self, fingerprint):
        """
        Convert a {bssid: rssi} dict to the fixed-length feature vector
        the model expects. Missing APs get -100.

        Args:
            fingerprint: dict of {bssid: rssi} values from a WiFi scan
        
        Returns:
            np.array of RSSI values ordered according to self.feature_names
        """
        vector = []
        for ap in self.feature_names:
            vector.append(fingerprint.get(ap, -100))
        return np.array(vector, dtype=float)