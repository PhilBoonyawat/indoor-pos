"""
model_loader.py — Load model configurations from JSON and create scikit-learn model instances, and save best parameters back to JSON after tuning.

Usage:
1. Load models:
   from model_loader import load_models_from_config
   models = load_models_from_config()  # loads from default path
2. Save best parameters after tuning:
   from model_loader import save_best_params
   save_best_params(grid_results)  # grid_results is dict of {model_name: {best_params: {...}, ...}}    
"""


import json
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODEL_CLASSES = {
    "KNeighborsClassifier": KNeighborsClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "MLPClassifier": MLPClassifier,
}

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'model_configs.json'
)


def load_models_from_config(config_path=None):
    """
    Load model configurations from JSON and create scikit-learn model instances.

    Args:
        config_path: path to model_configs.json 

    Returns:
        dict of {model_name: (model_instance, description_string)}
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, 'r') as f:
        configs = json.load(f)

    models = {}
    for name, config in configs.items():
        model_class_name = config["model"]
        params = config["params"]

        # Convert lists back to tuples for hidden_layer_sizes
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        model_class = MODEL_CLASSES.get(model_class_name)
        if model_class is None:
            print(f"[Config] Unknown model class: {model_class_name}, skipping {name}")
            continue

        model = model_class(**params)

        # Build description string from params
        key_params = {k: v for k, v in params.items()
                      if k not in ['random_state', 'n_jobs', 'probability', 'verbose']}
        model_description = ", ".join(f"{k}={v}" for k, v in key_params.items())

        models[name] = (model, model_description)
        print(f"[Config] Loaded {name}: {model_description}")

    return models


def save_best_params(grid_results, config_path=None):
    """
    Save best parameters from GridSearchCV results back to model_configs.json.

    Args:
        grid_results: dict of {model_name: {best_params: {...}, ...}}
        config_path: path to model_configs.json
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = os.path.abspath(config_path)

    # Load existing config to preserve model class names and non-tuned params
    with open(config_path, 'r') as f:
        configs = json.load(f)

    for name, result in grid_results.items():
        if name not in configs:
            print(f"[Config] Warning: {name} not in config file, skipping")
            continue

        best_params = result["best_params"]

        # Convert tuples to lists for JSON serialisation
        for k, v in best_params.items():
            if isinstance(v, tuple):
                best_params[k] = list(v)

        # Update only the tuned params, keep the rest
        configs[name]["params"].update(best_params)
        print(f"[Config] Updated {name}: {best_params}")

    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)

    print(f"[Config] Saved updated config to {config_path}")