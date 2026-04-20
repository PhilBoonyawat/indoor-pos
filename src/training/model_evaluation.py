"""
model_evaluation.py — Additional evaluation for deeper analysis.

Provides:
    - Feature importance analysis (which APs matter most)
    - Learning curves (accuracy vs training set size)
    - Misclassification analysis (which rooms get confused)

Usage:
    python model_evaluation.py
    python model_evaluation.py --db ../../data/raw/wifi_scans.db --models-dir ../../models/
"""

import argparse
import os
import json
import sqlite3
from contextlib import closing
import numpy as np
import joblib

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocess import load_and_preprocess
from model_loader import load_models_from_config

import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('pdf')
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-paper")

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})


DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)

MODEL_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'models'
)

FIG_OUTPUT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'assets', 'evaluation'
)

MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'model_configs.json'
)


def load_bssid_to_ssid_map(db_path=DATABASE_PATH):
    """
    Load mapping of BSSID to SSID from the database for better feature labeling.

    Args:
        db_path: path to SQLite database containing 'ssid' table

    Returns:
        dict of {bssid: ssid} for all APs in the database
    """
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute("SELECT bssid, ssid FROM ssid")
        mapping = {}
        for bssid, ssid in cursor.fetchall():
            mapping[bssid] = ssid if ssid else "Hidden"
    return mapping


def format_ap_label(bssid, bssid_to_ssid):
    """
    Format AP label as 'SSID (last 8 of BSSID)' for readability.

    Args:
        bssid: full BSSID string (e.g. "AA:BB:CC:DD:EE:FF")
        bssid_to_ssid: dict mapping BSSID to SSID

    Returns:
        formatted label string (e.g. "eduroam (DD:EE:FF)")
    """
    ssid = bssid_to_ssid.get(bssid, "Unknown")
    short_bssid = bssid[-8:]
    return f"{ssid} ({short_bssid})"


def load_trained_models(models_dir=MODEL_DIRECTORY):
    """
    Load all trained model .pkl files from the models directory.

    Args:
        models_dir: directory containing trained model .pkl files

    Returns:
        dict of {display_name: trained_model_object}
    """
    model_files = {
        'weighted_knn': 'Weighted KNN',
        'random_forest': 'Random Forest',
        'svm': 'SVM',
        'mlp': 'MLP',
    }

    models = {}
    for filename, display_name in model_files.items():
        model_path = os.path.join(models_dir, f'{filename}.pkl')
        if os.path.exists(model_path):
            models[display_name] = joblib.load(model_path)
            print(f"[Evaluate] Loaded {display_name} from {filename}.pkl")
        else:
            print(f"[Evaluate] Warning: {model_path} not found, skipping {display_name}")

    return models


def evaluate_models(models, X_test, y_test, label_encoder):
    """
    Run predictions on the test set for each model and collect results.

    Args:
        models: dict of {name: trained_model}
        X_test: test feature matrix
        y_test: test labels (encoded)
        label_encoder: fitted LabelEncoder for decoding predictions

    Returns:
        dict of {name: {y_pred, accuracy, confusion_matrix, report_str}}
    """
    results = {}

    print("\n" + "=" * 60)
    print("  Model Evaluation on Test Set")
    print("=" * 60)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_str = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            digits=4, zero_division=0
        )

        print(f"\n  {name} — Accuracy: {acc:.4f}")
        for line in report_str.split('\n'):
            print(f"    {line}")

        results[name] = {
            'y_pred': y_pred,
            'accuracy': acc,
            'confusion_matrix': cm,
            'report_str': report_str,
        }

    return results

def feature_importance_analysis(feature_names, models_dir=MODEL_DIRECTORY,
                                output_dir=FIG_OUTPUT_DIRECTORY,
                                db_path=DATABASE_PATH, top_n=20):
    """
    Analyze feature importance from the Random Forest model and plot the top N features.

    Args:
        feature_names: list of feature names (BSSIDs) corresponding to model input
        models_dir: directory where trained model files are stored
        output_dir: directory to save the plot
        db_path: path to database for BSSID -> SSID mapping
        top_n: number of top features to display
    """
    os.makedirs(output_dir, exist_ok=True)

    bssid_to_ssid = {}
    if db_path and os.path.exists(db_path):
        bssid_to_ssid = load_bssid_to_ssid_map(db_path)
        print(f"[Evaluate] Loaded {len(bssid_to_ssid)} BSSID -> SSID mappings")

    model_path = os.path.join(models_dir, 'random_forest.pkl')
    if not os.path.exists(model_path):
        print("[Evaluate] Random Forest model not found, skipping feature importance.")
        return

    model = joblib.load(model_path)
    importances = model.feature_importances_

    indices = np.argsort(importances)[-top_n:]
    top_importances = importances[indices]

    if bssid_to_ssid:
        top_features = [format_ap_label(feature_names[i], bssid_to_ssid) for i in indices]
    else:
        top_features = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(top_n), top_importances, color='#3b82f6', alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=6)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest — Top Access Points for Room Classification', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.pdf'), bbox_inches="tight")
    plt.close()
    print("[Evaluate] Saved feature_importance.pdf")


def learning_curve_analysis(X_train, y_train, output_dir=FIG_OUTPUT_DIRECTORY):
    """
    Compute and plot learning curves to assess how performance scales with training data.

    Args:
        X_train: training feature matrix
        y_train: training labels
        output_dir: directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    model_configs = load_models_from_config(MODEL_CONFIG_PATH)

    num_models = len(model_configs)
    cols = 2
    rows = (num_models + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6, 2.5 * rows))
    axes = axes.flatten()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ax, (name, (model, _)) in zip(axes, model_configs.items()):
        print(f"[Evaluate] Computing learning curve for {name}...")
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring='f1_weighted', n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color='#3b82f6')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.1, color='#10b981')
        ax.plot(train_sizes, train_mean, 'o-', color='#3b82f6', label='Training')
        ax.plot(train_sizes, val_mean, 'o-', color='#10b981', label='Validation')

        ax.set_xlabel('Training Samples')
        ax.set_ylabel('F1 Score (weighted)')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.02)

    for i in range(num_models, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Learning Curves — Would More Data Help?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.pdf'), dpi=150)
    plt.close()
    print("[Evaluate] Saved learning_curves.pdf")


def misclassification_analysis(results, label_encoder, output_dir=FIG_OUTPUT_DIRECTORY):
    """
    Analyse which rooms get confused with each other most often.
    Generates a single grouped bar chart of the top confusion pairs across all models.

    Args:
        results: dict from evaluate_models() containing confusion matrices
        label_encoder: fitted LabelEncoder with room class names
        output_dir: directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Misclassification Analysis")
    print("=" * 60)

    classes = label_encoder.classes_
    model_names = list(results.keys())

    # Collect all confusion pairs from all models
    all_confusions = {}
    for name in model_names:
        cm = results[name]['confusion_matrix']
        total = cm.sum()
        errors = total - np.trace(cm)
        print(f"  {name}: {int(errors)}/{int(total)} errors ({errors / total * 100:.2f}%)")

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i][j] > 0:
                    pair = f"{classes[i]}\n-> {classes[j]}"
                    if pair not in all_confusions:
                        all_confusions[pair] = {m: 0 for m in model_names}
                    all_confusions[pair][name] = cm[i][j]

    if not all_confusions:
        print("  No misclassifications found.")
        return

    # Sort by total count, take top pairs
    sorted_pairs = sorted(
        all_confusions.keys(),
        key=lambda p: sum(all_confusions[p].values()),
        reverse=True
    )[:6]

    colors = [
        "#4C72B0",  
        "#DD8452", 
        "#55A868", 
        "#C44E52"  
    ]

    hatches = ['', '///', '...', 'xxx']

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(sorted_pairs))
    width = 0.75 / len(model_names)

    for i, name in enumerate(model_names):
        counts = [all_confusions[pair][name] for pair in sorted_pairs]
        offset = (i - len(model_names) / 2 + 0.5) * width

        ax.bar(
            x + offset,
            counts,
            width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.5,
            hatch=hatches[i % len(hatches)]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_pairs, fontsize=7)
    ax.set_ylabel('Count')
    ax.set_title('Most Frequent Misclassification Pairs by Model', fontweight='bold')

    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'misclassification_pairs.pdf'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()

    print("[Evaluate] Saved misclassification_pairs.pdf")

def parse_args():
    """
    Parse command-line arguments for evaluation script.

    Returns:
        argparse.Namespace with attributes:
            db: path to SQLite database (default: ../../data/raw/wifi_scans.db)
            models_dir: directory containing trained model .pkl files (default: ../../models/)
            output: directory to save evaluation outputs (default: ../../assets/evaluation/)    
    """
    
    parser = argparse.ArgumentParser(description="Additional model evaluation")
    parser.add_argument('--db', type=str, default=DATABASE_PATH,
                        help='Path to SQLite database')
    parser.add_argument('--models-dir', type=str, default=MODEL_DIRECTORY,
                        help='Directory containing trained .pkl models')
    parser.add_argument('--output', type=str, default=FIG_OUTPUT_DIRECTORY,
                        help='Directory to save evaluation outputs')
    return parser.parse_args()

def main():
    """
    Main function to run the evaluation pipeline:
        1. Load and preprocess test data from SQLite database
        2. Load trained models from .pkl files      
        3. Evaluate each model on the test set and print classification reports
        4. Perform feature importance analysis for Random Forest and plot top APs
        5. Compute and plot learning curves for all models
        6. Analyse misclassifications to see which rooms get confused
        7. Save all outputs (plots, reports) to the specified output directory
    """
    
    args = parse_args()

    db_path = args.db
    models_dir = args.models_dir
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = load_and_preprocess(db_path)

    # 2. Load trained models
    models = load_trained_models(models_dir)

    if not models:
        print("[Evaluate] No trained models found. Run train.py first.")
        return

    # 3. Evaluate models on test set
    results = evaluate_models(models, X_test, y_test, label_encoder)

    # 4. Feature importance (Random Forest)
    feature_importance_analysis(feature_names, models_dir, output_dir, db_path=db_path)

    # 5. Learning curves
    learning_curve_analysis(X_train, y_train, output_dir)

    # 6. Misclassification analysis
    misclassification_analysis(results, label_encoder, output_dir)

    print("\n" + "=" * 60)
    print("  Evaluation Complete")
    print("=" * 60)
    print(f"\n  Models evaluated: {', '.join(results.keys())}")
    for name, data in results.items():
        print(f"    {name}: {data['accuracy']:.4f} accuracy")
    print(f"\n  Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()