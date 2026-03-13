"""
evaluate.py — Additional evaluation utilities for deeper analysis.

Provides:
    - Per-room accuracy breakdown
    - Feature importance analysis (which APs matter most)
    - Learning curve (accuracy vs training set size)
    - Misclassification analysis

Usage:
    python evaluate.py --db path/to/db --models-dir path/to/models/
"""

import argparse
import os
import json
import sqlite3
import numpy as np
import joblib

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from preprocess import load_and_preprocess


def load_bssid_to_ssid_map(db_path):
    """Load BSSID → SSID mapping from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT bssid, ssid FROM ssid")
    mapping = {}
    for bssid, ssid in cursor.fetchall():
        mapping[bssid] = ssid if ssid else "Hidden"
    conn.close()
    return mapping


def format_ap_label(bssid, bssid_to_ssid):
    """Format AP label as 'SSID (last 8 of BSSID)' for readability."""
    ssid = bssid_to_ssid.get(bssid, "Unknown")
    short_bssid = bssid[-8:]  # last 8 chars e.g. "DD:EE:01"
    return f"{ssid} ({short_bssid})"


def feature_importance_analysis(models_dir, feature_names, output_dir, db_path=None, top_n=20):
    """
    Extract and plot feature importance from Random Forest.
    Shows which access points are most useful for room discrimination.
    Maps BSSIDs back to SSIDs for readable labels.
    """
    if not HAS_PLOTTING:
        return

    # Load BSSID → SSID mapping if database provided
    bssid_to_ssid = {}
    if db_path and os.path.exists(db_path):
        bssid_to_ssid = load_bssid_to_ssid_map(db_path)
        print(f"[Evaluate] Loaded {len(bssid_to_ssid)} BSSID→SSID mappings")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, model_name in zip(axes, ['random_forest']):
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            ax.set_title(f'{model_name} — not found')
            continue

        model = joblib.load(model_path)
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        # Map BSSIDs to readable labels
        if bssid_to_ssid:
            top_features = [format_ap_label(feature_names[i], bssid_to_ssid) for i in indices]
        else:
            top_features = [feature_names[i] for i in indices]

        top_importances = importances[indices]

        ax.barh(range(top_n), top_importances, color='#3b82f6', alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features, fontsize=7)
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name.replace("_", " ").title()} — Top {top_n} APs')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Feature Importance — Most Discriminative Access Points', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    print(f"[Evaluate] Saved feature_importance.png")


def learning_curve_analysis(X_train, y_train, output_dir):
    """
    Plot learning curves to see if more data would help.
    Shows accuracy vs number of training samples.
    """
    if not HAS_PLOTTING:
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    models = {
        'Weighted KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                             max_iter=500, early_stopping=True, random_state=42),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    colors = {
        'Weighted KNN': ('#3b82f6', '#10b981'),
        'Random Forest': ('#3b82f6', '#10b981'),
        'SVM': ('#3b82f6', '#10b981'),
        'MLP': ('#3b82f6', '#10b981'),
    }

    for ax, (name, model) in zip(axes, models.items()):
        print(f"[Evaluate] Computing learning curve for {name}...")
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring='accuracy', n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3b82f6')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#10b981')
        ax.plot(train_sizes, train_mean, 'o-', color='#3b82f6', label='Training')
        ax.plot(train_sizes, val_mean, 'o-', color='#10b981', label='Validation')

        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

    plt.suptitle('Learning Curves — Would More Data Help?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150)
    plt.close()
    print(f"[Evaluate] Saved learning_curves.png")


def misclassification_analysis(results, label_encoder, output_dir):
    """
    Analyse which rooms get confused with each other most often.
    Useful for understanding model limitations.
    """
    print("\n[Evaluate] Misclassification Analysis")
    print("=" * 50)

    report_lines = []

    for name, data in results.items():
        cm = data['confusion_matrix']
        classes = label_encoder.classes_
        misclassified = []

        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i][j] > 0:
                    misclassified.append({
                        'actual': classes[i],
                        'predicted': classes[j],
                        'count': cm[i][j]
                    })

        misclassified.sort(key=lambda x: x['count'], reverse=True)

        total_errors = sum(m['count'] for m in misclassified)
        total_samples = cm.sum()

        line = f"\n  {name}: {total_errors}/{total_samples} errors ({total_errors/total_samples*100:.1f}%)"
        print(line)
        report_lines.append(line)

        for m in misclassified[:5]:  # top 5 confusions
            line = f"    {m['actual']} → {m['predicted']}: {m['count']} times"
            print(line)
            report_lines.append(line)

    # Save to file
    with open(os.path.join(output_dir, 'misclassification_report.txt'), 'w') as f:
        f.write("Misclassification Analysis\n")
        f.write("=" * 50 + "\n")
        f.write('\n'.join(report_lines))

    print(f"\n[Evaluate] Saved misclassification_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Additional model evaluation")
    parser.add_argument('--db', type=str, default='../../data/raw/wifi_scans.db')
    parser.add_argument('--models-dir', type=str, default='../../models/')
    parser.add_argument('--output', type=str, default='../../models/')

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, args.db) if not os.path.isabs(args.db) else args.db
    models_dir = os.path.join(script_dir, args.models_dir) if not os.path.isabs(args.models_dir) else args.models_dir
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    # Load data
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = load_and_preprocess(db_path)

    # Feature importance
    feature_importance_analysis(models_dir, feature_names, output_dir, db_path=db_path)

    # Learning curves
    learning_curve_analysis(X_train, y_train, output_dir)

    print("\n[Evaluate] All additional analyses complete!")
    print(f"[Evaluate] Results saved to {output_dir}")


if __name__ == "__main__":
    main()