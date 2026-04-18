"""
train.py — Train and compare 4 classification models for Wi-Fi fingerprint room prediction.

Models: Weighted KNN, Random Forest, MLP, SVM

Usage:
    python train.py                              # Uses default DB path and output directory
    python train.py --db ../../data/raw/wifi_scans.db --output ../../models

Outputs:
    - Trained model files (.pkl) in output directory
    - Comparison plots (accuracy, confusion matrices)
    - Classification reports for each model
"""

import argparse
import os
import json
import time
import warnings
import numpy as np
import joblib

from preprocess import load_and_preprocess, load_and_preprocess_temporal

from model_loader import load_models_from_config
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.base import clone

import matplotlib
matplotlib.use('pdf') 
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.figsize"] = (6,4)

warnings.filterwarnings('ignore')

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'model_configs.json'
)

FIG_OUTPUT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'assets', 'model_training', 'comparisons'
)

MODEL_OUTPUT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'models'
)

DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)

def get_models(model_config_path=DEFAULT_CONFIG_PATH):
    """
    Load model configurations from model_configs.json and create scikit-learn model instances.

    Args:
        model_config_path: path to model_configs.json

    Returns:
        dict of {model_name: (model_instance, description_string)}
    """
    return load_models_from_config(model_config_path)

def train_and_evaluate(models, X_train, X_test, y_train, y_test, label_encoder, cv_folds=5):
    """
    Train each model, run cross-validation, evaluate on test set.
    
    Args:
        models: dict of {model_name: (model_instance, description_string)}
        X_train, X_test, y_train, y_test: preprocessed data
        label_encoder: fitted LabelEncoder for room labels
        cv_folds: number of cross-validation folds (default: 5)

    Returns:
        dict of {model_name: {cv_scores, test_accuracy, test_f1, confusion_matrix, report, train_time, predict_time, y_pred, description}}
    """
    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("  Training & Evaluation — 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    for name, (model, desc) in models.items():
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"  Config: {desc}")
        print(f"{'─' * 60}")

        # Cross-validation
        start = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_time = time.time() - start

        print(f"  CV F1 (weighted):  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  CV F1 Scores:     {cv_scores}")
        print(f"  CV Time:      {cv_time:.2f}s")

        # Train on full training set
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Predict on test set
        start = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start

        # Metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, digits=4,
            target_names=label_encoder.classes_,
            output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y_test, y_pred, digits=4,
            target_names=label_encoder.classes_, zero_division=0
        )

        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1:       {test_f1:.4f}")
        print(f"  Train Time:    {train_time:.2f}s")
        print(f"  Predict Time:  {predict_time:.4f}s")
        print(f"\n  Classification Report:")
        for line in report_str.split('\n'):
            print(f"    {line}")

        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'confusion_matrix': cm,
            'report': report,
            'report_str': report_str,
            'train_time': train_time,
            'predict_time': predict_time,
            'y_pred': y_pred,
            'description': desc
        }

    return results


def plot_comparison(results, label_encoder, output_dir=FIG_OUTPUT_DIRECTORY):
    """
    Create comparison plots:
    1. Bar chart of CV F1 vs Test F1 for each model
    2. Confusion matrices for each model
    3. Training time and prediction time comparison
    4. Box plot of CV F1 score distributions

    Args:
        results: dict of {model_name: {cv_scores, test_accuracy, test_f1, confusion_matrix, report, train_time, predict_time, y_pred, description}}
        label_encoder: fitted LabelEncoder for room labels
        output_dir: directory to save plots
    """
    model_names = list(results.keys())
    cv_means = [results[n]['cv_mean'] for n in model_names]
    cv_stds = [results[n]['cv_std'] for n in model_names]
    test_f1s = [results[n]['test_f1'] for n in model_names]

    # ── 1. F1 Score Comparison Bar Chart ─────────
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, cv_means, width, label='CV F1 (weighted)', color='#3b82f6',
                   yerr=cv_stds, capsize=5, alpha=0.9)
    bars2 = ax.bar(x + width/2, test_f1s, width,
                   label='Test F1 (weighted)')

    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Model Comparison — WiFi Fingerprint Room Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(max(0, min(cv_means) - 0.15), 1.02)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.pdf'), dpi=150)
    plt.close()
    print(f"\n[Plot] Saved model_comparison.pdf")

    # ── 2. Confusion Matrices ────────────────────
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        ax.set_title(f'{name}\nAcc: {results[name]["test_accuracy"]:.3f}', fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Confusion Matrices — Per Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved confusion_matrices.pdf")

    # ── 3. Training & Prediction Time ────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.5))

    train_times = [results[n]['train_time'] for n in model_names]
    predict_times = [results[n]['predict_time'] * 1000 for n in model_names]  # ms

    ax1.barh(model_names, train_times, color='#f59e0b', alpha=0.9)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Training Time', fontweight='bold')
    for i, v in enumerate(train_times):
        ax1.text(v + 0.02, i, f'{v:.2f}s', va='center', fontsize=9)

    ax2.barh(model_names, predict_times, color='#8b5cf6', alpha=0.9)
    ax2.set_xlabel('Time (milliseconds)')
    ax2.set_title('Prediction Time (full test set)', fontweight='bold')
    for i, v in enumerate(predict_times):
        ax2.text(v + 0.02, i, f'{v:.1f}ms', va='center', fontsize=9)

    plt.suptitle('Computational Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison.pdf'), dpi=150)
    plt.close()
    print(f"[Plot] Saved timing_comparison.pdf")

    # ── 4. Cross-Validation Box Plot ─────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    cv_data = [results[n]['cv_scores'] for n in model_names]
    bp = ax.boxplot(cv_data, tick_labels=model_names, patch_artist=True)

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for patch, color in zip(bp['boxes'], colors[:len(model_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('F1 Score')
    ax.set_title('Cross-Validation F1 Score Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_boxplot.pdf'), dpi=150)
    plt.close()
    print(f"[Plot] Saved cv_boxplot.pdf")

def export_models(results, label_encoder, scaler, feature_names, output_dir=MODEL_OUTPUT_DIRECTORY):
    """
    Save trained models and metadata.
    
    Args:
        results: dict of {model_name: {model, cv_scores, test_accuracy, test_f1, confusion_matrix, report, train_time, predict_time, y_pred, description}}
        label_encoder: fitted LabelEncoder for room labels
        scaler: fitted scaler for feature normalization
        feature_names: list of feature names (BSSIDs)
        output_dir: directory to save models and metadata

    Returns:
        name of best model (highest test F1 score)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    for name, data in results.items():
        safe_name = name.lower().replace(' ', '_')
        model_path = os.path.join(output_dir, f'{safe_name}.pkl')
        joblib.dump(data['model'], model_path)
        print(f"[Export] Saved {model_path}")

    # Save label encoder and scaler
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # Save feature names (ordered list of BSSIDs)
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Save comparison summary
    summary = {}
    best_model = None
    best_acc = 0

    for name, data in results.items():
        summary[name] = {
            'cv_f1_mean': round(data['cv_mean'], 4),
            'cv_f1_std': round(data['cv_std'], 4),
            'test_accuracy': round(data['test_accuracy'], 4),
            'test_f1_weighted': round(data['test_f1'], 4),
            'train_time_seconds': round(data['train_time'], 3),
            'predict_time_seconds': round(data['predict_time'], 5),
            'config': data['description']
        }
        if data['test_f1'] > best_acc:
            best_acc = data['test_f1']
            best_model = name

    summary['_best_model'] = best_model
    summary['_best_accuracy'] = round(best_acc, 4)

    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Export] Saved comparison_summary.json")
    print(f"[Export] All models and metadata saved to {output_dir}/")

    return best_model


def temporal_validation(models, X_train, X_test, y_train, y_test, label_encoder):
    """
    Perform temporal validation to test for data leakage. Train on early scans and test on later scans.

    Args:
        models: dict of {model_name: (model_instance, description_string)}
        X_train, X_test, y_train, y_test: preprocessed data for temporal split
        label_encoder: fitted LabelEncoder for room labels

    Returns:
        dict of {model_name: {test_accuracy, test_f1, confusion_matrix, y_pred}}
    """
    print("\n" + "=" * 60)
    print("  TEMPORAL VALIDATION — Data Leakage Test")
    print("  (Train on early scans → Test on later scans)")
    print("=" * 60)

    temporal_results = {}

    for name, (model, desc) in models.items():
        # Create a fresh model instance (don't reuse fitted model)
        fresh_model = clone(model)

        fresh_model.fit(X_train, y_train)
        y_pred = fresh_model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        report_str = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_
            , digits=4, zero_division=0
        )

        print(f"\n  {name}")
        print(f"  Temporal Test Accuracy: {test_acc:.4f}")
        print(f"  Temporal Test F1:       {test_f1:.4f}")
        print(f"\n  Classification Report:")
        for line in report_str.split('\n'):
            print(f"    {line}")

        temporal_results[name] = {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }

    return temporal_results


def plot_leakage_comparison(random_results, temporal_results, label_encoder, output_dir=FIG_OUTPUT_DIRECTORY):
    """
    Create comparison plots for data leakage test:
    1. Side-by-side bar chart of test accuracy for random split vs temporal split
    2. Confusion matrices for temporal split

    Args:
        random_results: dict of {model_name: {test_accuracy, ...}} from random split
        temporal_results: dict of {model_name: {test_accuracy, confusion_matrix, ...}}
        label_encoder: fitted LabelEncoder for room labels
        output_dir: directory to save plots
    """
    names = [n for n in random_results.keys() if n in temporal_results]
    random_accs = [random_results[n]['test_accuracy'] for n in names]
    temporal_accs = [temporal_results[n]['test_accuracy'] for n in names]

    # ── 1. Side-by-side accuracy comparison ──────
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, random_accs, width, label='Random Split', color='#3b82f6', alpha=0.9)
    bars2 = ax.bar(x + width/2, temporal_accs, width, label='Temporal Split', color='#f59e0b', alpha=0.9)

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Data Leakage Test — Random vs Temporal Split', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(max(0, min(temporal_accs) - 0.1), 1.02)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leakage_test.pdf'), dpi=150)
    plt.close()
    print(f"\n[Plot] Saved leakage_test.pdf")

    # ── 2. Temporal confusion matrices ───────────
    n_models = len(names)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        cm = temporal_results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        ax.set_title(f'{name}\nTemporal Acc: {temporal_results[name]["test_accuracy"]:.3f}', fontsize=11)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('Temporal Split — Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_confusion_matrices.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved temporal_confusion_matrices.pdf")

def parse_args():
    """
    Parse command-line arguments for training script.

    Returns:
        argparse.Namespace with attributes:
            db: path to SQLite database (default: ../../data/raw/wifi_scans.db)
            output: output directory for trained models (default: ../../models/)
            test_size: test set proportion (default: 0.2)
            cv_folds: number of cross-validation folds (default: 5)
            min_detection_rate: minimum AP detection rate to keep (default: 0.05)   
    """
    parser = argparse.ArgumentParser(description="Train WiFi fingerprint room classifiers")
    parser.add_argument('--db', type=str, default=DATABASE_PATH,
                        help='Path to SQLite database')
    parser.add_argument('--output', type=str, default=MODEL_OUTPUT_DIRECTORY,
                        help='Output directory for trained models')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--min-detection-rate', type=float, default=0.05,
                        help='Min AP detection rate to keep (default: 0.05)')
    return parser.parse_args()


def main():
    """
    Main function to run the training and evaluation pipeline:
    1. Load and preprocess data from SQLite database
    2. Load model configurations and create model instances
    3. Train each model and evaluate on test set
    4. Create comparison plots
    5. Export trained models and metadata
    6. Run temporal validation to test for data leakage
    7. Print final summary of results and leakage assessment
    """
    args = parse_args()

    # # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, args.db) if not os.path.isabs(args.db) else args.db
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    os.makedirs(output_dir, exist_ok=True)

    # 1. Preprocess
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = load_and_preprocess(
        db_path,
        test_size=args.test_size,
        random_state=42,
        min_detection_rate=args.min_detection_rate
    )

    # 2. Get models
    models = get_models()

    # 3. Train and evaluate
    results = train_and_evaluate(
        models, X_train, X_test, y_train, y_test,
        label_encoder, cv_folds=args.cv_folds
    )

    # 4. Plot comparisons
    plot_comparison(results, label_encoder, output_dir)

    # 5. Export models
    best_model = export_models(results, label_encoder, scaler, feature_names, output_dir)

    # 6. Temporal validation (data leakage test)
    print("\n\n" + "#" * 60)
    print("  RUNNING TEMPORAL VALIDATION (LEAKAGE TEST)")
    print("#" * 60)

    X_train_t, X_test_t, y_train_t, y_test_t, _, _, _ = load_and_preprocess_temporal(
        db_path, test_ratio=args.test_size, min_detection_rate=args.min_detection_rate
    )

    models_fresh = get_models()
    temporal_results = temporal_validation(
        models_fresh, X_train_t, X_test_t, y_train_t, y_test_t, label_encoder
    )

    plot_leakage_comparison(results, temporal_results, label_encoder, output_dir)

    # 7. Final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"\n  {'Model':<20} {'CV F1':>10} {'Random':>10} {'Temporal':>10} {'Diff':>10}")
    print(f"  {'─' * 60}")

    for name in sorted(results.keys(), key=lambda n: results[n]['test_f1'], reverse=True):
        r = results[name]
        t_acc = temporal_results[name]['test_accuracy']
        diff = r['test_accuracy'] - t_acc
        marker = " ← BEST" if name == best_model else ""
        print(f"  {name:<20} {r['cv_mean']:>9.4f} {r['test_accuracy']:>10.4f} {t_acc:>10.4f} {diff:>+10.4f}{marker}")

    # Leakage assessment
    avg_random = np.mean([results[n]['test_accuracy'] for n in results])
    avg_temporal = np.mean([temporal_results[n]['test_accuracy'] for n in temporal_results])
    gap = avg_random - avg_temporal

    print(f"\n  Average Random Split:   {avg_random:.4f}")
    print(f"  Average Temporal Split: {avg_temporal:.4f}")
    print(f"  Gap:                    {gap:+.4f}")

    if gap < 0.02:
        print(f"\n  MINIMAL LEAKAGE — gap < 2%. Results are trustworthy.")
        print(f"  The model generalises well across time.")
    elif gap < 0.10:
        print(f"\n  MODERATE LEAKAGE — gap {gap:.1%}. Some temporal correlation.")
        print(f"  Random split may be slightly optimistic. Temporal accuracy is more realistic.")
    else:
        print(f"\n  SIGNIFICANT LEAKAGE — gap {gap:.1%}. Consecutive scans are too similar.")
        print(f"  Use temporal split results as your true accuracy.")

    print(f"\n  Best model: {best_model} ({results[best_model]['test_accuracy']:.4f} random / {temporal_results.get(best_model, {}).get('test_accuracy', 0):.4f} temporal)")
    print(f"  Models saved to: {output_dir}")
    print(f"\n  To use in the web app:")
    safe_best = best_model.lower().replace(' ', '_')
    print(f"    python run.py --model {output_dir}/{safe_best}.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()