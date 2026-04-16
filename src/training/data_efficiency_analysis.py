"""
data_efficiency_analysis.py — Test model robustness when training data is reduced.

Trains all 4 models on 100%, 75%, 50%, 25% of training data and compares.
Shows how each model degrades (or not) with less data.

Usage:
    python3 src/training/data_efficiency_analysis.py 
"""

import argparse
import os
import warnings
import numpy as np

from preprocess import load_and_preprocess
from model_loader import load_models_from_config

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
plt.style.use("seaborn-v0_8-paper")

warnings.filterwarnings('ignore')


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'model_configs.json'
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'assets', 'data_efficiency'
)

DATABASE_DIFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)


def get_models(config_path=DEFAULT_CONFIG_PATH):
    """
    Load model instances from model_configs.json via model_loader.

    Args:
        config_path: path to model_configs.json
    """
    models = load_models_from_config(config_path)
    return {name: model for name, (model, _) in models.items()}


def run_experiment(X_train, X_test, y_train, y_test, label_encoder, fractions, config_path=DEFAULT_CONFIG_PATH):
    """
    Train each model on different fractions of training data.

    Args:
        X_train, X_test, y_train, y_test: preprocessed data
        label_encoder: fitted LabelEncoder for room labels
        fractions: list of floats (e.g. [0.25, 0.5, 0.75, 1.0])
        config_path: path to model_configs.json

    Returns:
        dict of {model_name: {fraction: {accuracy, f1, n_train, report}}}
    """
    results = {}
    base_models = get_models(config_path)

    for name, model in base_models.items():
        results[name] = {}

        for frac in fractions:
            if frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train,
                    train_size=frac,
                    stratify=y_train if frac >= 0.1 else None, # avoid stratify for very small fractions to prevent errors
                    random_state=42
                )
            else:
                X_sub, y_sub = X_train, y_train

            # Clone gives a fresh unfitted copy with the same params
            model_fresh = clone(model)
            model_fresh.fit(X_sub, y_sub)

            # Evaluate
            y_pred = model_fresh.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            report = classification_report(y_test, y_pred,
                                           target_names=label_encoder.classes_,
                                           digits=4, output_dict=True)

            results[name][frac] = {
                'accuracy': acc,
                'f1': f1,
                'n_train': len(X_sub),
                'report': report
            }

            print(f"  {name:<15} | {frac*100:5.0f}% ({len(X_sub):4d} samples) "
                f"| Acc: {acc:.4f} | F1: {f1:.4f}")
    return results


def plot_degradation_curves(results, fractions, output_dir=OUTPUT_DIR):
    """
    Plot accuracy vs data fraction for each model.

    Args:
        results: dict of {model_name: {fraction: {accuracy, f1, n_train, report}}}
        fractions: list of data fractions used
        output_dir: directory to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = {
        'Weighted KNN': '#4C72B0',
        'Random Forest': '#DD8452',
        'SVM': '#55A868',
        'MLP': '#C44E52',
    }
    markers = {
        'Weighted KNN': 'o',
        'Random Forest': 's',
        'SVM': '^',
        'MLP': 'D',
    }

    for name in results:
        accs = [results[name][f]['accuracy'] for f in fractions]
        f1s = [results[name][f]['f1'] for f in fractions]
        percentages = [f * 100 for f in fractions]
        sample_sizes = [results[name][f]['n_train'] for f in fractions]

        labels = [f"{int(p)}%\n(n={n})" for p, n in zip(percentages, sample_sizes)]

        ax1.plot(percentages, accs, marker=markers[name], color=colors[name],
                label=name, linewidth=2, markersize=7)
        ax2.plot(percentages, f1s, marker=markers[name], color=colors[name],
                label=name, linewidth=2, markersize=7)

    for ax, metric in [(ax1, 'Accuracy'), (ax2, 'F1 Score (weighted)')]:
        ax.set_xlabel(r'\% of Training Data Used', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs Training Data Size', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xticks(percentages)
        ax.set_xticklabels(labels)
        ax.set_ylim(max(0, ax.get_ylim()[0] - 0.05), 1.02)

    plt.suptitle('Data Efficiency --- Model Robustness to Reduced Training Data',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_efficiency_curves.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Plot] Saved data_efficiency_curves.pdf")


def plot_bar_comparison(results, output_dir=OUTPUT_DIR):
    """
    Bar chart comparing 100% vs 50% for each model.

    Args:
        results: dict of {model_name: {fraction: {accuracy, f1, n_train, report}}}
        output_dir: directory to save the plot
    """
    names = list(results.keys())
    full_f1 = [results[n][1.0]['f1'] for n in names]
    half_f1 = [results[n][0.5]['f1'] for n in names]
    quarter_f1 = [results[n][0.25]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width, full_f1, width, label=r'100\% Data', color='#4C72B0', alpha=0.85, hatch='///', edgecolor='black')
    bars2 = ax.bar(x, half_f1, width, label=r'50\% Data', color='#C44E52', alpha=0.85, hatch='..', edgecolor='black')
    bars3 = ax.bar(x + width, quarter_f1, width, label=r'25\% Data', color='#55A868', alpha=0.85, hatch='xxx', edgecolor='black')

    ax.set_ylabel('F1 Score (weighted)', fontsize=11)
    ax.set_title('Impact of Training Data Reduction on Model Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(max(0, min(quarter_f1) - 0.1), 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 4), textcoords="offset points", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_efficiency_bars.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved data_efficiency_bars.pdf")


def plot_per_room_degradation(results, label_encoder, output_dir=OUTPUT_DIR):
    """
    Show per-room F1 at 100% vs 50% for each model.

    Args:
        results: dict of {model_name: {fraction: {accuracy, f1, n_train, report}}}
        label_encoder: fitted LabelEncoder to get room names
        output_dir: directory to save the plot
    """
    rooms = label_encoder.classes_
    names = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, name in zip(axes, names):
        full_f1s = [results[name][1.0]['report'].get(r, {}).get('f1-score', 0) for r in rooms]
        half_f1s = [results[name][0.5]['report'].get(r, {}).get('f1-score', 0) for r in rooms]

        x = np.arange(len(rooms))
        width = 0.35

        ax.bar(x - width/2, full_f1s, width, label=r'100\% Data', color='#4C72B0', alpha=0.8, hatch='..', edgecolor='black')
        ax.bar(x + width/2, half_f1s, width, label=r'50\% Data', color='#C44E52', alpha=0.8, hatch='///', edgecolor='black')

        ax.set_ylabel('F1 Score')
        ax.set_title(name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rooms, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Room Performance: Full vs Half Training Data', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_efficiency_per_room.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved data_efficiency_per_room.pdf")

def parse_args():
    """
    Parse command-line arguments for the data efficiency experiment.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, default=DATABASE_DIFAULT_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    return parser.parse_args()
    

def main():
    """
    Main function to run the data efficiency experiment.    
    """
    
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, args.db) if not os.path.isabs(args.db) else args.db
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = load_and_preprocess(db_path)
    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]

    print("\n" + "=" * 70)
    print("  DATA EFFICIENCY EXPERIMENT")
    print(f"  Testing fractions: {[f'{f*100:.0f}%' for f in fractions]}")
    print(f"  Full training size: {len(X_train)}")
    print("=" * 70)

    results = run_experiment(X_train, X_test, y_train, y_test, label_encoder, fractions)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':<20}", end="")
    for f in fractions:
        n = results[list(results.keys())[0]][f]['n_train']
        print(f"  {f*100:4.0f}%({n})", end="")
    print(f"  {'Drop':>8}")
    print(f"  {'─' * 65}")

    for name in results:
        print(f"  {name:<20}", end="")
        for f in fractions:
            print(f"  {results[name][f]['f1']:.3f}", end="")
        drop = results[name][1.0]['f1'] - results[name][0.5]['f1']
        print(f"  {drop:+7.4f}")

    
    print("\nGenerating plots...")
    plot_degradation_curves(results, fractions, output_dir)
    plot_bar_comparison(results, output_dir)
    plot_per_room_degradation(results, label_encoder, output_dir)

if __name__ == "__main__":
    main()