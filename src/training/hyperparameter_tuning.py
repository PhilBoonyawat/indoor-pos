"""
hyperparameter_tuning.py — Run GridSearchCV for KNN, Random Forest, SVM, and MLP using configurations from model_loader.py, and generate line plots to visualize hyperparameter performance.

Usage:
    Run tuning and plotting from command line:
        python3 src/training/hyperparameter_tuning.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from model_loader import load_models_from_config, save_best_params
from preprocess import load_and_preprocess

DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'data', 'raw', 'wifi_scans.db'
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'assets', 'hyperparameter_tuning', 'grid_search'
)

def get_param_grids():
    """
    Returns: 
        dictionary of hyperparameter grids for each model to be used in GridSearchCV
    """
    
    return {
        "Weighted KNN": {
            "n_neighbors": [3,5,7,9,11],
            "weights": ["uniform", "distance"],
            "metric": ["manhattan", "euclidean"]
        },
        "Random Forest": {
            "n_estimators": [50, 100,200],
            "max_depth": [None,10,20],
            "min_samples_split": [2,5],
            "min_samples_leaf": [1,2],
            "max_features": ["sqrt","log2"]
        },
        "SVM": {
            "C": [0.1,1,10],
            "gamma": ["scale",0.01,0.1],
            "kernel": ["rbf"]
        },
        "MLP": {
            "hidden_layer_sizes": [
                (64,),
                (128,),
                (128,64),
                (256,128),
                (128,64,32),
                (256,128,64)
            ],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-4, 5e-4, 1e-3]
        }
    }

def run_grid_search(models, param_grids, X_train, y_train):
    """
    Runs GridSearchCV for each model and returns a dictionary of results.

    Args: 
        models: dict of {model_name: model_instance}
        param_grids: dict of {model_name: param_grid_dict}
        X_train, y_train: training data for fitting the GridSearchCV

    Returns:
        dict of {model_name: {best_model, best_params, best_score, cv_results}}
    """
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running GridSearchCV for each model...")

    for name, (model, _) in models.items():
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        results[name] = {
            "best_model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "cv_results": grid.cv_results_
        }

    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_knn_lines(results, output_dir):
    """
    Plots CV performance of KNN across n_neighbors, weights, and metric.

    Args:
        results: dict of GridSearchCV results for KNN
        output_dir: directory to save the plot

    Returns:
        Saves a PDF plot of KNN hyperparameter performance
    """
    
    df = pd.DataFrame(results["Weighted KNN"]["cv_results"])
    metrics = df["param_metric"].unique()

    _, axes = plt.subplots(1, len(metrics), figsize=(10,4), sharey=True)

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        subset_metric = df[df["param_metric"] == metric]

        for weight in ["uniform", "distance"]:
            subset = subset_metric[subset_metric["param_weights"] == weight]
            subset = subset.sort_values("param_n_neighbors")

            ax.plot(
                subset["param_n_neighbors"],
                subset["mean_test_score"],
                marker='o',
                label=weight
            )

        ax.set_title(f"{metric.capitalize()} Distance", fontweight='bold')
        ax.set_xlabel("k")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("CV F1 Score")
    axes[0].legend(title="Weights")

    plt.suptitle("KNN Hyperparameter Tuning", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/knn_grid_search.pdf")
    plt.close()

    print("[Plot] Saved knn_grid_search.pdf")

def plot_rf_lines(results, output_dir):
    """
    Plots CV performance of Random Forest across n_estimators, max_depth, and max_features using faceted line plots.

    Args:
        results: dict of GridSearchCV results for Random Forest
        output_dir: directory to save the plot

    Returns:
        Saves a PDF plot of Random Forest hyperparameter performance
    """

    df = pd.DataFrame(results["Random Forest"]["cv_results"])
    df["param_max_depth"] = df["param_max_depth"].apply(
        lambda x: "None" if x is None else str(x)
    )

    # Fixed parameters for clearer visualisation (based on preliminary analysis) 
    df = df[
        (df["param_min_samples_split"] == 5) &
        (df["param_min_samples_leaf"] == 1)
    ]

    df = df.sort_values("param_n_estimators")

    max_features_vals = sorted(df["param_max_features"].unique())
    depths = sorted(df["param_max_depth"].unique(), key=lambda x: (x != "None", x))

    n_cols = len(max_features_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=True)

    if n_cols == 1:
        axes = [axes]

    colors = {
        "None": "#4C72B0",
        "10": "#55A868",
        "20": "#C44E52"
    }

    linestyles = {
        "None": "-",
        "10": "--",
        "20": ":"
    }

    markers = ["o", "s", "^"]

    for ax, max_feat in zip(axes, max_features_vals):
        subset_feat = df[df["param_max_features"] == max_feat]

        for i, depth in enumerate(depths):
            subset = subset_feat[subset_feat["param_max_depth"] == depth]

            if subset.empty:
                continue

            ax.plot(
                subset["param_n_estimators"],
                subset["mean_test_score"],
                marker=markers[i % len(markers)],
                linestyle=linestyles.get(depth, "-"),
                linewidth=2,
                markersize=6,
                label=f"depth={depth}",
                color=colors.get(depth, None)
            )

        ax.set_title(f"max_features = {max_feat}", fontweight='bold')
        ax.set_xlabel("Number of Trees")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("CV F1 Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Max Depth",
               loc='lower center', ncol=len(depths), fontsize=9)

    ymin = df["mean_test_score"].min() - 0.01
    ymax = df["mean_test_score"].max() + 0.005
    for ax in axes:
        ax.set_ylim(ymin, min(1.01, ymax))

    plt.suptitle("Random Forest Hyperparameter Tuning",
                 fontsize=14, fontweight='bold')

    plt.subplots_adjust(bottom=0.2, top=0.88)
    plt.savefig(os.path.join(output_dir, "rf_grid_search.pdf"), dpi=150)
    plt.close()

    print("[Plot] Saved rf_grid_search.pdf")

def plot_svm_lines(results, output_dir):
    """
    Plots CV performance of SVM across C and gamma parameter values using line plots.

    Args:
        results: dict of GridSearchCV results for SVM
        output_dir: directory to save the plot  

    Returns:
        Saves a PDF plot of SVM hyperparameter performance
    """

    df = pd.DataFrame(results["SVM"]["cv_results"])
    df["param_gamma"] = df["param_gamma"].astype(str)
    df = df.sort_values("param_C")

    gammas = sorted(df["param_gamma"].unique())


    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D"]

    plt.figure(figsize=(6, 4))

    for i, gamma in enumerate(gammas):
        subset = df[df["param_gamma"] == gamma]

        plt.plot(
            subset["param_C"],
            subset["mean_test_score"],
            label=f"gamma={gamma}",
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6
        )

    plt.xscale("log")  
    plt.xlabel("C (log scale)", fontsize=11)
    plt.ylabel("CV F1 Score", fontsize=11)
    plt.title("SVM Hyperparameter Tuning", fontsize=13, fontweight='bold')

    plt.legend(title="Gamma", fontsize=9)
    plt.grid(alpha=0.3)

    ymin = df["mean_test_score"].min() - 0.01
    ymax = df["mean_test_score"].max() + 0.005
    plt.ylim(ymin, min(1.01, ymax))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/svm_grid_search.pdf", dpi=150)
    plt.close()

    print("[Plot] Saved svm_grid_search.pdf")


def plot_mlp_grid(results, output_dir):
    """
    Plots CV performance of MLP across hidden_layer_sizes and learning_rate_init using a grid of line plots.

    Args:
        results: dict of GridSearchCV results for MLP
        output_dir: directory to save the plot

    Returns:
        Saves a PDF plot of MLP hyperparameter performance
    """
    df = pd.DataFrame(results["MLP"]["cv_results"])

    # Focus on alpha=1e-5 for clearer visualisation (based on preliminary analysis)
    df = df[df["param_alpha"] == 1e-5]

    df["architecture"] = df["param_hidden_layer_sizes"].astype(str)
    df["lr"] = df["param_learning_rate_init"]

    plt.figure(figsize=(6, 4))

    palette = {
        0.0001: "#1f77b4",
        0.0005: "#d62728",
        0.001: "#2ca02c",
        0.01: "#DD8452",
        0.1: "#9421a6"
    }

    sns.barplot(
        data=df,
        x="architecture",
        y="mean_test_score",
        hue="lr",
        palette=palette
    )

    plt.xlabel("Hidden Layer Architecture")
    plt.ylabel("CV F1 Score")
    plt.title("MLP Hyperparameter Tuning", fontweight='bold')

    plt.xticks(rotation=20)
    plt.legend(title="Learning Rate", loc='lower right')
    plt.grid(axis='y', alpha=0.3)

    ymin = df["mean_test_score"].min() - 0.005
    ymax = df["mean_test_score"].max() + 0.002
    plt.ylim(ymin, min(1.01, ymax))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mlp_grid_search.pdf", dpi=150)
    plt.close()

    print("[Plot] Saved mlp_grid_search.pdf")

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    X_train, _, y_train, _, _, _, _ = load_and_preprocess(DATABASE_PATH)

    models = load_models_from_config()
    param_grids = get_param_grids()
    results = run_grid_search(models, param_grids, X_train, y_train)

    print("\nGenerating plots...")
    print("Result keys:", list(results.keys()))
    plot_knn_lines(results, OUTPUT_PATH)
    plot_rf_lines(results, OUTPUT_PATH)
    plot_svm_lines(results, OUTPUT_PATH)
    plot_mlp_grid(results, OUTPUT_PATH)

    print("\n" + "="*60)
    print("BEST PARAMETERS")
    print("="*60)

    for name, res in results.items():
        print(f"\n{name}")
        print(f"  Params: {res['best_params']}")
        print(f"  CV F1:  {res['best_score']:.4f}")

    save_best_params(results)


if __name__ == "__main__":
    main()