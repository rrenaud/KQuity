#!/usr/bin/env python3
"""Plot scaling results from exclusive equal-states experiments."""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Capacity schedule used in the scaling experiment
CAPACITY = {
    5000: (70, 70),
    10000: (100, 100),
    20000: (100, 100),
    40000: (150, 150),
    75000: (200, 200),
}


def load_scaling_results():
    """Load all scaling_exclusive_*.json files, return sorted by max_games."""
    files = sorted(glob.glob('model_experiments/scaling_exclusive_*.json'))
    if not files:
        raise FileNotFoundError("No scaling_exclusive_*.json files found")

    results = []
    for path in files:
        # Extract max_games from filename
        max_games = int(path.split('_')[-1].replace('.json', ''))
        with open(path) as f:
            runs = json.load(f)
        results.append((max_games, runs))

    results.sort(key=lambda x: x[0])
    return results


def main():
    results = load_scaling_results()

    metrics = [
        ('log_loss', 'Log Loss'),
        ('auc_roc', 'AUC-ROC'),
        ('egg_inversion_rate', 'Egg Inversion Rate'),
        ('symmetry_deviation', 'Symmetry Deviation'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        game_counts = []
        qf_means, qf_stds = [], []
        li_means, li_stds = [], []

        for max_games, runs in results:
            qf_vals = [r['qf_exclusive'][metric_key] for r in runs]
            li_vals = [r['li_exclusive'][metric_key] for r in runs]

            game_counts.append(max_games)
            qf_means.append(np.mean(qf_vals))
            qf_stds.append(np.std(qf_vals))
            li_means.append(np.mean(li_vals))
            li_stds.append(np.std(li_vals))

        qf_means = np.array(qf_means)
        qf_stds = np.array(qf_stds)
        li_means = np.array(li_means)
        li_stds = np.array(li_stds)

        ax.errorbar(game_counts, qf_means, yerr=qf_stds,
                     marker='o', capsize=4, label='QF-exclusive')
        ax.errorbar(game_counts, li_means, yerr=li_stds,
                     marker='s', capsize=4, label='LI-exclusive')

        ax.set_xlabel('Max Games')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # X-axis: show game count + capacity
        ax.set_xticks(game_counts)
        labels = []
        for g in game_counts:
            cap = CAPACITY.get(g)
            if cap:
                labels.append(f'{g//1000}K\n{cap[0]}L/{cap[1]}T')
            else:
                labels.append(f'{g//1000}K')
        ax.set_xticklabels(labels, fontsize=8)

    fig.suptitle('Exclusive Equal-States Scaling: QF vs LI\n'
                 '(symmetry-augmented holdout, 10 variance runs)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_experiments/exclusive_scaling_plot.png', dpi=150)
    print("Saved model_experiments/exclusive_scaling_plot.png")

    # Print summary table
    print("\nScaling Summary (mean +/- std):")
    print(f"{'Games':>8} {'Cap':>10} {'States':>10}  "
          f"{'QF Loss':>18}  {'LI Loss':>18}  "
          f"{'QF Egg':>18}  {'LI Egg':>18}")
    print('-' * 110)
    for max_games, runs in results:
        cap = CAPACITY.get(max_games, ('?', '?'))
        qf_loss = [r['qf_exclusive']['log_loss'] for r in runs]
        li_loss = [r['li_exclusive']['log_loss'] for r in runs]
        qf_egg = [r['qf_exclusive']['egg_inversion_rate'] for r in runs]
        li_egg = [r['li_exclusive']['egg_inversion_rate'] for r in runs]
        n_states = int(np.mean([r['qf_exclusive']['n_states'] for r in runs]))
        print(f"{max_games:>8} {cap[0]}L/{cap[1]}T {n_states:>10,}  "
              f"{np.mean(qf_loss):.4f} +/- {np.std(qf_loss):.4f}  "
              f"{np.mean(li_loss):.4f} +/- {np.std(li_loss):.4f}  "
              f"{np.mean(qf_egg):.4f} +/- {np.std(qf_egg):.4f}  "
              f"{np.mean(li_egg):.4f} +/- {np.std(li_egg):.4f}")


if __name__ == '__main__':
    main()
