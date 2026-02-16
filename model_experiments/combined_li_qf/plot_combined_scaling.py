#!/usr/bin/env python3
"""Plot results from combined scaling experiments.

Supports two plot types based on the JSON mode field:
  - union:     Scaling curves (metric vs game count) with error bars
  - mix-ratio: Metric vs QF ratio at fixed game count
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt


METRICS = [
    ('standard', 'log_loss', 'Log Loss', False),
    ('standard', 'auc_roc', 'AUC-ROC', True),
    ('kq', 'egg_inversion_rate', 'Egg Inversion Rate', False),
    ('kq', 'symmetry_deviation', 'Symmetry Deviation', False),
    ('calibration', 'ece', 'ECE', False),
]


def extract_metric(run, suite, key):
    """Extract a metric value from a run's nested metrics dict."""
    return run['metrics'][suite][key]


def plot_union(data, output_path):
    """Plot union-mode scaling curves."""
    runs = data['runs']

    # Group by (max_games, num_leaves, num_trees)
    groups = {}
    for run in runs:
        key = (run['max_games'], run['num_leaves'], run['num_trees'])
        groups.setdefault(key, []).append(run)

    sorted_keys = sorted(groups.keys())
    game_counts = [k[0] for k in sorted_keys]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (suite, metric_key, label, higher_better) in zip(axes, METRICS):
        means, stds = [], []
        for key in sorted_keys:
            vals = [extract_metric(r, suite, metric_key) for r in groups[key]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(game_counts, means, yerr=stds,
                     marker='o', capsize=4, linewidth=2, color='#2196F3')
        ax.fill_between(game_counts, means - stds, means + stds,
                         alpha=0.15, color='#2196F3')

        ax.set_xlabel('Games in Training Pool')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

        # X-axis labels with capacity info
        ax.set_xticks(game_counts)
        labels = []
        for key in sorted_keys:
            mg, nl, nt = key
            if mg >= 1000:
                labels.append(f'{mg//1000}K\n{nl}L/{nt}T')
            else:
                labels.append(f'{mg}\n{nl}L/{nt}T')
        ax.set_xticklabels(labels, fontsize=7)

    # N_states subplot in last panel
    ax = axes[5]
    states_means, states_stds = [], []
    for key in sorted_keys:
        vals = [r['n_states'] for r in groups[key]]
        states_means.append(np.mean(vals))
        states_stds.append(np.std(vals))
    ax.errorbar(game_counts, states_means, yerr=states_stds,
                 marker='s', capsize=4, linewidth=2, color='#4CAF50')
    ax.set_xlabel('Games in Training Pool')
    ax.set_ylabel('Training States')
    ax.set_title('Training States')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(game_counts)
    labels = []
    for key in sorted_keys:
        mg, nl, nt = key
        if mg >= 1000:
            labels.append(f'{mg//1000}K\n{nl}L/{nt}T')
        else:
            labels.append(f'{mg}\n{nl}L/{nt}T')
    ax.set_xticklabels(labels, fontsize=7)

    n_seeds = max(len(groups[k]) for k in sorted_keys)
    fig.suptitle(f'Combined Union Scaling ({n_seeds} seeds per point)\n'
                 'Symmetry-augmented tournament holdout',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_mix_ratio(data, output_path):
    """Plot mix-ratio results: metric vs QF ratio."""
    runs = data['runs']

    # Group by qf_ratio
    groups = {}
    for run in runs:
        ratio = run['qf_ratio']
        groups.setdefault(ratio, []).append(run)

    sorted_ratios = sorted(groups.keys())
    max_games = runs[0]['max_games']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (suite, metric_key, label, higher_better) in zip(axes, METRICS):
        means, stds = [], []
        for ratio in sorted_ratios:
            vals = [extract_metric(r, suite, metric_key) for r in groups[ratio]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(sorted_ratios, means, yerr=stds,
                     marker='o', capsize=4, linewidth=2, color='#FF5722')
        ax.fill_between(sorted_ratios, means - stds, means + stds,
                         alpha=0.15, color='#FF5722')

        ax.set_xlabel('QF Fraction')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # Mark endpoints
        ax.axvline(0.0, color='gray', linestyle=':', alpha=0.3, label='Pure LI')
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.3, label='Pure QF')

    # N_states in last panel
    ax = axes[5]
    states_means, states_stds = [], []
    for ratio in sorted_ratios:
        vals = [r['n_states'] for r in groups[ratio]]
        states_means.append(np.mean(vals))
        states_stds.append(np.std(vals))
    ax.errorbar(sorted_ratios, states_means, yerr=states_stds,
                 marker='s', capsize=4, linewidth=2, color='#4CAF50')
    ax.set_xlabel('QF Fraction')
    ax.set_ylabel('Training States')
    ax.set_title('Training States (deduped)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    n_seeds = max(len(groups[r]) for r in sorted_ratios)
    num_leaves = runs[0]['num_leaves']
    num_trees = runs[0]['num_trees']
    fig.suptitle(f'Mix-Ratio Experiment: {max_games//1000}K games, '
                 f'{num_leaves}L/{num_trees}T ({n_seeds} seeds)\n'
                 'QF Fraction = 0 is pure LI, 1 is pure QF',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot combined scaling experiment results')
    parser.add_argument('input', help='JSON results file from '
                        'combined_scaling_experiment.py')
    parser.add_argument('--output', '-o', default=None,
                        help='Output plot path (default: auto from input)')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    mode = data['mode']
    if args.output:
        output_path = args.output
    else:
        base = args.input.rsplit('.', 1)[0]
        output_path = f'{base}_plot.png'

    if mode == 'union':
        plot_union(data, output_path)
    elif mode == 'mix-ratio':
        plot_mix_ratio(data, output_path)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print(f"\nSummary ({mode}):")
    runs = data['runs']
    if mode == 'union':
        groups = {}
        for run in runs:
            key = run['max_games']
            groups.setdefault(key, []).append(run)
        print(f"{'Games':>8}  {'Seeds':>5}  {'Loss':>18}  "
              f"{'AUC':>18}  {'Egg':>18}  {'ECE':>18}")
        print('-' * 95)
        for mg in sorted(groups.keys()):
            g = groups[mg]
            loss = [r['metrics']['standard']['log_loss'] for r in g]
            auc = [r['metrics']['standard']['auc_roc'] for r in g]
            egg = [r['metrics']['kq']['egg_inversion_rate'] for r in g]
            ece = [r['metrics']['calibration']['ece'] for r in g]
            print(f'{mg:>8}  {len(g):>5}  '
                  f'{np.mean(loss):.4f}+/-{np.std(loss):.4f}  '
                  f'{np.mean(auc):.4f}+/-{np.std(auc):.4f}  '
                  f'{np.mean(egg):.4f}+/-{np.std(egg):.4f}  '
                  f'{np.mean(ece):.4f}+/-{np.std(ece):.4f}')
    elif mode == 'mix-ratio':
        groups = {}
        for run in runs:
            key = run['qf_ratio']
            groups.setdefault(key, []).append(run)
        print(f"{'Ratio':>7}  {'Seeds':>5}  {'Loss':>18}  "
              f"{'AUC':>18}  {'Egg':>18}  {'ECE':>18}")
        print('-' * 95)
        for ratio in sorted(groups.keys()):
            g = groups[ratio]
            loss = [r['metrics']['standard']['log_loss'] for r in g]
            auc = [r['metrics']['standard']['auc_roc'] for r in g]
            egg = [r['metrics']['kq']['egg_inversion_rate'] for r in g]
            ece = [r['metrics']['calibration']['ece'] for r in g]
            print(f'{ratio:>7.2f}  {len(g):>5}  '
                  f'{np.mean(loss):.4f}+/-{np.std(loss):.4f}  '
                  f'{np.mean(auc):.4f}+/-{np.std(auc):.4f}  '
                  f'{np.mean(egg):.4f}+/-{np.std(egg):.4f}  '
                  f'{np.mean(ece):.4f}+/-{np.std(ece):.4f}')


if __name__ == '__main__':
    main()
