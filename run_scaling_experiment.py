#!/usr/bin/env python3
"""Run TabNet scaling experiments across data sizes and hyperparameter configs.

Runs train_tabnet.py with various configurations, collects JSON results,
and generates a markdown report.

Usage:
    python run_scaling_experiment.py
    python run_scaling_experiment.py --quick   # Just 1k and 5k for smoke test
"""

import argparse
import json
import os
import subprocess
import sys
import time


# Experiment configurations: (num_games, n_d, n_a, n_steps, label)
CONFIGS = [
    # Phase 1: Scale data with default architecture
    (1000,   8,  8, 3, "1k/8-8-3"),
    (5000,   8,  8, 3, "5k/8-8-3"),
    (10000,  8,  8, 3, "10k/8-8-3"),
    (50000,  8,  8, 3, "50k/8-8-3"),
    (100000, 8,  8, 3, "100k/8-8-3"),
    (205000, 8,  8, 3, "205k/8-8-3"),
    # Phase 2: Larger architectures at bigger data sizes
    (10000,  16, 16, 5, "10k/16-16-5"),
    (50000,  16, 16, 5, "50k/16-16-5"),
    (100000, 16, 16, 5, "100k/16-16-5"),
    (205000, 16, 16, 5, "205k/16-16-5"),
    (50000,  32, 32, 5, "50k/32-32-5"),
    (100000, 32, 32, 5, "100k/32-32-5"),
    (205000, 32, 32, 5, "205k/32-32-5"),
]

QUICK_CONFIGS = [
    (1000,  8,  8, 3, "1k/8-8-3"),
    (5000,  8,  8, 3, "5k/8-8-3"),
    (5000,  16, 16, 5, "5k/16-16-5"),
]


def _batch_size_for_games(num_games):
    """Scale batch_size with data size so epoch time stays manageable."""
    if num_games <= 5000:
        return 1024
    elif num_games <= 10000:
        return 4096
    elif num_games <= 50000:
        return 16384
    else:
        return 32768


def _virtual_batch_size_for_batch(batch_size):
    """Virtual batch size should be a fraction of batch_size."""
    return min(256, batch_size // 4)


def run_one(num_games, n_d, n_a, n_steps, label, results_dir):
    """Run a single training configuration and return the result dict."""
    json_path = os.path.join(results_dir, f"{label.replace('/', '_')}.json")

    # Skip if result already exists
    if os.path.exists(json_path):
        print(f"\n{'='*60}")
        print(f"SKIPPING (cached): {label}")
        print(f"{'='*60}")
        with open(json_path) as f:
            return json.load(f)

    batch_size = _batch_size_for_games(num_games)
    vbatch_size = _virtual_batch_size_for_batch(batch_size)
    patience = 3
    cmd = [
        sys.executable, 'train_tabnet.py',
        '--num-train-games', str(num_games),
        '--n-d', str(n_d),
        '--n-a', str(n_a),
        '--n-steps', str(n_steps),
        '--batch-size', str(batch_size),
        '--virtual-batch-size', str(vbatch_size),
        '--patience', str(patience),
        '--results-json', json_path,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    wall_time = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {label} (exit code {result.returncode})")
        return None

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        data['wall_time_s'] = wall_time
        data['label'] = label
        # Re-write with wall time
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    return None


def generate_report(results, output_path):
    """Generate a markdown report from collected results."""
    lines = [
        "# TabNet Scaling Experiment Results\n",
        "## Configuration",
        f"- **Test set:** late_tournament_games/late_tournament_game_events.csv.gz",
        f"- **Train/test overlap:** Excluded from training",
        f"- **Device:** {results[0].get('device', 'unknown') if results else 'unknown'}",
        "",
    ]

    if results and results[0].get('device') == 'cuda':
        lines.append("")

    # Table: scaling with data size
    lines.append("## Results\n")
    lines.append("| Config | Train Games | Train Vectors | Log Loss | Accuracy | Egg Inv. | Best Epoch | Train Time | Wall Time |")
    lines.append("|--------|------------|---------------|----------|----------|----------|------------|------------|-----------|")

    for r in results:
        train_vecs = r['train_shape'][0]
        lines.append(
            f"| {r['label']} "
            f"| {r['num_train_games']:,} "
            f"| {train_vecs:,} "
            f"| {r['log_loss']:.4f} "
            f"| {100*r['accuracy']:.1f}% "
            f"| {100*r['inversions']:.1f}% "
            f"| {r['best_epoch']} "
            f"| {r['train_time_s']:.1f}s "
            f"| {r['wall_time_s']:.1f}s |"
        )

    # Analysis section
    lines.append("\n## Analysis\n")

    # Find best configs by metric
    if results:
        best_ll = min(results, key=lambda r: r['log_loss'])
        best_acc = max(results, key=lambda r: r['accuracy'])
        best_inv = min(results, key=lambda r: r['inversions'])
        lines.append(f"- **Best log loss:** {best_ll['label']} ({best_ll['log_loss']:.4f})")
        lines.append(f"- **Best accuracy:** {best_acc['label']} ({100*best_acc['accuracy']:.1f}%)")
        lines.append(f"- **Fewest egg inversions:** {best_inv['label']} ({100*best_inv['inversions']:.1f}%)")

    lines.append("")

    report = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Run TabNet scaling experiments')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick smoke test (3 configs)')
    parser.add_argument('--results-dir', default='scaling_results',
                        help='Directory for JSON results')
    parser.add_argument('--report', default='tabnet_scaling_report.md',
                        help='Output markdown report path')
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else CONFIGS
    os.makedirs(args.results_dir, exist_ok=True)

    results = []
    for num_games, n_d, n_a, n_steps, label in configs:
        r = run_one(num_games, n_d, n_a, n_steps, label, args.results_dir)
        if r:
            results.append(r)

    if results:
        report = generate_report(results, args.report)
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")
        print(report)
    else:
        print("\nNo results collected. All runs failed.")


if __name__ == '__main__':
    main()
