#!/usr/bin/env python3
"""Run TabNet hyperparameter sweep varying n_d/n_a and n_steps independently.

Uses the fixed early stopping (logloss-based) and low patience.
Fixed at 50k training games where accuracy plateaued in Phase 1.

Usage:
    python run_hparam_sweep.py
"""

import json
import os
import subprocess
import sys
import time


NUM_GAMES = 50000
BATCH_SIZE = 16384
VIRTUAL_BATCH_SIZE = 256
PATIENCE = 3
MAX_EPOCHS = 20

# Sweep: vary n_d/n_a and n_steps independently
N_D_VALUES = [8, 16, 32, 64]
N_STEPS_VALUES = [3, 5, 7]

RESULTS_DIR = 'hparam_sweep_results'


def run_one(n_d, n_steps):
    """Run a single config and return result dict."""
    label = f"{n_d}-{n_d}-{n_steps}"
    json_path = os.path.join(RESULTS_DIR, f"{label}.json")

    if os.path.exists(json_path):
        print(f"\nSKIPPING (cached): {label}")
        with open(json_path) as f:
            return json.load(f)

    cmd = [
        sys.executable, 'train_tabnet.py',
        '--num-train-games', str(NUM_GAMES),
        '--n-d', str(n_d),
        '--n-a', str(n_d),
        '--n-steps', str(n_steps),
        '--batch-size', str(BATCH_SIZE),
        '--virtual-batch-size', str(VIRTUAL_BATCH_SIZE),
        '--patience', str(PATIENCE),
        '--max-epochs', str(MAX_EPOCHS),
        '--results-json', json_path,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {label} (n_d=n_a={n_d}, n_steps={n_steps})")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    wall_time = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {label}")
        return None

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        data['wall_time_s'] = wall_time
        data['label'] = label
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    return None


def generate_report(results):
    """Generate markdown report as a grid of n_d vs n_steps."""
    lines = [
        "# TabNet Hyperparameter Sweep Results\n",
        "## Setup",
        f"- **Training data:** {NUM_GAMES:,} games (~7.9M feature vectors)",
        "- **Test set:** late_tournament_games (693 games, 161K vectors)",
        "- **Train/test overlap:** Excluded from training",
        "- **Early stopping:** logloss-based, patience=3, max_epochs=20",
        f"- **Batch size:** {BATCH_SIZE}, virtual batch size: {VIRTUAL_BATCH_SIZE}",
        "- **Device:** CUDA (RTX 3090)",
        "",
    ]

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r['n_d'], r['n_steps'])] = r

    # Log Loss grid
    lines.append("## Log Loss (lower is better)\n")
    header = "| n_d/n_a |"
    sep = "|---------|"
    for ns in N_STEPS_VALUES:
        header += f" steps={ns} |"
        sep += "----------|"
    lines.append(header)
    lines.append(sep)
    for nd in N_D_VALUES:
        row = f"| **{nd}** |"
        for ns in N_STEPS_VALUES:
            r = lookup.get((nd, ns))
            if r:
                row += f" {r['log_loss']:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    # Accuracy grid
    lines.append("\n## Accuracy (higher is better)\n")
    lines.append(header)
    lines.append(sep)
    for nd in N_D_VALUES:
        row = f"| **{nd}** |"
        for ns in N_STEPS_VALUES:
            r = lookup.get((nd, ns))
            if r:
                row += f" {100*r['accuracy']:.1f}% |"
            else:
                row += " — |"
        lines.append(row)

    # Egg inversions grid
    lines.append("\n## Egg Inversions (lower is better)\n")
    lines.append(header)
    lines.append(sep)
    for nd in N_D_VALUES:
        row = f"| **{nd}** |"
        for ns in N_STEPS_VALUES:
            r = lookup.get((nd, ns))
            if r:
                row += f" {100*r['inversions']:.1f}% |"
            else:
                row += " — |"
        lines.append(row)

    # Training time grid
    lines.append("\n## Training Time\n")
    lines.append(header)
    lines.append(sep)
    for nd in N_D_VALUES:
        row = f"| **{nd}** |"
        for ns in N_STEPS_VALUES:
            r = lookup.get((nd, ns))
            if r:
                row += f" {r['train_time_s']:.0f}s (ep {r['best_epoch']}) |"
            else:
                row += " — |"
        lines.append(row)

    # Best configs
    lines.append("\n## Best Configurations\n")
    if results:
        best_ll = min(results, key=lambda r: r['log_loss'])
        best_acc = max(results, key=lambda r: r['accuracy'])
        best_inv = min(results, key=lambda r: r['inversions'])
        lines.append(f"- **Best log loss:** {best_ll['label']} ({best_ll['log_loss']:.4f})")
        lines.append(f"- **Best accuracy:** {best_acc['label']} ({100*best_acc['accuracy']:.1f}%)")
        lines.append(f"- **Fewest egg inversions:** {best_inv['label']} ({100*best_inv['inversions']:.1f}%)")
    lines.append("")

    report = "\n".join(lines)
    output_path = 'tabnet_hparam_sweep_report.md'
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport written to {output_path}")
    return report


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for n_d in N_D_VALUES:
        for n_steps in N_STEPS_VALUES:
            r = run_one(n_d, n_steps)
            if r:
                results.append(r)

    if results:
        report = generate_report(results)
        print(f"\n{'='*60}")
        print("SWEEP COMPLETE")
        print(f"{'='*60}")
        print(report)
    else:
        print("\nNo results collected.")


if __name__ == '__main__':
    main()
