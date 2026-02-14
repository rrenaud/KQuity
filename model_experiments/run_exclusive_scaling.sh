#!/bin/bash
# Run exclusive equal-states scaling experiment: 5K doubling to exhaustion
# Capacity grows with data based on prior LGB scaling results (commit bf28e14)
set -e

SCHEDULE=(
    # max_games num_leaves num_trees
    "5000 70 70"
    "10000 100 100"
    "20000 100 100"
    "40000 150 150"
    "75000 200 200"
)

for entry in "${SCHEDULE[@]}"; do
    read -r games leaves trees <<< "$entry"
    echo ""
    echo "============================================================"
    echo "  max-games=$games  leaves=$leaves  trees=$trees"
    echo "============================================================"
    python model_experiments/data_quality_experiment.py \
        --exclusive --equal-states --variance 10 \
        --max-games "$games" --num-leaves "$leaves" --num-trees "$trees" \
        --output "model_experiments/scaling_exclusive_${games}.json"
done

echo ""
echo "All scaling runs complete."
