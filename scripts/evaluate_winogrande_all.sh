#!/bin/bash
# Evaluate all Winogrande result folders

MAX_SAMPLES="$1"
DATASET_CONFIG="${2:-winogrande_xl}"
SPLIT="${3:-validation}"
ROOT_DIR="./result/winogrande"

if [ -n "$MAX_SAMPLES" ]; then
    echo "Evaluating Winogrande results (max $MAX_SAMPLES samples per experiment)..."
else
    echo "Evaluating all Winogrande results..."
fi

echo "Dataset config: $DATASET_CONFIG (split: $SPLIT)"
echo "================================"

find "$ROOT_DIR" -type f -name "output.txt" -print0 | while IFS= read -r -d '' output_file; do
    result_dir=$(dirname "$output_file")
    exp_name=${result_dir#./}
    echo ""
    echo "Evaluating: $exp_name"
    echo "-------------------"

    if [ -n "$MAX_SAMPLES" ]; then
        python3 evaluate_winogrande.py \
            --output_file "$output_file" \
            --save_results "$result_dir/evaluation.json" \
            --max_samples "$MAX_SAMPLES" \
            --dataset_config "$DATASET_CONFIG" \
            --split "$SPLIT"
    else
        python3 evaluate_winogrande.py \
            --output_file "$output_file" \
            --save_results "$result_dir/evaluation.json" \
            --dataset_config "$DATASET_CONFIG" \
            --split "$SPLIT"
    fi

    echo ""
done

echo ""
echo "================================"
echo "All evaluations completed!"
echo ""
echo "Summary:"
echo "--------"

find "$ROOT_DIR" -type f -name "evaluation.json" -print0 | while IFS= read -r -d '' eval_file; do
    exp_name=$(dirname "$eval_file")
    exp_name=${exp_name#./}
    accuracy=$(EVAL_FILE="$eval_file" python3 - <<'PY'
import json, os
path = os.environ['EVAL_FILE']
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"{data.get('accuracy', 0):.2f}%")
PY
)
    echo "$exp_name: $accuracy"
done
