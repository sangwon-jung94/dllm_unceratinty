#!/bin/bash
# Run Winogrande blank uncertainty analysis
# This script analyzes uncertainty at blank positions in Winogrande sentences

set -e

#############################################
# Configuration - Edit these values directly
#############################################
MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
DEVICE="cuda:4"
NUM_SAMPLES=1267
MC_SAMPLES=8
DROPOUT_P=0.1
USE_ENSEMBLE=true    # true: EnsembleLLaDA, false: MC Dropout
OUTPUT_DIR="./result/winogrande_blank_uncertainty"
EXISTING_RESULTS_DIR="./result/winogrande"
RUN_POSTHOC=true
#############################################

echo "=============================================="
echo "Winogrande Blank Uncertainty Analysis"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Samples: $NUM_SAMPLES"
echo "MC Samples: $MC_SAMPLES"
echo "Dropout: $DROPOUT_P"
echo "Use Ensemble: $USE_ENSEMBLE"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Build command
CMD="python analyze_winogrande_blank_uncertainty.py \
    --model_path $MODEL_PATH \
    --device $DEVICE \
    --num_samples $NUM_SAMPLES \
    --mc_samples $MC_SAMPLES \
    --dropout_p $DROPOUT_P \
    --output_dir $OUTPUT_DIR \
    --integrate_results \
    --existing_results_dir $EXISTING_RESULTS_DIR"

# Add ensemble flag if enabled
if [ "$USE_ENSEMBLE" = "true" ]; then
    CMD="$CMD --use_ensemble"
fi

eval $CMD

# Run post-hoc analysis if requested
if [ "${RUN_POSTHOC:-true}" = "true" ]; then
    # Find the latest result directory
    LATEST_DIR=$(ls -td "$OUTPUT_DIR"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "${LATEST_DIR}blank_uncertainty_data.json" ]; then
        echo ""
        echo "Running post-hoc analysis..."
        python analyze_winogrande_posthoc.py \
            --uncertainty_data "${LATEST_DIR}blank_uncertainty_data.json" \
            --integrated_results "${LATEST_DIR}integrated_analysis.json" \
            --output_dir "${LATEST_DIR}posthoc"
    fi
fi

echo ""
echo "Analysis complete!"
