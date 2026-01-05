#!/bin/bash

# Script to demonstrate the speed difference between standard MC Dropout and EnsembleLLaDA
# This shows how EnsembleLLaDA computes uncertainty with a SINGLE forward pass

set -e

echo "============================================================"
echo "Performance Comparison: MC Dropout vs EnsembleLLaDA"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - mc_samples: 4"
echo "  - gen_length: 32"
echo "  - steps: 16"
echo "  - num_samples: 2"
echo ""
echo "Expected behavior:"
echo "  - Standard MC Dropout: 4 forward passes per uncertainty calculation"
echo "  - EnsembleLLaDA:       1 forward pass (num_ensembles=4)"
echo "============================================================"
echo ""

# Output directory
OUTPUT_DIR="./result/ensemble_comparison"
mkdir -p "$OUTPUT_DIR"

# Common arguments
COMMON_ARGS="
  --use_prompt \
  --benchmark gsm8k \
  --num_samples 2 \
  --steps 16 \
  --gen_length 32 \
  --block_length 32 \
  --remasking uncertainty_aware \
  --mc_samples 4 \
  --alpha 1.0 \
  --beta 1.0 \
  --dropout_p 0.1
"

echo ""
echo "TEST 1: Standard MC Dropout (4 × forward passes)"
echo "------------------------------------------------------------"
time python visualize_uncertainty_by_timestep.py \
  $COMMON_ARGS \
  --exp_name "standard_mc_dropout" \
  --output_dir "$OUTPUT_DIR" 2>&1 | grep -E "(forward|Uncertainty|Loading|Generated)"

echo ""
echo ""
echo "TEST 2: EnsembleLLaDA (1 × forward pass)"
echo "------------------------------------------------------------"
time python visualize_uncertainty_by_timestep.py \
  --use_ensemble_model \
  $COMMON_ARGS \
  --exp_name "ensemble_llada" \
  --output_dir "$OUTPUT_DIR" 2>&1 | grep -E "(forward|Uncertainty|Loading|Generated|SINGLE)"

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Expected speedup: ~4x faster with EnsembleLLaDA"
echo "(actual speedup depends on model size and mc_samples)"
echo "============================================================"
