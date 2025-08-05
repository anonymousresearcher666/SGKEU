#!/bin/bash

# Script for quick testing and debugging
# Usage: ./run_quick_test.sh [dataset] [method]

set -e  # Exit on any error

# Default parameters
DATASET=${1:-"FB15k-237-10"}
METHOD=${2:-"SGKU"}

echo "========================================="
echo "RUNNING QUICK TEST"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "Note: Debug mode enabled - limited epochs"
echo "========================================="

# Navigate to main directory
cd src/main/

# Create quick test configuration with minimal parameters
cat > quick_test_config.yaml << EOF
defaults:
  # Global settings
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  pretrain_save_path: "../../checkpoint_pretrain"
  log_path: "../../logs/"
  
  # Quick test parameters (reduced for speed)
  seed: 1234
  batch_size: 512  # Smaller batch size for faster testing
  emb_dim: 128     # Smaller embedding dimension
  margin: 8.0
  epoch_num: 3     # Very few epochs for quick testing
  valid_gap: 1     # Validate every epoch
  lr: 1.0e-3
  patience: 5
  
  # Model settings
  timesteps_num: 2  # Fewer timesteps for quick testing
  unlearning_method: "$METHOD"
  kge: "transe"
  valid_metrics: mrr
  neg_ratio: 10     # Fewer negative samples
  begin_unlearning: true
  debug: true       # Enable debug mode
  
  # SGKU parameters (if applicable)
  epsilon_grpo: 0.3
  beta_grpo: 0.001
  grpo_lambda: 0.5
  grouping_strategy: "relation"

datasets:
  - name: "$DATASET"
    experiments:
      - name: "quick_test"
        unlearning_method: ["$METHOD"]
EOF

echo "Starting quick test with config:"
echo "Method: $METHOD"
echo "Epochs: 3, Timesteps: 2, Batch size: 512"
echo "This should complete quickly for debugging purposes."

# Run quick test
python main.py

# Clean up temporary config
rm quick_test_config.yaml

echo "========================================="
echo "QUICK TEST COMPLETED"
echo "Check logs/ for detailed output"
echo "========================================="