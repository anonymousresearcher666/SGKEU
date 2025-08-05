#!/bin/bash

# Script to pretrain KGE models on different datasets
# Usage: ./run_pretrain.sh [dataset] [kge_model]

set -e  # Exit on any error

# Default parameters
DATASET=${1:-"fb15k-237-10"}
KGE_MODEL=${2:-"transe"}

echo "========================================="
echo "PRETRAINING KGE MODEL"
echo "Dataset: $DATASET"
echo "KGE Model: $KGE_MODEL"
echo "========================================="

# Navigate to main directory
cd src/main/

# Create a temporary config for pretraining
cat > pretrain_config.yaml << EOF
defaults:
  # Global settings
  device: mps
  data_path: "../../data/"
  pretrain_save_path: "../../checkpoint_pretrain"
  log_path: "../../logs/"
  
  # Training parameters
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  margin: 8.0
  epoch_num: 50
  valid_gap: 5
  lr: 1.0e-3
  patience: 10
  
  # Model settings
  kge: "$KGE_MODEL"
  valid_metrics: "mrr"
  neg_ratio: 15
  
  # Data settings
  data_name: "$DATASET"
EOF

echo "Starting pretraining with config:"
cat pretrain_config.yaml

# Run pretraining
python pretrain_model.py

# Clean up temporary config
rm pretrain_config.yaml

echo "========================================="
echo "PRETRAINING COMPLETED"
echo "Checkpoints saved to: checkpoint_pretrain/"
echo "Logs saved to: logs/"
echo "========================================="