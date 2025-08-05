#!/bin/bash

# Script to run SGKU baseline experiments
# Usage: ./run_sgku_baseline.sh [dataset]

set -e  # Exit on any error

# Default parameters
DATASET=${1:-"FB15k-237-10"}

echo "========================================="
echo "RUNNING SGKU BASELINE EXPERIMENTS"
echo "Dataset: $DATASET"
echo "========================================="

# Navigate to main directory
cd src/main/

# Create baseline configuration
cat > sgku_baseline_config.yaml << EOF
defaults:
  # Global settings
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  pretrain_save_path: "../../checkpoint_pretrain"
  log_path: "../../logs/"
  
  # Training parameters
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  margin: 8.0
  epoch_num: 15
  valid_gap: 5
  lr: 1.0e-3
  patience: 10
  
  # Model settings
  timesteps_num: 3
  unlearning_method: "SGKU"
  kge: "transe"
  valid_metrics: mrr
  neg_ratio: 15
  begin_unlearning: true
  
  # SGKU parameters
  epsilon_grpo: 0.3
  beta_grpo: 0.001
  grpo_frequency: 2
  group_size_grpo: 128
  grpo_lambda: 0.5
  preservation_lambda: 0.5
  grouping_strategy: "relation"
  
  # Schema parameters
  weight_schema: 1.0
  weight_preference: 0.5
  weight_contrastive: 1.0
  weight_regularization: 0.5

datasets:
  - name: "$DATASET"
    experiments:
      - name: "sgku_baseline"
        unlearning_method: ["SGKU"]
        epsilon_grpo: [0.2, 0.3, 0.4]
        beta_grpo: [0.001, 0.002]
        grpo_lambda: [0.5, 0.6]
        grouping_strategy: ["relation"]
EOF

echo "Starting SGKU baseline experiments with config:"
echo "Dataset: $DATASET"
echo "Parameter combinations: epsilon_grpo=[0.2,0.3,0.4], beta_grpo=[0.001,0.002], grpo_lambda=[0.5,0.6]"

# Run experiments
python main.py

# Clean up temporary config
rm sgku_baseline_config.yaml

echo "========================================="
echo "SGKU BASELINE EXPERIMENTS COMPLETED"
echo "Results saved to: checkpoint_unlearning/"
echo "Logs saved to: logs/"
echo "========================================="