#!/bin/bash

# Script to run comparison between different unlearning methods
# Usage: ./run_comparison.sh [dataset]

set -e  # Exit on any error

# Default parameters
DATASET=${1:-"FB15k-237-10"}

echo "========================================="
echo "RUNNING METHOD COMPARISON STUDY"
echo "Dataset: $DATASET"
echo "Methods: SGKU vs Pretrain vs Finetune"
echo "========================================="

# Navigate to main directory
cd src/main/

# Create comparison configuration
cat > comparison_config.yaml << EOF
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
  epoch_num: 20
  valid_gap: 5
  lr: 1.0e-3
  patience: 10
  
  # Model settings
  timesteps_num: 3
  kge: "transe"
  valid_metrics: mrr
  neg_ratio: 15
  begin_unlearning: true
  
  # SGKU parameters
  epsilon_grpo: 0.3
  beta_grpo: 0.001
  grpo_lambda: 0.5
  grouping_strategy: "relation"

datasets:
  - name: "$DATASET"
    experiments:
      - name: "method_comparison"
        unlearning_method: ["SGKU", "pretrain", "finetune"]
        
      - name: "sgku_best_params"
        unlearning_method: ["SGKU"]
        epsilon_grpo: [0.3]
        beta_grpo: [0.001]
        grpo_lambda: [0.5, 0.6]
        grouping_strategy: ["relation"]
EOF

echo "Starting method comparison experiments..."
echo "This will compare SGKU, pretrain, and finetune methods"

# Run comparison experiments
python main.py

# Clean up temporary config
rm comparison_config.yaml

echo "========================================="
echo "METHOD COMPARISON COMPLETED"
echo "Results saved to: checkpoint_unlearning/"
echo "Logs saved to: logs/"
echo "Compare the results across different methods"
echo "========================================="