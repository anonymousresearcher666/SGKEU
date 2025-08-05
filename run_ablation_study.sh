#!/bin/bash

# Script to run SGKU ablation studies
# Usage: ./run_ablation_study.sh [study_type] [dataset]
# study_type: grpo, grouping, kge_models, all

set -e  # Exit on any error

# Default parameters
STUDY_TYPE=${1:-"grpo"}
DATASET=${2:-"FB15k-237-10"}

echo "========================================="
echo "RUNNING SGKU ABLATION STUDY"
echo "Study Type: $STUDY_TYPE"
echo "Dataset: $DATASET"
echo "========================================="

# Navigate to main directory
cd src/main/

case $STUDY_TYPE in
    "grpo")
        echo "Running GRPO parameter ablation study..."
        cat > ablation_config.yaml << EOF
defaults:
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  log_path: "../../logs/"
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  epoch_num: 15
  valid_gap: 5
  lr: 1.0e-3
  timesteps_num: 3
  unlearning_method: "SGKU"
  kge: "transe"
  valid_metrics: mrr
  grouping_strategy: "relation"

datasets:
  - name: "$DATASET"
    experiments:
      - name: "grpo_ablation"
        unlearning_method: ["SGKU"]
        epsilon_grpo: [0.1, 0.2, 0.3, 0.4, 0.5]
        beta_grpo: [0.0005, 0.001, 0.002, 0.005]
        grpo_lambda: [0.3, 0.5, 0.7]
        grouping_strategy: ["relation"]
EOF
        ;;
        
    "grouping")
        echo "Running grouping strategy ablation study..."
        cat > ablation_config.yaml << EOF
defaults:
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  log_path: "../../logs/"
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  epoch_num: 15
  valid_gap: 5
  lr: 1.0e-3
  timesteps_num: 3
  unlearning_method: "SGKU"
  kge: "transe"
  valid_metrics: mrr
  epsilon_grpo: 0.3
  beta_grpo: 0.001
  grpo_lambda: 0.5

datasets:
  - name: "$DATASET"
    experiments:
      - name: "grouping_ablation"
        unlearning_method: ["SGKU"]
        grouping_strategy: ["relation", "entity", "schema", "batch"]
EOF
        ;;
        
    "kge_models")
        echo "Running KGE model comparison study..."
        cat > ablation_config.yaml << EOF
defaults:
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  log_path: "../../logs/"
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  epoch_num: 15
  valid_gap: 5
  lr: 1.0e-3
  timesteps_num: 3
  unlearning_method: "SGKU"
  valid_metrics: mrr
  epsilon_grpo: 0.3
  beta_grpo: 0.001
  grpo_lambda: 0.5
  grouping_strategy: "relation"

datasets:
  - name: "$DATASET"
    experiments:
      - name: "kge_models_comparison"
        unlearning_method: ["SGKU"]
        kge: ["transe", "rotate", "distmult", "complexe"]
EOF
        ;;
        
    "all")
        echo "Running comprehensive ablation study..."
        cat > ablation_config.yaml << EOF
defaults:
  device: mps
  data_path: "../../data/"
  unlearning_save_path: "../../checkpoint_unlearning"
  log_path: "../../logs/"
  seed: 1234
  batch_size: 1024
  emb_dim: 256
  epoch_num: 15
  valid_gap: 5
  lr: 1.0e-3
  timesteps_num: 3
  unlearning_method: "SGKU"
  valid_metrics: mrr

datasets:
  - name: "$DATASET"
    experiments:
      - name: "comprehensive_ablation"
        unlearning_method: ["SGKU"]
        kge: ["transe", "rotate"]
        epsilon_grpo: [0.2, 0.3, 0.4]
        beta_grpo: [0.001, 0.002]
        grpo_lambda: [0.5, 0.6]
        grouping_strategy: ["relation", "entity"]
EOF
        ;;
        
    *)
        echo "Unknown study type: $STUDY_TYPE"
        echo "Available options: grpo, grouping, kge_models, all"
        exit 1
        ;;
esac

echo "Starting ablation study with config:"
head -20 ablation_config.yaml
echo "..."

# Run experiments
python main.py

# Clean up temporary config
rm ablation_config.yaml

echo "========================================="
echo "ABLATION STUDY COMPLETED"
echo "Study Type: $STUDY_TYPE"
echo "Results saved to: checkpoint_unlearning/"
echo "Logs saved to: logs/"
echo "========================================="