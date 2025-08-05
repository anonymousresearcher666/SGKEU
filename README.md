# KGUNLEARNING: Schema-Guided Knowledge Unlearning for Knowledge Graphs

A PyTorch implementation of Schema-Guided Knowledge Unlearning (SGKU) for knowledge graphs, designed to selectively forget specific knowledge while preserving the overall model performance.

## Overview

This project implements SGKU, a novel approach for knowledge unlearning in Knowledge Graph Embeddings (KGE). The system supports multiple KGE models (TransE, RotatE, DistMult, ComplexE) and uses schema-guided optimization to efficiently unlearn targeted knowledge from pre-trained models.

### Key Features

- **Schema-Guided Unlearning**: Uses knowledge graph schemas to guide the unlearning process
- **Multiple KGE Models**: Support for TransE, RotatE, DistMult, and ComplexE
- **Gradient-Guided Optimization**: Advanced optimization techniques for selective forgetting
- **Temporal Knowledge Graphs**: Support for time-evolving knowledge graphs
- **Comprehensive Evaluation**: Includes both forget and retain performance metrics

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PyYAML
- PrettyTable

Install dependencies:
```bash
pip install torch numpy pyyaml prettytable
```

## Dataset Preparation

1. Extract the provided datasets in the `data/` folder:
   ```bash
   cd data/
   unzip fb15k-237-10.zip
   unzip fb15k-237-20.zip
   unzip wn18rr-10.zip
   unzip wn18rr-20.zip
   ```

2. **Generate temporal timesteps** (Required first step):
   ```bash
   cd data/
   python generate_timesteps.py
   ```
   This script creates the temporal timesteps needed for the unlearning process.

3. **Generate schema stores** (Required second step):
   ```bash
   cd data/
   python generate_schema_store.py
   ```
   This script builds the schema knowledge stores that guide the unlearning process.

4. The datasets include:
   - **FB15k-237**: Subset of Freebase knowledge graph
   - **WN18RR**: WordNet knowledge graph
   - Both available in 10 and 20 timestep versions

**Note**: Steps 2 and 3 are essential before running any experiments as they prepare the temporal data structure and schema information required by SGKU.

## Usage

### 1. Pretraining

First, pretrain a KGE model:
```bash
cd src/main/
python pretrain_model.py
```

### 2. Schema-Guided Unlearning

Run the main unlearning experiments:
```bash
cd src/main/
python main.py
```

The script automatically reads configuration from `hyperparameters.yaml` and runs all specified experiments.

### 3. Experiment Scripts

Several shell scripts are provided for convenient execution of different experiment configurations:

```bash
# Make scripts executable
chmod +x *.sh

# Run pretraining on a specific dataset
./run_pretrain.sh fb15k-237-10 transe

# Run SGKU baseline experiments
./run_sgku_baseline.sh FB15k-237-10

# Run ablation studies
./run_ablation_study.sh grpo FB15k-237-10      # GRPO parameter study
./run_ablation_study.sh grouping FB15k-237-10  # Grouping strategy study
./run_ablation_study.sh kge_models FB15k-237-10 # KGE model comparison
./run_ablation_study.sh all FB15k-237-10       # Comprehensive study

# Run method comparison (SGKU vs baselines)
./run_comparison.sh FB15k-237-10

# Quick test for debugging
./run_quick_test.sh FB15k-237-10 SGKU
```

## Configuration

Hyperparameters are configured in `src/main/hyperparameters.yaml`:

### Key Parameters

- **Model Settings**:
  - `unlearning_method`: "SGKU", "pretrain", or "finetune"
  - `kge`: Knowledge graph embedding model ("transe", "rotate", "distmult", "complexe")
  - `emb_dim`: Embedding dimension (default: 256)

- **Training Parameters**:
  - `epoch_num`: Number of training epochs (default: 15)
  - `batch_size`: Batch size (default: 1024)
  - `lr`: Learning rate (default: 1e-3)

- **SGKU-Specific Parameters**:
  - `epsilon_grpo`: Clipping parameter for GRPO (default: 0.3)
  - `beta_grpo`: KL divergence coefficient (default: 0.001)
  - `grpo_lambda`: GRPO loss weight (default: 0.5)
  - `grouping_strategy`: Triple grouping strategy ("relation", "entity", "schema", "batch")

### Experiment Configuration

The YAML file supports running multiple experiments with different parameter combinations:

```yaml
datasets:
  - name: "FB15k-237-10"
    experiments:
      - name: "sgku_baseline"
        unlearning_method: ["SGKU"]
        epsilon_grpo: [0.2, 0.3, 0.4]
        beta_grpo: [0.001, 0.002]
```

## Project Structure

```
KGUNLEARNING/
├── src/
│   ├── main/
│   │   ├── main.py              # Main experiment runner
│   │   ├── pretrain_model.py    # Model pretraining
│   │   └── hyperparameters.yaml # Configuration file
│   ├── loading/
│   │   ├── KG.py               # Knowledge graph data loader
│   │   └── loader.py           # Data loading utilities
│   ├── model/
│   │   ├── SGKU.py             # Schema-guided unlearning model
│   │   ├── Retrain.py          # Baseline retraining model
│   │   └── kge_models/         # KGE model implementations
│   ├── runners/
│   │   ├── trainer.py          # Training logic
│   │   └── tester.py           # Evaluation logic
│   └── utilities/
│       └── utilities.py        # Utility functions
├── data/                       # Datasets
├── checkpoint_unlearning/      # Model checkpoints
└── logs/                      # Training logs
```

## Evaluation Metrics

The system evaluates both:
- **Forget Performance**: How well the model forgets targeted knowledge (lower is better)
- **Retain Performance**: How well the model preserves non-targeted knowledge (higher is better)

Metrics include:
- Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10

## Output

Results are logged to:
- Console output with training progress
- Log files in `logs/` directory
- Model checkpoints in `checkpoint_unlearning/`

Example output format:
```
Timestep | Time | MRR+ | Hits@1+ | Hits@10+ | MRR- | Hits@1- | Hits@10-
---------|------|------|---------|----------|------|---------|----------
0        | 45.2 | 0.324| 0.251   | 0.467    | 0.123| 0.089   | 0.134
```



## License

Please **DO NOT** redistribute this code.