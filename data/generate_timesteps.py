#!/usr/bin/env python3
"""
Dataset Snapshot Generator for Knowledge Graph Unlearning

This script generates N progressive unlearning snapshots from a knowledge graph dataset.
Each timestep removes a fixed percentage of the remaining triples.
"""

import random
import os
from typing import List, Tuple, Set
import logging

# Add this at the top of the file after imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR)  # Go up one level from generate_snapshots.py



# === USER-CONFIGURABLE VARIABLES ===
DATASET = "fb15k-237-20"  # Dataset folder name (e.g., "fb15k-237-10", "fb15k-237-20", "wn18rr-10", "wn18rr-20")
N_STEPS = 3  # Number of timesteps to generate
PERCENTAGE = 10  # Percentage of triples to unlearn at each step
RANDOM_SEED = 42  # Random seed for reproducibility

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_triples(filename: str) -> List[str]:
    """Load triples from a text file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            triples = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(triples)} triples from {filename}")
        return triples
    except FileNotFoundError:
        logger.error(f"File {filename} not found!")
        raise
    except Exception as e:
        logger.error(f"Error loading file {filename}: {e}")
        raise


def parse_triple(triple_str: str) -> Tuple[str, str, str]:
    """Parse a triple string into head, relation, tail components."""
    if '\t' in triple_str:
        parts = triple_str.split('\t')
    else:
        parts = triple_str.split()

    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Invalid triple format: {triple_str}")


def get_entities_and_relations(triples: List[str]) -> Tuple[Set[str], Set[str]]:
    """Extract unique entities and relations from triples."""
    entities = set()
    relations = set()

    for triple in triples:
        try:
            head, relation, tail = parse_triple(triple)
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
        except ValueError as e:
            logger.warning(f"Skipping invalid triple: {e}")

    return entities, relations


def generate_timesteps():
    """Generate N timesteps with progressive unlearning using configured variables."""

    # Set random seed
    random.seed(RANDOM_SEED)

    # Construct paths
    input_file = os.path.join(BASE_DIR, DATASET, "triples.txt")
    output_dir = os.path.join(BASE_DIR, DATASET, "timesteps")

    # Load triples
    logger.info(f"Loading triples from {input_file}")
    triples = load_triples(input_file)

    # Calculate statistics
    total_triples = len(triples)
    entities, relations = get_entities_and_relations(triples)
    num_entities = len(entities)
    num_relations = len(relations)

    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Dataset Statistics:")
    logger.info(f"  Total triples: {total_triples:,}")
    logger.info(f"  Entities: {num_entities:,}")
    logger.info(f"  Relations: {num_relations}")

    # Calculate unlearning amounts
    unlearn_per_step = int(total_triples * (PERCENTAGE / 100))
    logger.info(f"Unlearning {PERCENTAGE}% = {unlearn_per_step:,} triples per step")

    # Validate that we have enough triples
    max_unlearnable = unlearn_per_step * N_STEPS
    if max_unlearnable > total_triples:
        logger.warning(f"Cannot unlearn {PERCENTAGE}% for {N_STEPS} steps: "
                       f"would need {max_unlearnable:,} triples but only have {total_triples:,}")
        logger.warning("Will proceed and stop when all triples are unlearned")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a copy of triples list for manipulation
    remaining_triples = triples.copy()
    random.shuffle(remaining_triples)  # Shuffle for random selection

    # Keep track of cumulative unlearned triples
    all_unlearned = []

    # Generate timesteps
    for step in range(0, N_STEPS ):
        # Select triples to unlearn at this step
        if len(remaining_triples) >= unlearn_per_step:
            # Remove triples for unlearning
            unlearned_this_step = remaining_triples[:unlearn_per_step]
            remaining_triples = remaining_triples[unlearn_per_step:]
        else:
            # If not enough triples left, unlearn all remaining
            unlearned_this_step = remaining_triples
            remaining_triples = []
            logger.warning(f"Step {step}: Only {len(unlearned_this_step)} triples left to unlearn")

        # Add to cumulative unlearned
        all_unlearned.extend(unlearned_this_step)

        # Create output filename
        filename = os.path.join(output_dir, f"{step}.txt")

        # Write remaining triples to timestep file
        with open(filename, 'w', encoding='utf-8') as f:
            for triple in remaining_triples:
                f.write(triple + '\n')

        # Log statistics for this step
        num_unlearned_this_step = len(unlearned_this_step)
        num_unlearned_total = len(all_unlearned)
        num_remaining = len(remaining_triples)

        logger.info(f"Timestep {step}:")
        logger.info(f"  Unlearned this step: {num_unlearned_this_step:,} triples")
        logger.info(f"  Total unlearned: {num_unlearned_total:,} triples")
        logger.info(f"  Remaining: {num_remaining:,} triples")
        logger.info(f"  Saved to: {filename}")

        # Write statistics file
        stats_filename = os.path.join(output_dir, "statistics.txt")
        with open(stats_filename, 'a' if step > 1 else 'w', encoding='utf-8') as f:
            if step == 1:
                # Write header on first step
                f.write(f"Dataset Statistics for {DATASET}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Original dataset:\n")
                f.write(f"  Input file: {input_file}\n")
                f.write(f"  Entities: {num_entities:,}\n")
                f.write(f"  Relations: {num_relations}\n")
                f.write(f"  Total triples: {total_triples:,}\n")
                f.write(f"  Unlearning percentage: {PERCENTAGE}%\n")
                f.write(f"  Triples per step: {unlearn_per_step:,}\n")
                f.write(f"  Total timesteps: {N_STEPS}\n\n")

            f.write(f"Timestep {step}:\n")
            f.write(f"  Unlearned this step: {num_unlearned_this_step:,}\n")
            f.write(f"  Total unlearned: {num_unlearned_total:,}\n")
            f.write(f"  Remaining triples: {num_remaining:,}\n\n")

        # Also save unlearned triples for this timestep (optional)
        unlearned_filename = os.path.join(output_dir, f"unlearned_{step}.txt")
        with open(unlearned_filename, 'w', encoding='utf-8') as f:
            for triple in unlearned_this_step:
                f.write(triple + '\n')

        # Break if no triples left
        if not remaining_triples:
            logger.warning(f"No triples remaining after timestep {step}")
            break

    logger.info("Timestep generation completed successfully!")
    logger.info(f"Output files saved in '{output_dir}' directory")


if __name__ == "__main__":
    # Simply run the function with the configured variables
    generate_timesteps()