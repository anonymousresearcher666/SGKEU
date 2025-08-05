import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn


def get_datetime():
    """ Get current time """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Utility function to calculate unlearning metrics
def calculate_unlearning_metrics(forget_processor, reserve_processor):
    """
    Calculate unlearning metrics based on MRR values from forget and reserve processors.

    Args:
        forget_processor: Processor for the forgetting dataset
        reserve_processor: Processor for the reserved dataset

    Returns:
        dict: Dictionary with unlearning metrics (MRR_Avg and MRR_F1)
    """
    # Get MRR values
    mrr_f = forget_processor.get_mrr_f()  # MRR on forgotten data
    mrr_r = reserve_processor.get_mrr_r()  # MRR on retained data

    # Calculate metrics
    # For effective forgetting, we want to minimize MRR_f, hence (1-MRR_f)
    one_minus_mrr_f = 1.0 - mrr_f

    # Calculate MRR_Avg (balanced accuracy)
    mrr_avg = (mrr_r + one_minus_mrr_f) / 2.0

    # Calculate MRR_F1 (harmonic mean)
    denominator = mrr_r + one_minus_mrr_f
    if denominator > 0:
        mrr_f1 = (2.0 * mrr_r * one_minus_mrr_f) / denominator
    else:
        mrr_f1 = 0.0

    # Print results
    print("\n=== Unlearning Performance Metrics ===")
    print(f"Retained MRR (MRR_r): {mrr_r:.4f}")
    print(f"Forgotten MRR (MRR_f): {mrr_f:.4f}")
    print(f"Forget Success (1-MRR_f): {one_minus_mrr_f:.4f}")
    print(f"MRR_Avg: {mrr_avg:.4f}")
    print(f"MRR_F1: {mrr_f1:.4f}")

    return {
        'mrr_r': mrr_r,
        'mrr_f': mrr_f,
        'mrr_avg': mrr_avg,
        'mrr_f1': mrr_f1
    }



def set_seeds(seed):
    """ Set  seeds """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    pass
