"""
This is a simple script to compute metrics from the precomputed
predictions of our models
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional
from pathlib import Path

_DEBUG_LOCAL = False

def main():
    parser = argparse.ArgumentParser("Model evaluation")
    parser.add_argument(
        "dataset",
        choices=["nyu-depth-v2-ma-downsampled", "kitti-official"],
        help="dataset to load predictions from",
    )
    parser.add_argument("density", type=str_or_int, help="density used")
    args = parser.parse_args()

    samples = torch.hub.load(
        "andreaconti/sparsity-agnostic-depth-completion" if not _DEBUG_LOCAL
        else str(Path(__file__).parent),
        args.dataset.replace("-", "_") + "_precomputed",
        args.density,
        in_memory=True,
        source="github" if not _DEBUG_LOCAL else "local",
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    metrics = []
    if args.dataset == "kitti-official":
        used_metrics = ["rmse", "mae"]
    else:
        used_metrics = ["rel", "rmse"]

    for ex in tqdm(samples):
        metrics.append(
            compute_metrics(
                to_device(ex["pred"], device),
                to_device(ex["gt"], device),
                used_metrics,
                max_depth=80. if args.dataset == "kitti-official" else 8,
            )
        )

    results = pd.DataFrame(metrics)
    print(results.mean())


## utilities


def to_device(t: np.ndarray, device):
    return torch.from_numpy(t).to(device)


def compute_metrics(pred, gt, metrics=["rmse", "rel", "mae"], max_depth: Optional[float] = None):
    mask = gt > 0
    if max_depth is not None:
        mask = mask & (gt <= max_depth)
    out = {}
    diff = torch.abs(pred[mask] - gt[mask])
    if "rmse" in metrics:
        out["rmse"] = torch.sqrt(torch.mean(torch.square(diff))).cpu().item()
    if "mae" in metrics:
        out["mae"] = torch.mean(diff).cpu().item()
    if "rel" in metrics:
        out["rel"] = torch.mean(diff / gt[mask]).cpu().item()
    return out


def str_or_int(i):
    try:
        return int(i)
    except ValueError:
        return str(i)


if __name__ == "__main__":
    main()
