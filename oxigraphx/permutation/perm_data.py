"""
perm_data.py
============
Dataset loading and file-pairing utilities for permutation importance.
"""

import os.path as osp
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data as PyGData

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import prep_datasets_from_cif
from config import config
from utils import generate_deg
from perm_config import (
    N_FEATURES, BATCH_SIZE, NUM_WORKERS,
    TRAIN_P, VAL_P, NORMALIZE_FEATURES, NORMALIZE_TARGET
)


def load_test_dataset():
    """
    Load the test split using the same parameters as training.
    Returns (test_dataset as list, in_channels, deg).
    Uses the cached .pt file so this is fast after the first run.
    """
    _, _, test_dataset, _ = prep_datasets_from_cif(
        root=config.dataset_root,
        train_p=TRAIN_P,
        val_p=VAL_P,
        normalize_features=NORMALIZE_FEATURES,
        normalize_target=NORMALIZE_TARGET
    )
    test_dataset = list(test_dataset)
    in_channels  = test_dataset[0].x.shape[1]
    deg          = generate_deg(test_dataset)

    print(f"Test set: {len(test_dataset)} crystals, {in_channels} features/atom")
    return test_dataset, in_channels, deg


def extract_timestamp(filepath):
    """Extract the long numeric timestamp from a filename."""
    match = re.search(r'(\d{10,})', osp.basename(filepath))
    return match.group(1) if match else None


def pair_models_and_csvs(model_dir, csv_dir):
    """
    Pair .pth model files with their feature_importance_*.csv files
    by shared timestamp in the filename.
    Falls back to sorted-order pairing with a warning if no match found.

    Returns list of (model_path, csv_path) tuples.
    """
    pth_files = sorted(glob.glob(osp.join(model_dir, '*.pth')))
    csv_files = sorted(glob.glob(osp.join(csv_dir, '*feature_importance.csv')))

    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in: {model_dir}")
    if not csv_files:
        raise FileNotFoundError(f"No feature_importance_*.csv files in: {csv_dir}")

    pth_by_ts = {extract_timestamp(f): f for f in pth_files if extract_timestamp(f)}
    csv_by_ts = {extract_timestamp(f): f for f in csv_files if extract_timestamp(f)}
    shared_ts = sorted(set(pth_by_ts) & set(csv_by_ts))

    if shared_ts:
        pairs = [(pth_by_ts[ts], csv_by_ts[ts]) for ts in shared_ts]
        print(f"Paired {len(pairs)} model/CSV pairs by timestamp.")
    else:
        n = min(len(pth_files), len(csv_files))
        pairs = list(zip(pth_files[:n], csv_files[:n]))
        print(f"WARNING: Could not match by timestamp. "
              f"Paired {n} files by sorted order — verify this is correct.")

    print("\nPairings:")
    for i, (pth, csv) in enumerate(pairs):
        print(f"  [{i+1}] {osp.basename(pth)}")
        print(f"       {osp.basename(csv)}")

    return pairs


def load_ceal_weights_from_csv(csv_path):
    """
    Load CEAL attention weights from a feature_importance_*.csv file.
    Returns np.array of shape (N_FEATURES,) indexed by feature index.
    """
    df = pd.read_csv(csv_path)
    weights = np.zeros(N_FEATURES)
    for _, row in df.iterrows():
        idx = int(str(row['feature']).replace('feat_', ''))
        weights[idx] = float(row['attn_weight'])
    return weights


def permute_feature_globally(dataset, feat_idx, rng=None):
    """
    Return a new dataset where feature column feat_idx is globally shuffled
    across ALL atoms in the entire test set.

    Global permutation matches the scope of CEAL's global attention weights:
    it breaks the dataset-level statistical relationship between this feature
    and the target, not just local within-crystal patterns.

    Args:
        dataset:   list of PyG Data objects
        feat_idx:  feature column index to permute
        rng:       np.random.Generator for reproducibility (optional)

    Returns:
        list of new PyG Data objects with feat_idx column shuffled
    """
    col_values = torch.cat([data.x[:, feat_idx] for data in dataset])

    if rng is not None:
        perm = torch.from_numpy(rng.permutation(len(col_values))).long()
    else:
        perm = torch.randperm(len(col_values))
    col_shuffled = col_values[perm]

    permuted = []
    offset = 0
    for data in dataset:
        n     = data.x.shape[0]
        x_new = data.x.clone()
        x_new[:, feat_idx] = col_shuffled[offset:offset + n]
        offset += n
        permuted.append(PyGData(
            x=x_new,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=data.y,
        ))
    return permuted
