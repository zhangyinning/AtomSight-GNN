"""
perm_data.py
============
Dataset loading and file-pairing utilities for permutation importance.
CGCNN + AtomSight version — uses CIFData and CGCNN's collate_pool.
"""

import os
import os.path as osp
import glob
import re
import numpy as np
import pandas as pd
import torch



from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from perm_config import (
    N_FEATURES, BATCH_SIZE, NUM_WORKERS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


# ============================================================================
# ATOM FEATURE NORMALIZER — copied from main.py
# Must match exactly so test features are on the same scale as training.
# ============================================================================

class AtomFeatureNormalizer:
    """
    Normalize atom feature vectors column-wise to [0, 1].
    MinMaxScaler is more appropriate than standardization for atom
    features since they have known physical bounds and the full
    element range should be represented.
    Copied from main.py to ensure identical normalization at inference time.
    """
    def __init__(self, tensor):
        # tensor shape: [n_atoms, n_features]
        self.min = tensor.min(dim=0).values   # [n_features]
        self.max = tensor.max(dim=0).values   # [n_features]
        denom = self.max - self.min
        denom[denom == 0] = 1.0  # avoid div by zero for constant dims

    def norm(self, tensor):
        denom = self.max - self.min
        denom[denom == 0] = 1.0
        return (tensor - self.min) / denom


def load_test_dataset(data_root):
    """
    Load train/val/test loaders and apply atom feature normalization,
    exactly mirroring main.py. Returns test_loader and dataset.

    Args:
        data_root: path to the CIF dataset root directory
                   (must contain id_prop.csv, atom_init_new.json, *.cif)

    Returns:
        test_loader: DataLoader for the test split
        dataset:     CIFData object (full dataset, normalized)
    """
    dataset = CIFData(data_root)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        pin_memory=False,
        train_size=None,
        val_size=None,
        test_size=None,
        return_test=True)

    # Normalize atom features to [0,1] — exact copy from main.py
    all_atom_fea = torch.tensor(
        np.array([dataset.ari.get_atom_fea(key)
                  for key in dataset.ari._embedding]),
        dtype=torch.float32
    )
    atom_fea_normalizer = AtomFeatureNormalizer(all_atom_fea)
    for key in dataset.ari._embedding:
        original = torch.tensor(dataset.ari.get_atom_fea(key),
                                dtype=torch.float32).unsqueeze(0)
        normalized = atom_fea_normalizer.norm(original).squeeze(0).numpy()
        dataset.ari._embedding[key] = normalized

    print(f"Dataset size: {len(dataset)}")
    print(f"Test loader : {len(test_loader.sampler)} crystals")
    return test_loader, dataset


def extract_timestamp(filepath):
    """Extract the long numeric timestamp from a filename."""
    match = re.search(r'(\d{10,})', osp.basename(filepath))
    return match.group(1) if match else None


def pair_models_and_csvs(model_dir, csv_dir):
    """
    Pair .pth.tar model files with their feature_importance_*.csv files
    by shared timestamp in the filename.
    Falls back to sorted-order pairing with a warning if no match found.

    Returns list of (model_path, csv_path) tuples.
    """
    # CGCNN saves as model_best_<timestamp>.pth.tar
    pth_files = sorted(glob.glob(osp.join(model_dir, 'model_best_*.pth.tar')))
    csv_files = sorted(glob.glob(osp.join(csv_dir,   'feature_importance_*.csv')))

    if not pth_files:
        raise FileNotFoundError(f"No model_best_*.pth.tar files found in: {model_dir}")
    if not csv_files:
        raise FileNotFoundError(f"No feature_importance_*.csv files in: {csv_dir}")

    pth_by_ts  = {extract_timestamp(f): f for f in pth_files if extract_timestamp(f)}
    csv_by_ts  = {extract_timestamp(f): f for f in csv_files if extract_timestamp(f)}
    shared_ts  = sorted(set(pth_by_ts) & set(csv_by_ts))

    if shared_ts:
        pairs = [(pth_by_ts[ts], csv_by_ts[ts]) for ts in shared_ts]
        print(f"Paired {len(pairs)} model/CSV pairs by timestamp.")
    else:
        n     = min(len(pth_files), len(csv_files))
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
    df      = pd.read_csv(csv_path)
    weights = np.zeros(N_FEATURES)
    for _, row in df.iterrows():
        idx           = int(str(row['feature']).replace('feat_', ''))
        weights[idx]  = float(row['attn_weight'])
    return weights


def permute_feature_globally(dataset, test_indices, feat_idx, rng=None):
    """
    Return a list of modified (atom_fea, nbr_fea, nbr_fea_idx) tuples where
    feature column feat_idx is globally shuffled across ALL atoms in the
    entire test set.

    Global permutation matches the scope of CEAL's global attention weights:
    it breaks the dataset-level statistical relationship between this feature
    and the target, not just local within-crystal patterns.

    Args:
        dataset:      CIFData object
        test_indices: list of indices into dataset for the test split
        feat_idx:     feature column index to permute
        rng:          np.random.Generator for reproducibility (optional)

    Returns:
        list of ((atom_fea_permuted, nbr_fea, nbr_fea_idx), target, cif_id)
        in the same order as test_indices
    """
    # Load all test samples first
    test_samples = [dataset[i] for i in test_indices]

    # Collect the target feature column across all atoms in all crystals
    col_values = torch.cat([s[0][0][:, feat_idx] for s in test_samples])

    if rng is not None:
        perm = torch.from_numpy(rng.permutation(len(col_values))).long()
    else:
        perm = torch.randperm(len(col_values))
    col_shuffled = col_values[perm]

    permuted = []
    offset   = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id in test_samples:
        n         = atom_fea.shape[0]
        af_new    = atom_fea.clone()
        af_new[:, feat_idx] = col_shuffled[offset:offset + n]
        offset   += n
        permuted.append(((af_new, nbr_fea, nbr_fea_idx), target, cif_id))

    return permuted