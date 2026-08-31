"""
perm_compute.py
===============
Model loading, MAE evaluation, per-model permutation importance,
Spearman rho computation, and CSV saving.
CGCNN + AtomSight version.

Plotting is handled separately in perm_plot.py after results are saved.
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from cgcnn.data import collate_pool
from cgcnn.model_atomsight_noembedding import AtomSightGNN
from perm_config import (
    FEATURE_LABELS, N_FEATURES, SELECTED_INDICES,
    ATOM_FEA_LEN, NBR_FEA_LEN, N_CONV, H_FEA_LEN, N_H,
    BATCH_SIZE, NUM_WORKERS,
)
from perm_data import permute_feature_globally, load_ceal_weights_from_csv, load_test_dataset


# ============================================================================
# MODEL
# ============================================================================

def build_and_load_model(device, model_path):
    """
    Instantiate AtomSightGNN and load a saved checkpoint.
    CGCNN checkpoints are saved as state dicts inside a dict with key
    'state_dict', matching save_checkpoint() in main.py.
    """
    model = AtomSightGNN(
        orig_atom_fea_len = N_FEATURES,
        nbr_fea_len       = NBR_FEA_LEN,
        atom_fea_len      = ATOM_FEA_LEN,
        n_conv            = N_CONV,
        h_fea_len         = H_FEA_LEN,
        n_h               = N_H,
        classification    = False,
    )
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both raw state dict and wrapped checkpoint dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


# ============================================================================
# NORMALIZER
# ============================================================================

class Normalizer:
    """Mirrors the Normalizer class in main.py for denormalization."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    @classmethod
    def from_checkpoint(cls, checkpoint):
        nd = checkpoint['normalizer']
        return cls(mean=nd['mean'], std=nd['std'])


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_mae(model, samples, normalizer, device):
    """
    Compute MAE (in original eV/atom units) over a list of CGCNN samples.

    Args:
        model:      AtomSightGNN in eval mode
        samples:    list of ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
        normalizer: Normalizer object for denormalization
        device:     torch device
    """
    model.eval()
    loader = DataLoader(
        samples,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        collate_fn  = collate_pool,
    )
    total_error  = 0.0
    total_count  = 0

    for (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, _ in loader:
        # Move inputs to device
        atom_fea         = atom_fea.to(device)
        nbr_fea          = nbr_fea.to(device)
        nbr_fea_idx      = nbr_fea_idx.to(device)
        crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
        target           = target.to(device)

        output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        # Denormalize prediction before computing MAE
        pred_denorm   = normalizer.denorm(output.squeeze())

        total_error += (pred_denorm - target.squeeze()).abs().sum().item()
        total_count += target.shape[0]

    return total_error / total_count


# ============================================================================
# PER-MODEL PERMUTATION IMPORTANCE
# ============================================================================

def run_one_model(model, test_loader, dataset, normalizer,
                  device, csv_path, n_repeats, model_seed):
    """
    Run permutation importance for a single AtomSightGNN model.

    Process:
      1. Compute baseline MAE on the unmodified test set
      2. For each of the 40 features:
           - Shuffle that feature globally across all atoms (n_repeats times)
           - Record the MAE after each shuffle
           - importance[feat] = mean(permuted MAE - baseline MAE)
      3. Compute Spearman rho between importance and CEAL attention weights

    Args:
        model:        loaded AtomSightGNN in eval mode
        test_loader:  DataLoader for the test split (from load_test_dataset)
        dataset:      CIFData object (for permutation access)
        normalizer:   Normalizer for denormalization
        device:       torch device
        csv_path:     path to this model's feature_importance_*.csv
        n_repeats:    number of shuffle repeats per feature
        model_seed:   integer seed for reproducible shuffles per model

    Returns:
        importance:   np.array (N_FEATURES,) — mean DELTA-MAE per feature
        ceal_weights: np.array (N_FEATURES,) — CEAL attention weights
        rho:          float — Spearman rho between importance and ceal_weights
        pval:         float — p-value for rho
    """
    # Reconstruct test samples list from loader sampler indices
    test_indices = list(test_loader.sampler)
    test_samples = [dataset[i] for i in test_indices]

    baseline_mae = evaluate_mae(model, test_samples, normalizer, device)
    ceal_weights = load_ceal_weights_from_csv(csv_path)
    print(f"  Baseline MAE: {baseline_mae:.5f} eV/atom")

    rng        = np.random.default_rng(seed=model_seed * 1000)
    importance = np.zeros(N_FEATURES)

    for feat_idx in range(N_FEATURES):
        repeat_deltas = []
        for _ in range(n_repeats):
            perm_samples = permute_feature_globally(
                dataset, test_indices, feat_idx, rng=rng)
            perm_mae     = evaluate_mae(model, perm_samples, normalizer, device)
            repeat_deltas.append(perm_mae - baseline_mae)
        importance[feat_idx] = np.mean(repeat_deltas)

        marker = ' <- CEAL top' if feat_idx in SELECTED_INDICES else ''
        print(f"  feat_{feat_idx:02d}  {FEATURE_LABELS[feat_idx]:<25}  "
              f"dMAE={importance[feat_idx]:+.5f}{marker}")

    rho, pval = spearmanr(importance, ceal_weights)
    print(f"\n  Spearman rho (permutation vs CEAL): {rho:.4f}  (p={pval:.4e})")
    return importance, ceal_weights, rho, pval


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_csvs(importance_matrix, ceal_matrix, spearman_rhos, spearman_pvals,
              pairs, out_dir):
    """
    Save two CSV files:

    permutation_importance.csv
      - One row per feature
      - Columns: feature index, label, mean DELTA-MAE, std DELTA-MAE,
                 mean CEAL weight, ceal_top13 flag,
                 plus per-model DELTA-MAE and CEAL weight columns
      - Sorted by mean DELTA-MAE descending (most important first)

    spearman_rho_summary.csv
      - One row per model plus a MEAN and STD summary row
      - Columns: model filename, spearman_rho, p_value
    """
    n_models        = importance_matrix.shape[0]
    mean_importance = importance_matrix.mean(axis=0)
    std_importance  = importance_matrix.std(axis=0)
    mean_ceal       = ceal_matrix.mean(axis=0)
    rho_mean        = float(np.mean(spearman_rhos))
    rho_std         = float(np.std(spearman_rhos))

    # --- permutation_importance.csv ---
    results_df = pd.DataFrame({
        'feature_idx':      range(N_FEATURES),
        'feature_label':    FEATURE_LABELS,
        'delta_mae_mean':   mean_importance,
        'delta_mae_std':    std_importance,
        'ceal_weight_mean': mean_ceal,
        'ceal_top13':       [i in SELECTED_INDICES for i in range(N_FEATURES)],
    })
    for i in range(n_models):
        results_df[f'delta_mae_model{i+1}']   = importance_matrix[i]
        results_df[f'ceal_weight_model{i+1}'] = ceal_matrix[i]

    results_df.sort_values('delta_mae_mean', ascending=False).to_csv(
        osp.join(out_dir, 'permutation_importance.csv'), index=False)
    print("Saved: permutation_importance.csv")

    # --- spearman_rho_summary.csv ---
    pd.DataFrame({
        'model':        [osp.basename(p) for p, _ in pairs] + ['MEAN', 'STD'],
        'spearman_rho': list(spearman_rhos) + [rho_mean, rho_std],
        'p_value':      list(spearman_pvals) + [float('nan'), float('nan')],
    }).to_csv(osp.join(out_dir, 'spearman_rho_summary.csv'), index=False)
    print("Saved: spearman_rho_summary.csv")