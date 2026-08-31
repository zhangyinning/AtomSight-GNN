"""
perm_compute.py
===============
Model loading, MAE evaluation, per-model permutation importance,
Spearman rho computation, and CSV saving.

Plotting is handled separately in perm_plot.py after results are saved.
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.atomsight_ceal import MyCEALNetwork
from perm_config import (
    FEATURE_LABELS, N_FEATURES, SELECTED_INDICES,
    OUT_CHANNELS, AGGREGATORS, SCALERS, EDGE_DIM,
    TOWERS, NUM_LAYERS, PRE_LAYERS, POST_LAYERS,
    BATCH_SIZE, NUM_WORKERS,
)
from perm_data import permute_feature_globally, load_ceal_weights_from_csv


# ============================================================================
# MODEL
# ============================================================================

def build_and_load_model(in_channels, deg, device, model_path):
    """Instantiate MyCEALNetwork and load a saved state dict."""
    model = MyCEALNetwork(
        in_channels,
        OUT_CHANNELS,
        AGGREGATORS,
        SCALERS,
        deg,
        edge_dim=EDGE_DIM,
        towers=TOWERS,
        numLayers=NUM_LAYERS,
        pre_layers=PRE_LAYERS,
        post_layers=POST_LAYERS,
        divide_input=False,
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_mae(model, dataset, device):
    """Compute MAE over a dataset list."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS)
    total_error  = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        out  = model(data)
        total_error  += (out.squeeze() - data.y).abs().sum().item()
        total_graphs += data.num_graphs
    return total_error / total_graphs


# ============================================================================
# PER-MODEL PERMUTATION IMPORTANCE
# ============================================================================

def run_one_model(model, test_dataset, device, csv_path, n_repeats, model_seed):
    """
    Run permutation importance for a single model.

    Process:
      1. Compute baseline MAE on the unmodified test set
      2. For each of the 40 features:
           - Shuffle that feature globally across all atoms (n_repeats times)
           - Record the MAE after each shuffle
           - importance[feat] = mean(permuted MAE - baseline MAE)
      3. Compute Spearman rho between importance and CEAL attention weights

    Args:
        model:        loaded MyCEALNetwork in eval mode
        test_dataset: list of PyG Data objects
        device:       torch device
        csv_path:     path to this model's feature_importance_*.csv
        n_repeats:    number of shuffle repeats per feature
        model_seed:   integer seed — keeps shuffles reproducible per model

    Returns:
        importance:   np.array (N_FEATURES,) — mean DELTA-MAE per feature
        ceal_weights: np.array (N_FEATURES,) — CEAL attention weights
        rho:          float — Spearman rho between importance and ceal_weights
        pval:         float — p-value for rho
    """
    baseline_mae = evaluate_mae(model, test_dataset, device)
    ceal_weights = load_ceal_weights_from_csv(csv_path)
    print(f"  Baseline MAE: {baseline_mae:.5f} eV/atom")

    rng        = np.random.default_rng(seed=model_seed * 1000)
    importance = np.zeros(N_FEATURES)

    for feat_idx in range(N_FEATURES):
        repeat_deltas = []
        for _ in range(n_repeats):
            perm_data = permute_feature_globally(test_dataset, feat_idx, rng=rng)
            perm_mae  = evaluate_mae(model, perm_data, device)
            repeat_deltas.append(perm_mae - baseline_mae)
        importance[feat_idx] = np.mean(repeat_deltas)

        marker = ' <- CEAL top-13' if feat_idx in SELECTED_INDICES else ''
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
