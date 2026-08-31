"""
perm_main.py
============
Entry point for permutation importance computation.

Orchestrates:
  1. Load test dataset once
  2. Pair .pth model files with their feature_importance CSVs
  3. For each model: run permutation importance + compute Spearman rho
  4. Aggregate results across all models
  5. Save permutation_importance.csv and spearman_rho_summary.csv

Plotting is done separately in perm_plot.py after reviewing the CSVs.

Usage:
    python perm_main.py \
        --model_dir  ./Results/3_14 \
        --csv_dir    ./Results/3_14 \
        --output_dir ./Results/permutation \
        --n_repeats  10
"""

import os
import argparse
import numpy as np
import torch

from perm_data import load_test_dataset, pair_models_and_csvs
from perm_compute import build_and_load_model, run_one_model, save_csvs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test dataset once — reused across all models
    print("\n--- Loading test dataset ---")
    test_dataset, in_channels, deg = load_test_dataset()

    # 2. Pair model .pth files with their feature_importance CSVs
    print("\n--- Pairing models with CSVs ---")
    pairs    = pair_models_and_csvs(args.model_dir, args.csv_dir)
    n_models = len(pairs)

    # 3. Per-model permutation importance
    importance_matrix = np.zeros((n_models, 40))  # DELTA-MAE per model per feature
    ceal_matrix       = np.zeros((n_models, 40))  # CEAL weights per model per feature
    spearman_rhos     = []
    spearman_pvals    = []

    for model_idx, (model_path, csv_path) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Model {model_idx + 1}/{n_models}: {model_path}")

        model = build_and_load_model(in_channels, deg, device, model_path)

        importance, ceal_weights, rho, pval = run_one_model(
            model        = model,
            test_dataset = test_dataset,
            device       = device,
            csv_path     = csv_path,
            n_repeats    = args.n_repeats,
            model_seed   = model_idx,
        )

        importance_matrix[model_idx] = importance
        ceal_matrix[model_idx]       = ceal_weights
        spearman_rhos.append(rho)
        spearman_pvals.append(pval)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Aggregate and print summary
    rho_mean = float(np.mean(spearman_rhos))
    rho_std  = float(np.std(spearman_rhos))

    print(f"\n{'='*60}")
    print(f"DONE — {n_models} models completed")
    print(f"  Spearman rho = {rho_mean:.4f} +/- {rho_std:.4f}")
    print(f"  Individual:    {[f'{r:.3f}' for r in spearman_rhos]}")
    print(f"{'='*60}")

    # 5. Save CSVs
    print(f"\n--- Saving results to {args.output_dir} ---")
    save_csvs(importance_matrix, ceal_matrix,
              spearman_rhos, spearman_pvals, pairs, args.output_dir)

    print(f"\nNext step: run perm_plot.py --results_dir {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Permutation importance computation for AtomSight-GNN')
    parser.add_argument(
        '--model_dir', type=str, default='./Results/3_29_PNA',
        help='Directory containing the 10 saved .pth model files')
    parser.add_argument(
        '--csv_dir', type=str, default='./Results/3_29_PNA',
        help='Directory containing the 10 feature_importance_*.csv files')
    parser.add_argument(
        '--output_dir', type=str, default='./permutate_results',
        help='Output directory for result CSVs (default: ./Results/permutation)')
    parser.add_argument(
        '--n_repeats', type=int, default=3,
        help='Shuffle repeats per feature per model (default: 3 since the dataset is large)')
    args = parser.parse_args()
    main(args)
