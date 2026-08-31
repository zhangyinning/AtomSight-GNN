import os
import os.path as osp

import torch
import pandas as pd
import random
from tqdm import tqdm
import numpy as np

from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import config
from utils_data import cif_to_pyg_data, find_cif_for_mp_id, get_feature_dim


class MyCIFDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for crystal property prediction from CIF files.

    Reads a CSV file containing:
        mp_id column  (e.g., 'mp-11830')
        target column (e.g., 'formation_energy_ev')

    Loads corresponding CIF from <root>/<cif_dirname>/<mp_id>.cif and builds
    a crystal graph with:
        node features : 40-dimensional atomic encoding
                        (Identity, Size, Electronic, Oxidation, Physical)
        edges         : all atom pairs within 5 Å cutoff
        target        : y shape [1]
    """

    def __init__(
        self,
        root: str,
        cfg=config,
        normalize_features: bool = True,
        normalize_target: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.cfg = cfg
        self.normalize_features = normalize_features
        self.normalize_target = normalize_target
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        suffix = ''
        if self.normalize_features:
            suffix += '_normx'
        if self.normalize_target:
            suffix += '_normy'
        return [f'data{suffix}.pt']

    def download(self):
        pass

    def process(self):
        csv_path = osp.join(self.root, self.cfg.csv_filename)
        cif_dir  = osp.join(self.root, self.cfg.cif_dirname)

        if not osp.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not osp.isdir(cif_dir):
            raise FileNotFoundError(f"CIF directory not found: {cif_dir}")

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[self.cfg.mp_id_col, self.cfg.target_col]).copy()

        print(f"\nProcessing {len(df)} structures from CSV...")

        data_list = []
        skipped = 0
        pbar = tqdm(total=len(df), desc="Processing CIF graphs")

        for _, row in df.iterrows():
            mp_id  = str(row[self.cfg.mp_id_col]).strip()
            y_val  = float(row[self.cfg.target_col])

            cif_path = find_cif_for_mp_id(cif_dir, mp_id)
            if cif_path is None:
                skipped += 1
                pbar.update(1)
                continue

            data = cif_to_pyg_data(
                cif_path=cif_path,
                y_value=y_val,
                cutoff=float(self.cfg.neighbor_cutoff),
                use_edge_distances=bool(self.cfg.use_edge_distances),
            )
            data.mp_id = mp_id

            if self.pre_filter is not None and not self.pre_filter(data):
                skipped += 1
                pbar.update(1)
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            pbar.update(1)

        pbar.close()
        print(f"\nProcessing complete: {len(data_list)} successful, {skipped} skipped")

        if len(data_list) == 0:
            raise RuntimeError(
                f"No data was successfully processed! "
                f"Processed 0 out of {len(df)} entries. "
                f"Check your CIF files and CSV."
            )

        # ── Normalization ────────────────────────────────────────────────────
        scalers = {}

        if self.normalize_features:
            print("Normalizing node features to [0, 1]...")
            all_x = np.vstack([d.x.numpy() for d in data_list])
            x_scaler = MinMaxScaler()
            x_scaler.fit(all_x)
            scalers['x_scaler'] = x_scaler
            for d in data_list:
                d.x = torch.tensor(
                    x_scaler.transform(d.x.numpy()), dtype=torch.float32
                )
            print(f"  Feature range after normalization: "
                  f"min={data_list[0].x.numpy().min():.4f}, "
                  f"max={data_list[0].x.numpy().max():.4f}")

        if self.normalize_target:
            print("Normalizing targets (StandardScaler)...")
            all_y = np.concatenate([d.y.numpy() for d in data_list]).reshape(-1, 1)
            y_scaler = StandardScaler()
            y_scaler.fit(all_y)
            scalers['y_scaler'] = y_scaler
            print(f"  Target mean={y_scaler.mean_[0]:.4f}, "
                  f"std={y_scaler.scale_[0]:.4f}")
            for d in data_list:
                d.y = torch.tensor(
                    y_scaler.transform(d.y.numpy().reshape(-1, 1)).flatten(),
                    dtype=torch.float32,
                )

        # ── Save processed data and scalers ──────────────────────────────────
        os.makedirs(self.processed_dir, exist_ok=True)

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"\nSaved processed data to {self.processed_paths[0]}")

        if scalers:
            suffix = ''
            if self.normalize_features:
                suffix += '_normx'
            if self.normalize_target:
                suffix += '_normy'
            scalers_path = osp.join(self.processed_dir, f'scalers{suffix}.pt')
            torch.save(scalers, scalers_path)
            print(f"Saved scalers to {scalers_path}")


def split_dataset(dataset, train_p=0.70, val_p=0.15, shuffle=True):
    """
    Split a dataset into train / validation / test subsets.

    Args:
        dataset : PyG InMemoryDataset
        train_p : fraction for training   (default 0.70)
        val_p   : fraction for validation (default 0.15)
        shuffle : randomize order before splitting
    """
    n = dataset.len()
    if shuffle:
        idx = random.sample(range(n), n)
    else:
        idx = list(range(n))

    n_train = int(n * train_p)
    n_val   = int(n * val_p)

    train_ds = dataset[idx[:n_train]]
    val_ds   = dataset[idx[n_train:n_train + n_val]]
    test_ds  = dataset[idx[n_train + n_val:]]

    return train_ds, val_ds, test_ds


def prep_datasets_from_cif(
    root: str,
    train_p: float = 0.60,
    val_p: float = 0.20,
    shuffle: bool = True,
    normalize_features: bool = True,
    normalize_target: bool = True,
):
    """
    Convenience helper: create dataset and split into train / val / test.

    Args:
        root              : path to dataset root directory
        train_p           : fraction for training   (default 0.60)
        val_p             : fraction for validation (default 0.20)
        shuffle           : randomize order before splitting
        normalize_features: normalize node features to [0, 1]
        normalize_target  : normalize targets with StandardScaler

    Returns:
        train_ds, val_ds, test_ds, scalers
    """
    ds = MyCIFDataset(
        root=root,
        cfg=config,
        normalize_features=normalize_features,
        normalize_target=normalize_target,
    )

    print(f"\nTotal dataset size: {len(ds)}")
    print(f"Node feature dimension: {get_feature_dim()}")

    train_ds, val_ds, test_ds = split_dataset(
        ds, train_p=train_p, val_p=val_p, shuffle=shuffle
    )

    # Load scalers saved during processing
    suffix = ''
    if normalize_features:
        suffix += '_normx'
    if normalize_target:
        suffix += '_normy'
    scalers_path = osp.join(ds.processed_dir, f'scalers{suffix}.pt')

    if osp.exists(scalers_path):
        scalers = torch.load(scalers_path, weights_only=False)
        print(f"Loaded scalers from {scalers_path}")
    else:
        scalers = {}
        print("No scalers found (normalization may be disabled)")

    print(f"\nDataset split: "
          f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    return train_ds, val_ds, test_ds, scalers