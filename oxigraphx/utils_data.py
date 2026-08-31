"""
Utilities for reading CIF crystal structures and converting them to PyG graphs.

Node features: 40-dimensional perovskite encoding
    - Identity:  period one-hot (7), group one-hot (18), block one-hot (4)
    - Size:      atomic radius, average ionic radius (2)
    - Electronic: electronegativity, ionization energy, electron affinity,
                  valence electrons (4)
    - Oxidation: number of common oxidation states, max oxidation state (2)
    - Physical:  atomic mass, density of solid, molar volume (3)
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Tuple, Optional

import math
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure, Element


# ============================================================================
# SAFE FLOAT HELPER
# ============================================================================

def _safe_float(val, default=0.0):
    """Safely convert pymatgen FloatWithUnit or any value to float.
    Returns default if value is None, NaN, or cannot be converted."""
    if val is None:
        return default
    f = float(val)
    return default if math.isnan(f) or math.isinf(f) else f


# ============================================================================
# VALENCE ELECTRON LOOKUP
# ============================================================================

VALENCE_ELECTRONS = {
    # s-block
    1: 1,  2: 2,
    3: 1,  4: 2,
    11: 1, 12: 2,
    19: 1, 20: 2,
    37: 1, 38: 2,
    55: 1, 56: 2,
    87: 1, 88: 2,
    # p-block (s + p electrons)
    5: 3,  6: 4,  7: 5,  8: 6,  9: 7,  10: 8,
    13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
    31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,
    49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,
    81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,
    # d-block (s + d electrons)
    21: 3,  22: 4,  23: 5,  24: 6,  25: 7,
    26: 8,  27: 9,  28: 10, 29: 11, 30: 12,
    39: 3,  40: 4,  41: 5,  42: 6,  43: 7,
    44: 8,  45: 9,  46: 10, 47: 11, 48: 12,
    57: 3,  72: 4,  73: 5,  74: 6,  75: 7,
    76: 8,  77: 9,  78: 10, 79: 11, 80: 12,
    # f-block lanthanides
    58: 4,  59: 5,  60: 6,  61: 7,  62: 8,
    63: 9,  64: 10, 65: 11, 66: 12, 67: 13,
    68: 14, 69: 15, 70: 16, 71: 3,
    # actinides
    89: 3, 90: 4, 91: 5, 92: 6, 93: 7, 94: 8,
}


# ============================================================================
# ATOMIC FEATURE EXTRACTION (40-dim perovskite encoding)
# ============================================================================

def get_atom_features(atomic_number: int) -> np.ndarray:
    """
    40-dimensional atomic feature vector organized into five families:
        Identity  (29): period one-hot (7), group one-hot (18), block one-hot (4)
        Size       (2): atomic radius, average ionic radius
        Electronic (4): electronegativity, ionization energy, electron affinity,
                        valence electron count
        Oxidation  (2): number of common oxidation states, max oxidation state
        Physical   (3): atomic mass, density of solid, molar volume

    Returns: numpy array of shape (40,)
    """
    elem = Element.from_Z(atomic_number)
    ox_states = elem.common_oxidation_states if hasattr(elem, 'common_oxidation_states') else []

    # Period one-hot (7)
    period_oh = [0.0] * 7
    if 1 <= elem.row <= 7:
        period_oh[elem.row - 1] = 1.0

    # Group one-hot (18)
    group_oh = [0.0] * 18
    if 1 <= elem.group <= 18:
        group_oh[elem.group - 1] = 1.0

    # Block one-hot (4: s, p, d, f)
    block_oh = [0.0] * 4
    block_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    if elem.block in block_map:
        block_oh[block_map[elem.block]] = 1.0

    n_valence = float(VALENCE_ELECTRONS.get(atomic_number, 0))

    features = (
        period_oh +   # 7
        group_oh +    # 18
        block_oh +    # 4
        [
            # Size (2)
            _safe_float(elem.atomic_radius),
            _safe_float(elem.average_ionic_radius),
            # Electronic (4)
            _safe_float(elem.X),
            _safe_float(elem.ionization_energy),
            _safe_float(elem.electron_affinity),
            _safe_float(n_valence),
            # Oxidation (2)
            float(len(ox_states)),
            _safe_float(max(ox_states)) if ox_states else 0.0,
            # Physical (3)
            _safe_float(elem.atomic_mass),
            _safe_float(elem.density_of_solid),
            _safe_float(elem.molar_volume),
        ]
    )

    return np.array(features, dtype=np.float32)


def build_atom_features(structure) -> torch.Tensor:
    """
    Extract 40-dim feature vectors for all atoms in a structure.
    Returns: torch.Tensor of shape [num_atoms, 40]
    """
    features_list = [get_atom_features(int(site.specie.Z)) for site in structure.sites]
    return torch.tensor(np.array(features_list), dtype=torch.float32)


# ============================================================================
# EDGE CONSTRUCTION
# ============================================================================

def build_edges_with_cutoff(structure, cutoff: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges using distance cutoff (Å).
    Returns: (edge_index [2, E], edge_attr [E, 1])
    """
    neigh_lists = structure.get_all_neighbors(r=cutoff, include_index=True)

    src, dst, dist = [], [], []
    for i, neighs in enumerate(neigh_lists):
        for n in neighs:
            src.append(i)
            dst.append(int(n.index))
            dist.append(float(n.nn_distance))

    if len(src) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(dist, dtype=torch.float32).view(-1, 1)
    return edge_index, edge_attr


# ============================================================================
# CIF → PyG DATA
# ============================================================================

def cif_to_pyg_data(
    cif_path: str,
    y_value: float,
    cutoff: float = 5.0,
    use_edge_distances: bool = True,
) -> Data:
    """
    Convert a CIF file to a PyTorch Geometric Data object.

    Args:
        cif_path:           path to .cif file
        y_value:            target value (e.g., formation energy in eV/atom)
        cutoff:             neighbor cutoff radius in Å (default 5.0)
        use_edge_distances: include Gaussian-expanded bond distances as edge features

    Returns:
        Data(x [N,40], edge_index [2,E], edge_attr [E,1], y [1])
    """
    structure = Structure.from_file(cif_path)
    x = build_atom_features(structure)
    edge_index, edge_attr = build_edges_with_cutoff(structure, cutoff=cutoff)
    y = torch.tensor([float(y_value)], dtype=torch.float32)

    if use_edge_distances:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        data = Data(x=x, edge_index=edge_index, y=y)

    data.cif_path = cif_path
    return data


# ============================================================================
# FEATURE DIMENSION
# ============================================================================

def get_feature_dim() -> int:
    """Returns the number of node features per atom (always 40)."""
    return 40


# ============================================================================
# CIF FILE LOOKUP
# ============================================================================

def find_cif_for_mp_id(cif_dir: str, mp_id: str) -> Optional[str]:
    """
    Find the CIF file for a given Materials Project ID.
    Expects filenames of the form: mp-12345.cif
    """
    candidate = osp.join(cif_dir, f"{mp_id}.cif")
    if osp.exists(candidate):
        return candidate

    # Fallback: any file starting with mp_id
    for fn in os.listdir(cif_dir):
        if fn.lower().endswith(".cif") and fn.startswith(mp_id):
            return osp.join(cif_dir, fn)

    return None