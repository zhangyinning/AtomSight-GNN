"""
generate_atom_init.py
=====================

Usage:
    python generate_atom_init.py
It generates the atom_init file used by cgcnn model for atoms from 1 to 100, but with new features different from cgcnn's original all-one-hot-encoded atom features.
"""

import json
import math
import numpy as np
from pymatgen.core import Element


def _safe_float(val, default=0.0):
    if val is None:
        return default
    f = float(val)
    return default if math.isnan(f) or math.isinf(f) else f


def get_perovskite_features(atomic_number):
    elem      = Element.from_Z(atomic_number)
    ox_states = elem.common_oxidation_states if hasattr(elem, 'common_oxidation_states') else []

    row_onehot = [0.0] * 7
    if 1 <= elem.row <= 7:
        row_onehot[elem.row - 1] = 1.0

    group_onehot = [0.0] * 18
    if 1 <= elem.group <= 18:
        group_onehot[elem.group - 1] = 1.0

    block_onehot = [0.0] * 4
    block_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    if elem.block in block_map:
        block_onehot[block_map[elem.block]] = 1.0

    VALENCE_ELECTRONS = {
        # s-block
        1: 1,  2: 2,                                                         # H, He
        3: 1,  4: 2,                                                         # Li, Be
        11: 1, 12: 2,                                                        # Na, Mg
        19: 1, 20: 2,                                                        # K, Ca
        37: 1, 38: 2,                                                        # Rb, Sr
        55: 1, 56: 2,                                                        # Cs, Ba
        87: 1, 88: 2,                                                        # Fr, Ra
        # p-block (s + p electrons)
        5: 3,  6: 4,  7: 5,  8: 6,  9: 7,  10: 8,                         # B-Ne
        13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,                         # Al-Ar
        31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8,                         # Ga-Kr
        49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8,                         # In-Xe
        81: 3, 82: 4, 83: 5, 84: 6, 85: 7, 86: 8,                         # Tl-Rn
        # d-block (s + d electrons)
        21: 3,  22: 4,  23: 5,  24: 6,  25: 7,                            # Sc-Mn
        26: 8,  27: 9,  28: 10, 29: 11, 30: 12,                           # Fe-Zn
        39: 3,  40: 4,  41: 5,  42: 6,  43: 7,                            # Y-Tc
        44: 8,  45: 9,  46: 10, 47: 11, 48: 12,                           # Ru-Cd
        57: 3,  72: 4,  73: 5,  74: 6,  75: 7,                            # La, Hf-Re
        76: 8,  77: 9,  78: 10, 79: 11, 80: 12,                           # Os-Hg
        # f-block lanthanides (s + f electrons)
        58: 4,  59: 5,  60: 6,  61: 7,  62: 8,                            # Ce-Sm
        63: 9,  64: 10, 65: 11, 66: 12, 67: 13,                           # Eu-Ho
        68: 14, 69: 15, 70: 16, 71: 3,                                     # Er-Yb, Lu
        # actinides
        89: 3,  90: 4,  91: 5,  92: 6,  93: 7,  94: 8,                   # Ac-Pu
        # heavy actinides (Am-Fm) — predominantly +3 like lanthanides;
        # Cm is exception with half-filled 5f7 giving slightly higher count
        95: 3,  96: 4,  97: 3,  98: 3,  99: 3, 100: 3,                   # Am-Fm
    }
    n_valence = float(VALENCE_ELECTRONS.get(atomic_number, 0))

    features = (
        row_onehot + group_onehot + block_onehot +
        [
            _safe_float(elem.atomic_radius),
            _safe_float(elem.average_ionic_radius),
            _safe_float(elem.X),
            _safe_float(elem.ionization_energy),
            _safe_float(elem.electron_affinity),
            n_valence,
            float(len(ox_states)),
            _safe_float(max(ox_states)) if ox_states else 0.0,
            _safe_float(elem.atomic_mass),
            _safe_float(elem.density_of_solid),
            _safe_float(elem.molar_volume),
        ]
    )
    return np.array(features, dtype=np.float32).tolist()


atom_init = {}
for z in range(1, 101):
    atom_init[str(z)] = get_perovskite_features(z)


with open('atom_init_new.json', 'w') as f:
    json.dump(atom_init, f, indent=2)
