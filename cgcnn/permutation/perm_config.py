"""
perm_config.py
==============
All constants for the permutation importance experiment.
Edit this file if your training setup differs.
"""

# ============================================================================
# FEATURE LABELS — 40-feature perovskite vector
# ============================================================================
FEATURE_LABELS = (
    [f'Period {i+1}' for i in range(7)] +           # indices 0-6
    [f'Group {i+1}'  for i in range(18)] +          # indices 7-24
    ['Block s', 'Block p', 'Block d', 'Block f'] +  # indices 25-28
    [
        'Atomic radius',       # 29
        'Avg ionic radius',    # 30
        'Electronegativity',   # 31
        'Ionization energy',   # 32
        'Electron affinity',   # 33
        'Valence electrons',   # 34
        'N oxidation states',  # 35
        'Max oxidation state', # 36
        'Atomic mass',         # 37
        'Density of solid',    # 38
        'Molar volume',        # 39
    ]
)

N_FEATURES = 40

# CEAL top-13 indices selected by AtomSight attention rankings
SELECTED_INDICES = [6, 10, 12, 13, 14, 15, 16, 17, 24, 31, 33]

# ============================================================================
# MODEL HYPERPARAMETERS — must match main_new.py exactly
# ============================================================================
ATOM_FEA_LEN = 64
NBR_FEA_LEN  = 41   # GaussianDistance(dmin=0, dmax=8, step=0.2) -> 41 bins
N_CONV       = 3
H_FEA_LEN    = 128
N_H          = 1

# ============================================================================
# DATALOADER SETTINGS
# ============================================================================
BATCH_SIZE   = 256
NUM_WORKERS  = 0
 
# ============================================================================
# DATASET SPLIT — must match training split exactly so test set is the same
# ============================================================================
TRAIN_RATIO  = 0.6
VAL_RATIO    = 0.2
TEST_RATIO   = 0.2