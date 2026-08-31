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
SELECTED_INDICES = [9, 10, 11, 12, 13, 14, 16, 17, 24, 30, 31, 33, 35, 39]

# ============================================================================
# MODEL HYPERPARAMETERS — must match main.py exactly
# ============================================================================
OUT_CHANNELS = 80
AGGREGATORS  = ['sum', 'mean', 'min', 'max', 'std']
SCALERS      = ['identity']
EDGE_DIM     = 1
TOWERS       = 1
NUM_LAYERS   = 3
PRE_LAYERS   = 1
POST_LAYERS  = 1

# ============================================================================
# DATALOADER SETTINGS
# ============================================================================
BATCH_SIZE   = 64
NUM_WORKERS  = 2

# ============================================================================
# DATASET SPLIT — must match training split exactly so test set is the same
# ============================================================================
TRAIN_P             = 0.6
VAL_P               = 0.2
NORMALIZE_FEATURES  = True
NORMALIZE_TARGET    = False
