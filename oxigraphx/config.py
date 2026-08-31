"""
Configuration for CIF->Graph + formation energy regression.

Expected dataset layout (example):
  <dataset_root>/
    data.csv                 # contains mp_id and formation_energy_ev (or your chosen target_col)
    cif_files/
      mp-11830.cif
      mp-68610.cif
      ...

You can override these fields in your training script as needed.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../ABO3/finalcode

class Config:
    # Root folder that contains the CSV and CIF directory
    dataset_root = os.path.normpath(os.path.join(_THIS_DIR, "..", "data"))

    # CSV file that contains mp_id and target column
    csv_filename = "id_prop.csv"

    # Folder (inside dataset_root) containing CIF files
    cif_dirname = "cif_files"

    # Column names in the CSV
    mp_id_col = "mp_id"
    target_col = "formation_energy_ev"   # column name from your screenshot

    # Graph construction
    # Cutoff (Å) for building edges by neighbor distance
    neighbor_cutoff = 5.0

    # Whether to include edge distances as edge_attr (recommended)
    use_edge_distances = True

    # What to do when a CIF is missing for an mp_id in CSV:
    # "skip" -> ignore that row; "error" -> raise exception
    missing_cif_policy = "skip"


config = Config
