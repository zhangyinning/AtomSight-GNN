import os
import csv
import json
from tqdm import tqdm

from mp_api.client import MPRester



def build_mp_formation_energy_dataset(root_dir, api_key, max_entries=None):
    """
    Build the MP formation energy dataset used in CGCNN paper.

    Creates:
        root_dir/id_prop.csv
        root_dir/atom_init.json
        root_dir/mp-xxxx.cif
    """

    os.makedirs(root_dir, exist_ok=True)

    id_prop_path = os.path.join(root_dir, "id_prop.csv")

    # download official CGCNN atom features
    # download_official_atom_init_json(atom_init_path)

    print("[INFO] Querying Materials Project for stable structures + formation energies...")

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            is_stable=True,
            fields=["material_id", "formation_energy_per_atom", "structure"]
        )

        docs = list(docs)

        if max_entries is not None:
            docs = docs[:max_entries]

        print(f"[INFO] Retrieved {len(docs)} stable materials")

        id_prop_rows = []

        for doc in tqdm(docs, desc="Saving CIF + formation energy"):
            mid = str(doc.material_id)
            fe = doc.formation_energy_per_atom
            structure = doc.structure

            if fe is None or structure is None:
                continue

            cif_path = os.path.join(root_dir, 'cif_files', "f"{mid}.cif")

            try:
                structure.to(fmt="cif", filename=cif_path)
                id_prop_rows.append([mid, fe])
            except Exception as e:
                print(f"[ERROR] Failed writing {mid}: {e}")

    # write id_prop.csv
    with open(id_prop_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(id_prop_rows)

    print(f"[OK] Saved id_prop.csv with {len(id_prop_rows)} entries -> {id_prop_path}")
    print("[DONE] MP formation energy dataset successfully built.")


if __name__ == "__main__":

    ROOT_DIR = "."
    API_KEY = "your key"

    # For debugging you can set max_entries=2000
    build_mp_formation_energy_dataset(
        root_dir=ROOT_DIR,
        api_key=API_KEY,
        max_entries=None
    )
