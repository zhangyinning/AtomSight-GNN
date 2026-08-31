import os
import csv
from tqdm import tqdm

from mp_api.client import MPRester


def read_mp_ids_from_id_prop(id_prop_path):
    """
    Read MP IDs (and their target values) from an existing id_prop.csv.
    Expects rows of the form: mp_id,target_value  (no header).
    Returns:
        mp_ids   : list of mp-id strings, in file order
        id_to_y  : dict mapping mp_id -> target value (as float, if present)
    """
    mp_ids = []
    id_to_y = {}
    with open(id_prop_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            mid = row[0].strip()
            mp_ids.append(mid)
            if len(row) > 1:
                try:
                    id_to_y[mid] = float(row[1])
                except ValueError:
                    pass
    return mp_ids, id_to_y


def download_cifs_for_ids(root_dir, api_key, id_prop_path, max_entries=None,
                           chunk_size=1000):
    """
    Download CIFs for the MP IDs listed in an existing id_prop.csv.

    Creates:
        root_dir/mp-xxxx.cif   for every ID successfully retrieved
        root_dir/id_prop_downloaded.csv  -- id_prop rows actually written
                                             (i.e. structure was retrievable)

    Args:
        root_dir     : output directory for CIF files
        api_key      : Materials Project API key
        id_prop_path : path to the existing id_prop.csv containing mp_ids
                       (and optionally target values) to download
        max_entries  : optional cap, mainly for debugging
        chunk_size   : number of material_ids to request per API call
                       (MP API handles batching internally too, but chunking
                       keeps individual requests smaller/more resilient)
    """
    os.makedirs(root_dir, exist_ok=True)

    mp_ids, id_to_y = read_mp_ids_from_id_prop(id_prop_path)
    if max_entries is not None:
        mp_ids = mp_ids[:max_entries]

    print(f"[INFO] Read {len(mp_ids)} MP IDs from {id_prop_path}")
    print("[INFO] Querying Materials Project for structures + formation energies...")

    id_prop_rows = []
    missing_ids = []

    with MPRester(api_key) as mpr:
        for i in tqdm(range(0, len(mp_ids), chunk_size), desc="Querying MP in chunks"):
            batch_ids = mp_ids[i:i + chunk_size]

            docs = mpr.materials.summary.search(
                material_ids=batch_ids,
                fields=["material_id", "formation_energy_per_atom", "structure"],
            )
            docs = list(docs)

            found_ids = set()
            for doc in docs:
                mid = str(doc.material_id)
                found_ids.add(mid)

                fe = doc.formation_energy_per_atom
                structure = doc.structure
                if fe is None or structure is None:
                    continue

                cif_path = os.path.join(root_dir, 'cif_files', "f"{mid}.cif")
                try:
                    structure.to(fmt="cif", filename=cif_path)
                    # prefer the target value from the original id_prop.csv
                    # if present, otherwise fall back to the freshly-fetched fe
                    y_val = id_to_y.get(mid, fe)
                    id_prop_rows.append([mid, y_val])
                except Exception as e:
                    print(f"[ERROR] Failed writing {mid}: {e}")

            missing_ids.extend(mid for mid in batch_ids if mid not in found_ids)

    if missing_ids:
        print(f"[WARN] {len(missing_ids)} MP IDs from {id_prop_path} "
              f"were not found / retrievable in the current MP database.")
        missing_path = os.path.join(root_dir, "missing_mp_ids.txt")
        with open(missing_path, "w") as f:
            f.write("\n".join(missing_ids))
        print(f"[INFO] Wrote list of missing IDs -> {missing_path}")

    out_path = os.path.join(root_dir, "id_prop_downloaded.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(id_prop_rows)

    print(f"[OK] Saved {len(id_prop_rows)} CIF+target rows -> {out_path}")
    print("[DONE] CIF download complete.")


if __name__ == "__main__":
    ROOT_DIR = "./cif_files"
    ID_PROP_PATH = "id_prop.csv"
    API_KEY = "your key"

    # For debugging you can set max_entries=2000
    download_cifs_for_ids(
        root_dir=ROOT_DIR,
        api_key=API_KEY,
        id_prop_path=ID_PROP_PATH,
        max_entries=None,
    )