# GitHub Repository for "AtomSight-GNN: a global feature-attention layer that extracts reproducible, property-specific chemical drivers from crystal graph neural networks"
 
## Introduction
 
This repository accompanies the paper:
 
**Yinning Z., et al., "AtomSight-GNN: a global feature-attention layer that extracts reproducible, property-specific chemical drivers from crystal graph neural networks" (2026)**.
 
It contains the source code and data pipeline for **AtomSight-GNN**, which integrates a global feature-dimension attention layer into crystal graph neural networks. AtomSight-GNN learns task-specific atomic property importance rankings during end-to-end training, producing directly readable feature importance scores at no additional inference cost.
 
AtomSight-GNN is validated on two backbone architectures, **OxiGraphX** (PNA-based aggregation with learnable MLP weights) and **CGCNN** (weighted sum aggregation with Gaussian bond-distance edge features). Combined with two backbone architectures, AtomSight-GNN predicts formation energies (Ef) across 33,973 inorganic crystals accurately while yielding feature rankings that are highly reproducible across independent training runs (cross-seed Spearman ρ = 0.84 ± 0.06 and 0.87 ± 0.04, respectively). 
 
## Repository Overview
 
```
AtomSight-GNN/
├── data/                       ← shared dataset
│   ├── id_prop.csv             ← structure IDs and formation energy targets
│   ├── atom_init_new.json      ← 40-dimensional atomic feature encoding
│   ├── query_mp.py             ← fetch properties from Materials Project API
│   ├── download_cifs.py        ← download CIF files from Materials Project
│   ├── generate_atom_init.py   ← generate atom_init_new.json
│   └── cif_files/              ← CIF files (downloaded separately)
│
├── oxigraphx/                  ← OxiGraphX backbone
│   ├── main.py             ← training entry point
│   ├── train.py                ← training loop
│   ├── pred.py                 ← inference on new structures
│   ├── config.py               ← dataset paths and hyperparameters
│   ├── dataset.py              ← PyG InMemoryDataset
│   ├── utils_data.py           ← 40-dim feature extraction, graph building
│   ├── utils.py                ← plotting and saving utilities
│   ├── hyper_search.py         ← hyperparameter search
│   ├── model/                  ← saved model checkpoints
│   └── permutation/            ← permutation importance scripts
│
└── cgcnn/                      ← CGCNN backbone
    ├── main.py                 ← training entry point
    ├── predict.py              ← inference on new structures
    ├── cgcnn/                  ← CGCNN model and data loading code
    └── permutation/            ← permutation importance scripts
```
 
## Data Preparation
 
** Step 1 (Optional) — Fetch the most current list of  from Materials Project**
 
If you want to download the most current list of stable structures with formation energy and bandgap properties, Edit `data/query_mp.py` with your Materials Project API key, then run:
 
```bash
cd data
python query_mp.py
```
This generates `id_prop.csv` with structure IDs and formation energy values.

However, if you want to reproduce our experiment and just use the list we used in the experiment, you can skip Step 1, just use the file 'id_prop.csv' under the folder data. 

**Step 2 — Download CIF files**
 
```bash
python download_cifs.py
```
 
This downloads all 33,973 CIF files into `data/cif_files/`. 
 
**Step 3 — Generate atom features (optional)**
 
```bash
python generate_atom_init.py
```
This will generate the atom feature encoding file `atom_init_new.json`. However, this file is already included in the repository. You only run this code if you want modify the feature encoding:
 
 
## Environmental Requirements

```bash
conda env create -f env.yaml
conda activate atomsight
```

Key packages: `torch`, `torch-geometric`, `pymatgen`, `scikit-learn`, `pandas`, `matplotlib`
 
## Training the Model
 
**OxiGraphX backbone:**
 
Edit `oxigraphx/config.py` to select structure list `id_prop.csv`(formation energy, 33,973 CIF structures) or `bandgap.csv` (band gap, 17,089), and select the target property matching to the column name on the .csv file (`formation_energy_ev` or `band_gap`), then:
 
```bash
cd oxigraphx
python main.py
```
 
**CGCNN backbone:**
 
```bash
cd cgcnn
python main.py data
```
 
Results are saved with timestamped filenames including:
- `_feature_importance.csv` — CEAL attention weight rankings for all 40 features
- `_feature_importance.png` — top-20 feature importance bar chart
- `_training_progress.txt` — epoch-by-epoch train/val/test losses
- `_progress_plotting.png` - training progress plot with epoch-by-epoch train/val losses
- `_y.txt` — test set predictions vs. targets
- `_y.txt.png` — regression plot of test set predictions vs. targets
- `.pth` — model checkpoint
- `_MAE.txt` a summary of all MAEs from the run.

## Predictions
 
**OxiGraphX:**
 
Edit `CHECKPOINT` and `DATA_PATH` in `oxigraphx/pred.py`, Edit `oxigraphx/config.py`, then:
 
```bash
python pred.py
```

The cif files for the three experimental crystals are stored in `oxigraphx/data/pred_data`.

**CGCNN:**
 
```bash
python predict.py <checkpoint_path> <data_path>
```
 
Example:
```bash
python predict.py 08_14/model_best_20260814_180852.pth.tar pred_data/
```
 
 
## Citation
 
If you use this code for your research, please cite our paper:
 
```
@article{zhang2026atomsight,
  author={Zhang, Yinning, et al.},
  title={AtomSight-GNN: a global feature-attention layer that extracts reproducible, property-specific chemical drivers from crystal graph neural networks},
  year={2026}
}
```
 
## Contact
 
For any inquiries or assistance with the code, please contact:
 
* Yinning Zhang: [yzhang56@westga.edu](mailto:yzhang56@westga.edu)
We welcome your feedback and collaboration. Thank you!
