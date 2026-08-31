import torch
from torch.utils.data import DataLoader
from cgcnn.data import CIFData, collate_pool

# ── Import everything from the training script ───────────────────────────────
from main_withscaler import (
    AtomSightGNN,
    AtomFeatureNormalizer,
    Normalizer,
    validate
)

CHECKPOINT = '08_15/model_best_20260815_125020.pth.tar'
DATA_PATH  = 'pred_data'

 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load dataset
dataset = CIFData(DATA_PATH)

# 2. Apply atom feature normalization from checkpoint
checkpoint = torch.load(CHECKPOINT, map_location='cpu')

structures, _, _ = dataset[0]
orig_atom_fea_len = structures[0].shape[-1]
nbr_fea_len       = structures[1].shape[-1]

atom_fea_normalizer = AtomFeatureNormalizer(torch.zeros(orig_atom_fea_len))
atom_fea_normalizer.load_state_dict(checkpoint['atom_fea_normalizer'])

for key in dataset.ari._embedding:
    original   = torch.tensor(dataset.ari.get_atom_fea(key), dtype=torch.float32).unsqueeze(0)
    normalized = atom_fea_normalizer.norm(original).squeeze(0).numpy()
    dataset.ari._embedding[key] = normalized

# 3. DataLoader
test_loader = DataLoader(dataset, batch_size=len(dataset),
                         shuffle=False, collate_fn=collate_pool)

# 4. Build model and load weights
saved_args = checkpoint['args']
model = AtomSightGNN(orig_atom_fea_len, nbr_fea_len,
                     atom_fea_len=saved_args['atom_fea_len'],
                     n_conv=saved_args['n_conv'],
                     h_fea_len=saved_args['h_fea_len'],
                     n_h=saved_args['n_h'])
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# 5. Load target normalizer
normalizer = Normalizer(torch.zeros(1))
normalizer.load_state_dict(checkpoint['normalizer'])

# 6. Run prediction using the training script's validate function
criterion = torch.nn.MSELoss()
validate(test_loader, model, criterion, normalizer,
         test=True, test_results_file='predictions.csv')