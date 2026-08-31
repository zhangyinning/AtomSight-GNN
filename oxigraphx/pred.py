import torch
from model.atomsight_ceal import *
from torch_geometric.loader import DataLoader
from dataset import *
from utils import *
import time

def main():
    path = '/mnt/bst/a100/selene/yzhang56/ABO3/finalcode/Results/8/21/batch_size:64learning_rate:0.001num layers:3_1788089258069350.pth'  # add model path

    # Load the full checkpoint (weights + architecture + scalers), instead
    # of a bare state_dict + hardcoded architecture/ymin/ymax.
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    arch = checkpoint['model_config']['architecture']
 
    # Initialize network from the checkpoint's saved architecture
    model = MyCEALNetwork(in_channels = arch['in_channels'],
                out_channels = arch['out_channels'],
                aggregators = arch['aggregators'],
                scalers = arch['scalers'],
                deg = arch['deg'],
                numLayers = arch['numLayers'],
                edge_dim = arch['edge_dim'],
                towers = arch['towers'],
                pre_layers = arch['pre_layers'],
                post_layers = arch['post_layers'],
                divide_input = arch['divide_input'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
 
    # Feature and target scalers saved at training time. x_scaler is
    # applied manually below rather than through MyCIFDataset, because
    # MyCIFDataset(normalize_features=True) FITS A NEW SCALER on whatever
    # data it's given — for prediction on new points we need to reuse the
    # training scaler, not fit a fresh one on the new data.
    x_scaler = checkpoint['target_scaler'].get('x_scaler', None)
    y_scaler = checkpoint['target_scaler'].get('y_scaler', None)
 
    # normalize_features=False, normalize_target=False: load raw features
    # and raw targets, no fitting. x is scaled manually below with the
    # checkpoint's x_scaler; y is left in original scale for comparison
    # (only the model's output needs inverse-transforming, if y_scaler
    # was used at training time).
    pred_dataset = MyCIFDataset(root="/mnt/bst/a100/selene/yzhang56/ABO3/data/pred_data", cfg=config, normalize_features=False, normalize_target=False)
 
    pred_data_loader = DataLoader(pred_dataset,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
 
    my_out = torch.empty(0)
    my_y = torch.empty(0)
    mp_ids = []
    with torch.no_grad():
        for i, data in enumerate(pred_data_loader):
            if x_scaler is not None:
                data.x = torch.tensor(x_scaler.transform(data.x.numpy()), dtype=torch.float32)
            out = model(data)
            my_out = torch.cat((my_out, out.squeeze(-1)), 0)
            my_y = torch.cat((my_y, data.y), 0)
            mp_ids.extend(data.mp_id)  # data.mp_id is a list of ids for this batch
 
    # data.y is already in original scale (normalize_target=False above).
    # Only the model's output needs inverse-transforming, and only if
    # the model was trained on normalized targets.
    if y_scaler is not None:
        my_out = torch.tensor(y_scaler.inverse_transform(my_out.reshape(-1, 1).numpy()).flatten())
 
    for mp_id, pred in zip(mp_ids, my_out.tolist()):
        print(f"{mp_id}: {pred:.4f}")
 
 
if __name__ == "__main__":
    main()