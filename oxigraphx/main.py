import time
import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
# from torch.utils.data import ConcatDataset
import torch.optim as optim

from utils import *
from dataset import *
from model.atomsight_ceal import *
from train import *
from config import *


# read in hyperparameters


learning_rate = 0.001
numLayers = 3
batch_size = 64

'''
learning_rate =float(os.getenv('learning_rate')) 
numLayers = int(os.getenv('numLayers'))
batch_size = int(os.getenv('batch_size'))
'''

# Parameters needed
# numLayers = 5
seed = 123456
shuffle = True 
num_workers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_dim = 1
towers = 1
pre_layers = 1
post_layers = 1
divide_input = False
disp_step = 1
dispProgress = True
epochs = 120

print(device)


# ============================================================================
# LOAD DATASETS WITH ENHANCED FEATURES
# ============================================================================
result = prep_datasets_from_cif(
    root=config.dataset_root,
    train_p=0.6,
    val_p=0.2,
    normalize_features=True,
    normalize_target=False,
)

# Unpack results
combined_train_dataset, combined_val_dataset, combined_test_dataset, targetscaler = result  #targetscaler contains both x y scalers

print('combined_train_dataset ', len(combined_train_dataset))
print('combined_val_dataset ', len(combined_val_dataset))
print('combined_test_dataset ', len(combined_test_dataset))

#################################################################


in_channels = combined_train_dataset[0].x.size(dim=-1)
print('in_channels ', in_channels)

out_channels = 80
aggregators = ['sum', 'mean', 'min', 'max', 'std']

# Note: set scalers to ['identify'] to factually by-pass using any degree-based scalers
scalers = ['identity'] #, 'amplification', 'attenuation']
#scalers = ['identity', 'amplification', 'attenuation']
deg = generate_deg(combined_train_dataset)
print('degree', deg)

# Set Dataloader
g = torch.Generator()
g.manual_seed(seed)
train_data_loader = DataLoader(combined_train_dataset, 
                               batch_size=batch_size, 
                               shuffle=shuffle, 
                               num_workers=num_workers, 
                               worker_init_fn=worker_init_fn, 
                               generator=g)

val_data_loader =   DataLoader(combined_val_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)

test_data_loader =  DataLoader(combined_test_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)


model = MyCEALNetwork( in_channels,
                        out_channels,
                        aggregators,
                        scalers,
                        deg,
                        edge_dim = edge_dim,
                        towers = towers,
                        numLayers = numLayers,
                        pre_layers = pre_layers,
                        post_layers = post_layers,
                        divide_input = False
                    )

model.to(device)

model_config = {
    'architecture': {
        'in_channels': in_channels, 'out_channels': out_channels,
        'aggregators': aggregators, 'scalers': scalers, 'deg': deg,
        'edge_dim': edge_dim, 'towers': towers, 'numLayers': numLayers,
        'pre_layers': pre_layers, 'post_layers': post_layers,
        'divide_input': divide_input,
    },
}

# Set optimizer
optimizer = torch.optim.Adam([
    {'params': [p for n, p in model.named_parameters() if n != 'attention.w'],
     'lr': learning_rate,
     'weight_decay': 1e-4},
    {'params': [model.attention.w],
     'lr': learning_rate * 20,
     'weight_decay': 0.0},
], )

# Set learning rate scheduler for the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=0.000000001)


feature_names = [f"feat_{i}" for i in range(in_channels)]

trained_model, timestr = train(
   epochs, model, train_data_loader, val_data_loader, test_data_loader,
   optimizer, scheduler, device, disp_step, batch_size, learning_rate, numLayers,
   target_scaler=targetscaler, dispProgress=dispProgress, model_config=model_config,
   feature_names=feature_names,
)
