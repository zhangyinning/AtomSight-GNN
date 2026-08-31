import argparse
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from random import sample
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from cgcnn.data import *
from cgcnn.model_atomsight import AtomSightGNN, FeatureImportanceRanker

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.001)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=0.6, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.2, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='Adam', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error

    # Create timestamp and dated output folder
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    date_folder = now.strftime("%m_%d")
    os.makedirs(date_folder, exist_ok=True)
    print(f"Output folder: {date_folder}/")

    # Initialize training history
    training_history = []
    training_start_time = time.time()

    # ── Reproducibility seed (fixes everything EXCEPT weight init) ──
    FIXED_SEED = 42
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)

    # load data
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)
   

    # Normalize atom features to [0,1]. It is directly applied to the 
    # entire dataset becacuse the normalizer is calculated on all atoms
    all_atom_fea = torch.tensor(
        np.array([dataset.ari.get_atom_fea(key)
                  for key in dataset.ari._embedding]),
        dtype=torch.float32
    )
    atom_fea_normalizer = AtomFeatureNormalizer(all_atom_fea)

    for key in dataset.ari._embedding:
        original = torch.tensor(dataset.ari.get_atom_fea(key),
                                dtype=torch.float32).unsqueeze(0)
        normalized = atom_fea_normalizer.norm(original).squeeze(0).numpy()
        dataset.ari._embedding[key] = normalized


    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    weight_init_seed = torch.seed()
    print(f"Weight init seed: {weight_init_seed}")

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = AtomSightGNN(orig_atom_fea_len, nbr_fea_len,
                         atom_fea_len=args.atom_fea_len,
                         n_conv=args.n_conv,
                         h_fea_len=args.h_fea_len,
                         n_h=args.n_h,
                         classification=True if args.task ==
                                                'classification' else False)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD([
            {'params': [p for n, p in model.named_parameters() if n != 'attention.w'],
             'lr': args.lr},
            {'params': [model.attention.w],
             'lr': args.lr * 20},
        ], momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.optim == 'Adam':
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if n != 'attention.w'],
             'lr': args.lr, 'weight_decay': 1e-4},
            {'params': [model.attention.w],
             'lr': args.lr * 20, 'weight_decay': 0.0},
        ])

    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # Checkpoint filenames (timestamped, inside date folder)
    checkpoint_file = os.path.join(date_folder, f'checkpoint_{timestamp}.pth.tar')
    best_model_file = os.path.join(date_folder, f'model_best_{timestamp}.pth.tar')

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        if args.task == 'regression':
            val_loss, mae_error = validate(val_loader, model, criterion, normalizer)
        else:
            mae_error = validate(val_loader, model, criterion, normalizer)
            val_loss = None

        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss if args.task == 'regression' else mae_error
        })

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step(val_loss)

        # remember the best mae_error and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'atom_fea_normalizer': atom_fea_normalizer.state_dict(),
            'args': vars(args)
        }, is_best, best_model_file)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(best_model_file)
    model.load_state_dict(best_checkpoint['state_dict'])
    if args.task == 'regression':
        test_loss, test_mae = validate(test_loader, model, criterion, normalizer, test=True,
                                       test_results_file=os.path.join(date_folder, f'test_results_{timestamp}.csv'))
    else:
        test_mae = validate(test_loader, model, criterion, normalizer, test=True,
                            test_results_file=os.path.join(date_folder, f'test_results_{timestamp}.csv'))
        test_loss = None

    # ----------------------------------------------------------------
    # Feature Importance (AtomSight-GNN)
    # ----------------------------------------------------------------
    if args.task == 'regression':
        FEATURE_NAMES = [f"feat_{i}" for i in range(orig_atom_fea_len)]
        ranker = FeatureImportanceRanker(model=model, feature_names=FEATURE_NAMES)
        report = ranker.rank()
        print('\nTop-10 most important atomic features:')
        print(report.head(10).to_string(index=False))
        importance_filename = os.path.join(date_folder, f'feature_importance_{timestamp}.csv')
        ranker.save_report(importance_filename)
        ranker.plot(report, top_k=20,
                    title='AtomSight-GNN: Learned Atomic Feature Importance',
                    save_path=os.path.join(date_folder, f'feature_importance_{timestamp}.png'))

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    # Save training summary
    summary_filename = os.path.join(date_folder, f'training_summary_{timestamp}.txt')
    with open(summary_filename, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        # f.write(f"Output Folder: {date_folder}/\n")
        f.write(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)\n")
        f.write(f"Number of Epochs: {args.epochs}\n")
        f.write(f"Best Validation MAE: {best_mae_error:.4f}\n")
        if args.task == 'regression':
            f.write(f"Test Loss (MSE): {test_loss:.4f}\n")
            f.write(f"Test MAE: {test_mae:.4f}\n")
        else:
            f.write(f"Test AUC: {test_mae:.4f}\n")
        f.write(f"\nModel Parameters:\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Optimizer: {args.optim}\n")
        f.write(f"  Atom Feature Length: {args.atom_fea_len}\n")
        f.write(f"  Hidden Feature Length: {args.h_fea_len}\n")
        f.write(f"  Number of Conv Layers: {args.n_conv}\n")
        f.write(f"  Number of Hidden Layers: {args.n_h}\n")
        f.write(f"\nSaved Files:\n")
        f.write(f"  Checkpoint:      checkpoint_{timestamp}.pth.tar\n")
        f.write(f"  Best Model:      model_best_{timestamp}.pth.tar\n")
        f.write(f"  Test Results:    test_results_{timestamp}.csv\n")
        f.write(f"  Feature Import:  feature_importance_{timestamp}.csv\n")
        f.write(f"  Training Curve:  training_curve_{timestamp}.png\n")

    print(f"\nTraining summary saved to: {summary_filename}")

    # Save training progress (epoch by epoch)
    progress_filename = os.path.join(date_folder, f'training_progress_{timestamp}.txt')
    with open(progress_filename, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
        for record in training_history:
            f.write(f"{record['epoch']},{record['train_loss']:.6f},{record['val_loss']:.6f}\n")

    print(f"Training progress saved to: {progress_filename}")

    # Plot training progress
    print("\nGenerating training curve plot...")
    try:
        epochs = [record['epoch'] for record in training_history]
        train_losses = [record['train_loss'] for record in training_history]
        val_losses = [record['val_loss'] for record in training_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = os.path.join(date_folder, f'training_curve_{timestamp}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training curve plot saved to: {plot_filename}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate plot. Error: {e}")


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores))

    return losses.avg


def validate(val_loader, model, criterion, normalizer, test=False, test_results_file=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open(test_results_file, 'w') as f:  # fixed: was literal '{timestamp}'
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'

    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return losses.avg, mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg

class AtomFeatureNormalizer(object):
    """Normalize atom feature vectors column-wise to [0,1].
    MinMaxScaler is more appropriate than standardization for atom
    features since they have known physical bounds and the full
    element range should be represented."""

    def __init__(self, tensor):
        # tensor shape: [n_atoms, n_features]
        self.min = tensor.min(dim=0).values   # [n_features]
        self.max = tensor.max(dim=0).values   # [n_features]
        denom = self.max - self.min
        denom[denom == 0] = 1.0  # avoid div by zero for constant dims

    def norm(self, tensor):
        denom = self.max - self.min
        denom[denom == 0] = 1.0
        return (tensor - self.min) / denom

    def denorm(self, normed_tensor):
        return normed_tensor * (self.max - self.min) + self.min

    def state_dict(self):
        return {'min': self.min, 'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, best_model_file):
    if is_best:
        torch.save(state, best_model_file)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()