import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import time

from data.dataset import AFDataset
from data.ecg_augment import ECGAugment

# Sets the seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load the YAML configuration file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Sets the model based on the configuration file
def get_model(config):
    """Selects Model Architecture"""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    in_channels = config['model']['in_channels']

    # Non-GMCNet models are for comparison and need to be imported
    if model_name == 'resnet18_1d':
        model = resnet18_1d(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'resnet34_1d':
        model = resnet34_1d(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'sccnn':
        from model.sccnn import SCCNN
        model = SCCNN(num_classes=num_classes)
    elif model_name == 'imcresnet':
        from model.imcresnet import IMCResNet
        model = IMCResNet(num_classes=num_classes)
    elif model_name == 'moetrans':
        from model.moetrans import MoEAF
        model = MoEAF()
    elif model_name == 'seqafnet':
        from model.seqafnet import SeqAFNet
        model = SeqAFNet()
    elif model_name == 'mfegnet':
        from model.mfegnet import MFEGNet
        model = MFEGNet()
    elif model_name == 'mscgn':
        from model.mscgn import MSCGN
        model = MSCGN()

    # MGCNet
    elif model_name == 'bgm':
        from model.bgm import GMCNet
        model = GMCNet()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

# Optimizer - Hyperparameter study but Adam is the default
def get_optimizer(model, config):
    """Training Optimizer"""
    optim_type = config['optimizer']['type']
    params = config['optimizer']['params']

    if optim_type == 'Adam':
        return optim.Adam(model.parameters(), **params)
    elif optim_type == 'SGD':
        return optim.SGD(model.parameters(), **params)
    elif optim_type == 'AdamW':
        return optim.AdamW(model.parameters(), **params)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

# Learning Rate Scheduler - Hyperparameter choice
def get_scheduler(optimizer, config):
    """Learning Rate Scheduler"""
    sched_type = config['scheduler']['type']
    params = config['scheduler']['params']

    if sched_type == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif sched_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif sched_type == 'CosineAnnealingLR':
        T_max = params.get('T_max', 25)
        eta_min = params.get('eta_min', 0.000001)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    elif sched_type == 'InverseTimeLR':
        # lr = initial_lr / (1 + decay_rate * epoch)
        decay_rate = params.get('decay_rate', 0.0001)
        last_epoch = params.get('last_epoch', -1)

        def lr_lambda(epoch):
            return 1.0 / (1 + decay_rate * epoch)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")

# DataLoader same as in test.py
def get_dataloader(config, is_train=True):
    """DataLoader"""
    if is_train:
        data_cfg = config['dataset']['train']
        aug_cfg = config['augmentation']['train']
        split = data_cfg['split']
    else:
        data_cfg = config['dataset']['val']
        aug_cfg = config['augmentation']['val']
        split = data_cfg['split']

    # Data Augmentation Class
    transform = ECGAugment(**aug_cfg)

    # Dataset Class
    dataset = AFDataset(
        path1=data_cfg['split_dir'],
        path2=data_cfg['data_dir'],
        split=split,
        fold=data_cfg['fold'],
        transform=transform
    )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=is_train, # True: Order varies
        drop_last=is_train # True: Not full set used each time
    )
    return loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, config, max_batches = None):
    """Model Training"""

    start = time.perf_counter()

    best_accuracy = 0.0
    num_epochs = config['training']['num_epochs']
    save_dir = config['logging']['checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)

    patience = config['training'].get('patience', 10) # Early stop patience
    early_stop_counter = 0
    early_stopping = False # Starts as False, but can be changed later

    for epoch in range(num_epochs):

        if early_stopping:
            break

        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        batch_count_train = 0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for batch_idx, (ecgs,imgs, labels) in enumerate(t):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch_count_train += 1

                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs,total_scl,scl_losses = model(ecgs,imgs,labels,True,True,True)
                loss = criterion(outputs, labels) + 0.01*total_scl # Combined Loss Function
                loss.backward() # Backwards pass through the network
                optimizer.step() # Gradient step defined by the optimizer

                running_loss += loss.item() # Add loss to the sum for this epoch
                _, predicted = torch.max(outputs, 1) 
                correct_preds += (predicted == labels).sum().item() # Number of correct predictions
                total_preds += labels.size(0) # Total number of predictions made

                accuracy = 100 * correct_preds / total_preds
                t.set_postfix(loss=running_loss / (t.n + 1), accuracy=accuracy)

        # Epoch Summary
        epoch_loss = running_loss / batch_count_train
        epoch_accuracy = 100 * correct_preds / total_preds

        # Validation Performance
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        batch_count_val = 0

        with torch.no_grad(): # torch.no_grad() turns off gradient tracking, saving memory and time
            for batch_idx, (ecgs,imgs, labels) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch_count_val += 1
                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)
                outputs,total_scl,scl_losses = model(ecgs,imgs,labels,True,True,True)
                loss = criterion(outputs, labels) + 0.01*total_scl
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / batch_count_val

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Summary
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # TensorBoard Results
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch + 1)
        writer.add_scalar('Loss/valid', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/valid', val_accuracy, epoch + 1)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)

        # best_accuracy set to 0.0 to start
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy # Updates the best accuracy so far
            save_path = os.path.join(save_dir, f"{config['title']}.pth")
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"Best model saved: {save_path}, Accuracy: {best_accuracy:.2f}%")
            early_stop_counter = 0  # 
        else: # If the val_accuracy drops below best_accuracy, the early_stop_counter is incremented
            early_stop_counter += 1
            tqdm.write(f"Early stopping counter: {early_stop_counter}/{patience}")

            # if early_stop_counter >= patience and epoch+1 >= 20:
            if early_stop_counter >= patience: # If early_stop_counter >= 10 (patience)
                tqdm.write(f"Early stopping triggered after {epoch + 1} epochs.")
                early_stopping = True # Breaks the loop on the next cycle

    stop = time.perf_counter()
    elapsed = stop - start

    print(f"Training completed in {elapsed}. Best validation accuracy: {best_accuracy:.2f}%")

# Class Weights (Share of each int he dataset)
def get_class_weights(loader, device):
    class_counts = {}
    total_samples = 0

    # Counts the number of samples
    for _, labels in loader:
        for label in labels.numpy():
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1

    num_classes = len(class_counts) # Should be 2 for (N, AFIB) case
    weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        weight = total_samples / (num_classes * count) # 
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float).to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification Training")
    parser.add_argument(
        '--config',
        type=str,
        default='config/resnet18_1d_afdb.yaml',
        help='Path to the config file (YAML)'
    )
    parser.add_argument(
        '--max-batches', 
        type = int,
        default=None
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load the configuration YAML
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Max Number of Batches (for a smoke run)
    max_batches = args.max_batches

    # Set the seeds
    seed = config['training']['seed']
    set_seed(seed)
    print(f'Random seed set to {seed}')

    # Choose CUDA if possible, otherwise CPU
    device_config = config['training']['device']
    if device_config == 'auto':
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_config)
    print(f'Using device: {device}')

    # Load the data for training and validation sets
    train_loader = get_dataloader(config, is_train=True)
    val_loader = get_dataloader(config, is_train=False)

    print(f'Train set size: {len(train_loader.dataset)}')
    print(f'Validation set size: {len(val_loader.dataset)}')

    # Model
    model = get_model(config)
    model = model.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # class_weights = get_class_weights(train_loader, device)
    # print(f'Class weights: {class_weights}')
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and LR Scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # TensorBoard
    log_dir = os.path.join(config['logging']['tensorboard_dir'], config['title'])
    writer = SummaryWriter(log_dir)

    # Train the Model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, config, max_batches)

    writer.close() # Closes the TensorBoard file