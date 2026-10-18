import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from data.dataset import AFDataset
from data.ecg_augment import ECGAugment
import json

# Load the YAML configuration file 
def load_config(config_path="config/resnet18_1d_afdb.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Extract the model from the YAML configuration file
def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    in_channels = config['model']['in_channels']

    # All of these non-BGM models were used for comparative evaluations in the paper. 
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

def get_dataloader(config, is_train=False):
    """DataLoader"""
    if is_train: # For training set
        data_cfg = config['dataset']['train'] 
        aug_cfg = config['augmentation']['train']
        split = data_cfg['split']
    else: # For test set
        data_cfg = config['dataset']['test']
        aug_cfg = config['augmentation']['test']
        split = data_cfg['split']

    transform = ECGAugment(**aug_cfg)

    # Dataset with split, folds, and augmentation transforms
    dataset = AFDataset(
        path1=data_cfg['split_dir'],
        path2=data_cfg['data_dir'],
        split=split,
        fold=data_cfg['fold'],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False, # Fixed order
        drop_last=False # All samples used
    )
    return loader, dataset

# Loads the model weights (must be trained already)
def load_model_weights(model, checkpoint_path, device):
    """Model Weights"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() 
    print(f"Model weights loaded from {checkpoint_path}")
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="ECG Classification Training")
    parser.add_argument(
        '--config',
        type=str,
        default='config/resnet18_1d_afdb.yaml',
        help='Path to the config file (YAML)'
    )
    return parser.parse_args()

def main():
    # Load the config file
    args = parse_args()
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # CUDA if training.device is 'auto', CPU otherwise
    device_config = config['training']['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_config == 'auto' else torch.device(device_config)
    print(f'Using device: {device}')

    # Load the test dataset, wraps the data in ECGAugment, and generates AFDataset
    test_loader, test_dataset = get_dataloader(config, is_train=False)
    print(f"Test set size: {len(test_dataset)}")

    # Define model
    model = get_model(config) # Pick the architecture from the model.name 
    model.to(device)

    # Checkpoints
    checkpoint_dir = config['logging']['checkpoint_dir']
    title = config['title']
    checkpoint_path = os.path.join(checkpoint_dir, f"{title}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load trained model weights
    model = load_model_weights(model, checkpoint_path, device)

    # Labels, data, and logits all empty
    ## Recall that GMCNet returns logits, t_global (256), and f_global (256)
    all_labels = []
    all_probs = []
    all_preds = []

    all_t = []
    all_f = []

    criterion = nn.CrossEntropyLoss() # Primary component of the loss (+ weighted amount of CSCL)
    total_loss = 0.0 # Initialize loss

    with torch.no_grad(): # torch.no_grad() turns off gradient tracking, saving memory and time
        with tqdm(test_loader, desc="Testing", unit="batch") as t:
            for ecgs,imgs, labels in t:
                ecgs,imgs, labels = ecgs.to(device), imgs.to(device), labels.to(device)

                outputs,t,f = model(ecgs,imgs) # logits, t_global, f_global
                loss = criterion(outputs, labels) # Only using cross entropy loss here
                total_loss += loss.item() # Adds the loss to total_loss

                # Softmax to obtain probabilities from the logits, then argmax to get predictions
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                # Adds the labels, probabilities, predictions, t_global, and f_global to the lists
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                all_preds.extend(preds)

                all_t.extend(t.cpu().numpy())
                all_f.extend(f.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # np.array of the lists
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    t_arr = np.array(all_t)
    f_arr = np.array(all_f)

    # Likely two classes (N and AFIB)
    num_classes = config['model']['num_classes']

    # Metrics
    ## Accuracy, F1, precision, recall
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    precision = precision_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')
    recall = recall_score(y_true, y_pred, average='macro' if num_classes > 2 else 'binary')  # 敏感度

    # Area under the ROC curve (AUC)
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    # Specificity: TN / (TN + FP) = Proportion of real negatives that were predicted correctly
    specificity_list = []
    for cls in range(num_classes):
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list)

    # Summary of the metrics and results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity (avg): {avg_specificity:.4f}")
    print("-"*50)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("="*50)

    # Results are written to {result_dir}/{title}/
    result_dir = os.path.join(config['logging']['result_dir'], config['title'])
    os.makedirs(result_dir, exist_ok=True)

    # Further Results
    results_df = pd.DataFrame({
        'label': y_true,
        'prediction': y_pred,
    })

    for i in range(y_prob.shape[1]):
        results_df[f'prob_class_{i}'] = y_prob[:, i]

    results_df.to_csv(os.path.join(result_dir, 'test_predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(result_dir, 'test_predictions.csv')}")

    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'auc': float(auc),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),  # 同 recall
        'specificity_avg': float(avg_specificity),
        'specificity_per_class': [float(s) for s in specificity_list]
    }

    with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Results\n")
        f.write(f"Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Specificity (avg): {avg_specificity:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, digits=4))

    print(f"Metrics and report saved to {result_dir}")

    np.savez_compressed(
        os.path.join(result_dir, 'predictions.npz'),
        labels=y_true,
        t=t_arr,
        f=f_arr
    )

if __name__ == '__main__':
    main()