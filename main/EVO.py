"""
EVO.py: Evolutionary Algorithm for Hyperparameter Optimization

This script uses Evolutionary Algorithm to automatically search
for optimal hyperparameters of Baseline classifiers and HRL models.

Key Features:
1. 5-Fold Cross-Validation based fitness evaluation
2. Genetic algorithm operations: Selection, Crossover, Mutation
3. Support for HRL mode and Baseline mode
4. Multi-classifier architecture support (CRNN, CBGRU, Transformer)

Usage:
    # HRL mode
    Set USE_HRL = True and run: python main/EVO.py
    
    # Baseline mode
    Set USE_HRL = False and run: python main/EVO.py
"""

import numpy as np
import torch
import torch.nn as nn
from models import TemporalEmbedding, ROIEmbedding, MacroQNet, MicroQNet, CRNNClassifier, SpatioTemporalTransformer
import torch.optim as optim
import random
import torch.nn.functional as F
from utils.seed import seed_everything
from config import get_config

# Configuration
args = get_config()
seed_everything(args.seed, skip_random=True)  # Non-deterministic exploration for RL

# Standard imports
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import utils.util as util

# HRL-specific imports
from main.HRL import DFCSequenceDataset, ReplayBuffer, evaluate_dataset, compute_macro_dqn_loss, compute_micro_bandit_loss, HRL




##########################################
# 1. Dataset and Basic Utility Functions
##########################################

class BOLDDataset(Dataset):
    """
    Baseline mode BOLD signal dataset
    
    Extracts dFC features using fixed window size and step size.
    
    Args:
        bold_arr (np.ndarray): BOLD time series data (N, T, nROIs)
        label_arr (np.ndarray): Labels (N,) - 0: NC, 1: MDD
        patient_ids (np.ndarray): Patient IDs (N,)
        device_if_gpu (torch.device, optional): Device for GPU usage
    """
    def __init__(self, bold_arr, label_arr, patient_ids, device_if_gpu=None):
        super().__init__()
        self.bold_arr = bold_arr
        self.label_arr = label_arr
        self.patient_ids = patient_ids

        self.use_gpu_data = (device_if_gpu is not None)
        if self.use_gpu_data:
            self.bold_gpu = torch.tensor(bold_arr, dtype=torch.float, device=device_if_gpu)
            self.label_gpu = torch.tensor(label_arr, dtype=torch.long, device=device_if_gpu)
        else:
            self.bold_gpu = None
            self.label_gpu = None

    def __len__(self):
        return len(self.bold_arr)

    def __getitem__(self, idx):
        if self.use_gpu_data:
            # Return (T, nROIs), label, patient_id
            return self.bold_gpu[idx], self.label_gpu[idx].item(), self.patient_ids[idx]
        else:
            bd_i = torch.tensor(self.bold_arr[idx], dtype=torch.float)
            lb_i = self.label_arr[idx]
            return bd_i, lb_i, self.patient_ids[idx]

def extract_window(bold_data, start_idx, w):
    """
    Extract window from BOLD signal (with zero-padding)
    
    Args:
        bold_data (torch.Tensor): BOLD time series (T, nROIs)
        start_idx (int): Starting index
        w (int): Window size
        
    Returns:
        torch.Tensor: Extracted window (w, nROIs), zero-padded if necessary
    """
    T, nROIs = bold_data.shape
    end_idx = start_idx + w
    if start_idx >= T:
        return torch.zeros((w, nROIs), dtype=torch.float, device=bold_data.device)
    if end_idx > T:
        seg_valid = bold_data[start_idx:T, :]
        pad_len = w - (T - start_idx)
        seg_pad = torch.zeros((pad_len, nROIs), dtype=torch.float, device=bold_data.device)
        return torch.cat([seg_valid, seg_pad], dim=0)
    return bold_data[start_idx:end_idx, :]

def compute_dfc(bold_segment):
    """
    Compute Dynamic Functional Connectivity (dFC)
    
    Calculates correlation matrix between ROIs using Pearson correlation.
    
    Args:
        bold_segment (torch.Tensor): BOLD window (1, w, nROIs)
        
    Returns:
        torch.Tensor: Flattened correlation matrix (1, nROIs*nROIs)
    """
    x = bold_segment[0]  # (w, nROIs)
    w_len, nROIs = x.shape

    # Remove mean of each ROI
    x_t = x.transpose(0, 1) - x.transpose(0, 1).mean(dim=1, keepdim=True)

    # Covariance
    cov = (x_t @ x_t.transpose(0, 1)) / max(w_len - 1, 1)

    # Normalize to get correlation
    var = cov.diagonal().unsqueeze(0)
    std = torch.sqrt(var + 1e-12)
    corr = cov / (std.transpose(0, 1) * std + 1e-12)

    return corr.reshape(1, -1)  # (1, nROIs*nROIs)

def create_dfc_sequence(bold_data, window_size, step_size):
    """
    Generate dFC feature sequence using sliding window approach
    
    Args:
        bold_data (torch.Tensor): BOLD time series (T, nROIs)
        window_size (int): Window size
        step_size (int): Step interval
        
    Returns:
        torch.Tensor: dFC feature sequence (S, nROIs*nROIs)
        where S = number of windows
    """
    segments = []
    current_idx = 0
    while current_idx < bold_data.shape[0]:
        seg = extract_window(bold_data, current_idx, window_size).unsqueeze(0)  # (1, w, nROIs)
        dfc_feat = compute_dfc(seg)  # (1, nROIs*nROIs)
        segments.append(dfc_feat)
        current_idx += step_size
    return torch.cat(segments, dim=0)  # (S, nROIs*nROIs)


##################################
# 2. Model Creation Helper
##################################

def create_classifier(classifier_type, input_dim, hidden_dim, num_layers, dropout, 
                      num_classes=2, out_channels=32, kernel_size=3, pool_size=2,
                      num_heads=8, dim_feedforward=512, transformer_dropout=0.1):
    """
    Factory function for creating classifier models
    
    Args:
        classifier_type (str): 'crnn', 'cbgru', or 'transformer'
        input_dim (int): Input dimension (nROIs * nROIs)
        hidden_dim (int): Hidden layer dimension
        num_layers (int): Number of layers
        dropout (float): Dropout rate
        num_classes (int): Number of classification classes (default: 2)
        out_channels (int): CNN output channels (for CRNN/CBGRU)
        kernel_size (int): CNN kernel size
        pool_size (int): Pooling size
        num_heads (int): Number of attention heads (for Transformer)
        dim_feedforward (int): FFN dimension (for Transformer)
        transformer_dropout (float): Transformer dropout
        
    Returns:
        nn.Module: Created classifier model
    """
    if classifier_type == 'transformer':
        model = SpatioTemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            num_classes=num_classes
        )
    elif classifier_type == 'cbgru':
        model = CRNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            out_channels=out_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            bidirectional=True
        )
    else:  # 'crnn' (default)
        model = CRNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            out_channels=out_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            bidirectional=False
        )
    return model


###################################
# 3. Training / Evaluation Helpers
###################################

def calculate_classification_metrics(logits, labels):
    """
    Calculate classification performance metrics
    
    Args:
        logits (torch.Tensor): Model output logits (N, 2)
        labels (torch.Tensor): Ground truth labels (N,)
        
    Returns:
        dict: Performance metrics
            - acc: Accuracy
            - sen: Sensitivity (Recall)
            - spec: Specificity
            - f1: F1-Score
            - auc: AUC-ROC
    """
    preds = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    acc = accuracy_score(labels_np, preds)
    TP = ((preds == 1) & (labels_np == 1)).sum()
    TN = ((preds == 0) & (labels_np == 0)).sum()
    FP = ((preds == 1) & (labels_np == 0)).sum()
    FN = ((preds == 0) & (labels_np == 1)).sum()

    sen = TP / (TP + FN + 1e-8)
    spec= TN / (TN + FP + 1e-8)
    f1  = (2*TP) / (2*TP + FP + FN + 1e-8)

    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    try:
        auc_val = roc_auc_score(labels_np, probs)
    except ValueError:
        auc_val = 0.5

    return {
        'acc': acc,
        'sen': sen,
        'spec': spec,
        'f1': f1,
        'auc': auc_val
    }

def evaluate_dataset_with_minibatch(model, data_loader, device, window_size, step_size):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for bold_ts_batch, labels_batch, _ in data_loader:
            bold_ts_batch = bold_ts_batch.to(device)
            labels_batch  = labels_batch.to(device)

            # Convert each sample in the batch to a sequence of dFC features
            batch_features_list = []
            for i in range(bold_ts_batch.size(0)):
                seq_features = create_dfc_sequence(
                    bold_ts_batch[i], window_size, step_size
                )
                batch_features_list.append(seq_features.unsqueeze(0))

            batch_features = torch.cat(batch_features_list, dim=0)  # (B, S, input_dim)
            logits = model(batch_features)
            all_logits.append(logits)
            all_labels.append(labels_batch)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return calculate_classification_metrics(all_logits, all_labels)

def train_evaluate_hrl_with_minibatch(hparams, train_loader, val_loader, device, epochs=5):
    """
    HRL model training and evaluation
    
    Trains HRL model with given hyperparameters and evaluates validation performance.
    Trains Macro Q-Network (DQN) and Micro Q-Network (Contextual Bandit) together.
    
    Args:
        hparams (dict): Hyperparameter dictionary
            - crnn_hidden_dim, crnn_num_layers, crnn_dropout
            - crnn_lr, crnn_wd
            - macro_lr, macro_wd
            - micro_lr, micro_wd
            - rl_hidden_dim, rl_embed_dim, rl_gamma
            - rl_batch_size, rl_buffer_capacity, rl_target_update_freq
            - focal_alpha, focal_gamma
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Training device
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (val_metrics, hrl_model)
            - val_metrics (dict): Validation performance metrics
            - hrl_model (nn.Module): Trained HRL model
    """
    # Extract hyperparameters
    crnn_hidden_dim = hparams['crnn_hidden_dim']
    crnn_num_layers = hparams['crnn_num_layers']
    crnn_dropout = hparams['crnn_dropout']
    crnn_lr = hparams['crnn_lr']
    crnn_wd = hparams['crnn_wd']
    
    macro_lr = hparams['macro_lr']
    macro_wd = hparams['macro_wd']
    micro_lr = hparams['micro_lr']
    micro_wd = hparams['micro_wd']
    
    rl_hidden_dim = hparams['rl_hidden_dim']
    rl_embed_dim = hparams['rl_embed_dim']
    rl_gamma = hparams['rl_gamma']
    rl_batch_size = hparams['rl_batch_size']
    rl_buffer_capacity = hparams['rl_buffer_capacity']
    rl_target_update_freq = hparams['rl_target_update_freq']
    
    focal_alpha = hparams.get('focal_alpha', 0.75)
    focal_gamma = hparams.get('focal_gamma', 3.0)
    
    # Get dataset info
    nROIs = train_loader.dataset.bold_arr.shape[-1]
    input_dim = nROIs * nROIs
    
    # Get possible actions from dataset
    possible_window_sizes = train_loader.dataset.possible_window_sizes
    possible_shift_ratios = train_loader.dataset.possible_shift_ratios
    num_macro_actions = len(possible_window_sizes)
    num_micro_actions = len(possible_shift_ratios)
    
    # Create HRL model
    hrl_model = HRL(
        input_dim=input_dim,
        crnn_hidden_dim=crnn_hidden_dim,
        crnn_num_layers=crnn_num_layers,
        crnn_dropout=crnn_dropout,
        num_macro_actions=num_macro_actions,
        num_micro_actions=num_micro_actions,
        rl_hidden_dim=rl_hidden_dim,
        rl_embed_dim=rl_embed_dim,
        macro_in_dim=train_loader.dataset.macro_in_dim,
        micro_in_dim=train_loader.dataset.micro_in_dim,
        rl_dropout=crnn_dropout,
        num_classes=2
    ).to(device)
    
    # Optimizers
    classifier_optimizer = optim.Adam(hrl_model.classifier.parameters(), lr=crnn_lr, weight_decay=crnn_wd)
    macro_optimizer = optim.Adam(hrl_model.macroQ.parameters(), lr=macro_lr, weight_decay=macro_wd)
    micro_optimizer = optim.Adam(hrl_model.microQ.parameters(), lr=micro_lr, weight_decay=micro_wd)
    
    # Target networks
    macroQ_target1 = MacroQNet(train_loader.dataset.macro_in_dim, rl_hidden_dim, rl_embed_dim, num_macro_actions, dropout=crnn_dropout).to(device)
    macroQ_target2 = MacroQNet(train_loader.dataset.macro_in_dim, rl_hidden_dim, rl_embed_dim, num_macro_actions, dropout=crnn_dropout).to(device)
    macroQ_target1.load_state_dict(hrl_model.macroQ.state_dict())
    macroQ_target2.load_state_dict(hrl_model.macroQ.state_dict())
    macroQ_target1.eval()
    macroQ_target2.eval()
    
    # Replay buffers
    macro_buffer = ReplayBuffer(rl_buffer_capacity)
    micro_buffer = ReplayBuffer(rl_buffer_capacity)
    
    # Focal loss
    criterion = nn.CrossEntropyLoss()  # Simplified for EVO
    
    # Training loop
    for ep in range(epochs):
        hrl_model.train()
        
        for batch_data, batch_labels, _ in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # Classifier training
            classifier_optimizer.zero_grad()
            outputs = hrl_model.classifier(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hrl_model.classifier.parameters(), 6.0)
            classifier_optimizer.step()
            
            # RL Training: Macro Q-Network
            if len(macro_buffer) >= rl_batch_size:
                macro_batch = macro_buffer.sample(rl_batch_size)
                macro_loss = compute_macro_dqn_loss(
                    macro_batch, hrl_model.macroQ, macroQ_target1, macroQ_target2, rl_gamma, device
                )
                macro_optimizer.zero_grad()
                macro_loss.backward()
                torch.nn.utils.clip_grad_norm_(hrl_model.macroQ.parameters(), 6.0)
                macro_optimizer.step()
            
            # RL Training: Micro Q-Network
            if len(micro_buffer) >= rl_batch_size:
                micro_batch = micro_buffer.sample(rl_batch_size)
                micro_loss = compute_micro_bandit_loss(micro_batch, hrl_model.microQ, device)
                micro_optimizer.zero_grad()
                micro_loss.backward()
                torch.nn.utils.clip_grad_norm_(hrl_model.microQ.parameters(), 6.0)
                micro_optimizer.step()
        
        # Collect RL transitions every few epochs
        if (ep + 1) % 5 == 0:
            train_eval_results = hrl_model.evaluate_and_get_rewards(
                train_loader, criterion, device, 'full',
                fixed_window_size=30,
                fixed_shift_ratio=0.5,
                macro_in_dim=train_loader.dataset.macro_in_dim,
                micro_in_dim=train_loader.dataset.micro_in_dim,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                max_steps_per_trial=20,
                fold_idx=1,
                dataset_type='train',
                track_usage=False,
                verbose=False,
                save_dir=None
            )
            for transition in train_eval_results.get('macro_transitions', []):
                macro_buffer.push(transition)
            for transition in train_eval_results.get('micro_transitions', []):
                micro_buffer.push(transition)
        
        # Update target networks
        if (ep + 1) % rl_target_update_freq == 0:
            macroQ_target1.load_state_dict(hrl_model.macroQ.state_dict())
            macroQ_target2.load_state_dict(hrl_model.macroQ.state_dict())
    
    # Evaluate on validation set
    val_results = evaluate_dataset(
        hrl_model, val_loader, criterion, device, 'full',
        fixed_window_size=30,
        fixed_shift_ratio=0.5,
        macro_in_dim=train_loader.dataset.macro_in_dim,
        micro_in_dim=train_loader.dataset.micro_in_dim,
        possible_window_sizes=possible_window_sizes,
        possible_shift_ratios=possible_shift_ratios,
        max_steps_per_trial=20,
        fold_idx=1,
        dataset_type='val',
        track_usage=False,
        verbose=False,
        save_dir=None
    )
    
    val_metrics = {
        'acc': val_results['acc'],
        'sen': val_results['sen'],
        'spec': val_results['spec'],
        'f1': val_results['f1'],
        'auc': val_results['auc']
    }
    
    return val_metrics, hrl_model


def train_evaluate_classifier_with_minibatch(hparams, train_loader, val_loader, device, epochs=5):
    """
    Baseline classifier training and evaluation (fixed window/step)
    
    Trains and evaluates CRNN, CBGRU, or Transformer classifier.
    
    Args:
        hparams (dict): Hyperparameter dictionary
            - classifier_type: 'crnn', 'cbgru', or 'transformer'
            - hidden_dim, num_layers, dropout
            - lr, wd
            - window_size, step_size (fixed values)
            - num_heads, dim_feedforward (for Transformer)
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Training device
        epochs (int): Number of training epochs
        
    Returns:
        tuple: (val_metrics, model)
            - val_metrics (dict): Validation performance metrics
            - model (nn.Module): Trained classifier model
    """
    classifier_type = hparams.get('classifier_type', 'crnn')
    hidden_dim  = hparams['hidden_dim']
    num_layers  = hparams['num_layers']
    dropout     = hparams['dropout']
    lr          = hparams['lr']
    wd          = hparams['wd']
    window_size = hparams['window_size']
    step_size   = hparams['step_size']
    
    # Transformer-specific parameters
    num_heads = hparams.get('num_heads', 8)
    dim_feedforward = hparams.get('dim_feedforward', 512)
    transformer_dropout = hparams.get('transformer_dropout', 0.1)

    nROIs = train_loader.dataset.bold_arr.shape[-1]
    input_dim = nROIs * nROIs

    # -- Instantiate classifier based on type
    model = create_classifier(
        classifier_type=classifier_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
        out_channels=32,
        kernel_size=3,
        pool_size=2,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        transformer_dropout=transformer_dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # -- Training loop
    for ep in range(epochs):
        model.train()
        for bold_ts_batch, labels_batch, _ in train_loader:
            bold_ts_batch = bold_ts_batch.to(device)
            labels_batch  = labels_batch.to(device)

            # Build feature sequences for each item in the mini-batch
            batch_features_list = []
            for i in range(bold_ts_batch.size(0)):
                seq_features = generate_dFC_features(bold_ts_batch[i], window_size, step_size)
                batch_features_list.append(seq_features.unsqueeze(0))  # (1, S, input_dim)

            batch_features = torch.cat(batch_features_list, dim=0)  # (B, S, input_dim)

            logits = model(batch_features)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    val_metrics = evaluate_dataset_with_minibatch(model, val_loader, device, window_size, step_size)
    return val_metrics, model


###########################################
# 4. Evolutionary Optimization (Cross-Val)
###########################################

def create_random_hparams(param_ranges, use_hrl=False):
    """
    Generate random hyperparameters
    
    Creates random hyperparameter combinations from given ranges.
    
    Args:
        param_ranges (dict): List of possible values for each hyperparameter
        use_hrl (bool): True for HRL parameters, False for Baseline parameters
        
    Returns:
        dict: Random hyperparameter combination
    """
    if use_hrl:
        # HRL mode: different parameter set
        hparams = {
            # Classifier hyperparameters
            'crnn_hidden_dim': random.choice(param_ranges['crnn_hidden_dim']),
            'crnn_num_layers': random.choice(param_ranges['crnn_num_layers']),
            'crnn_dropout': random.choice(param_ranges['crnn_dropout']),
            'crnn_lr': random.choice(param_ranges['crnn_lr']),
            'crnn_wd': random.choice(param_ranges['crnn_wd']),
            'crnn_out_channels': random.choice(param_ranges['crnn_out_channels']),
            'crnn_kernel_size': random.choice(param_ranges['crnn_kernel_size']),
            'crnn_pool_size': random.choice(param_ranges['crnn_pool_size']),
            
            # RL hyperparameters
            'macro_lr': random.choice(param_ranges['macro_lr']),
            'macro_wd': random.choice(param_ranges['macro_wd']),
            'micro_lr': random.choice(param_ranges['micro_lr']),
            'micro_wd': random.choice(param_ranges['micro_wd']),
            'rl_hidden_dim': random.choice(param_ranges['rl_hidden_dim']),
            'rl_embed_dim': random.choice(param_ranges['rl_embed_dim']),
            'macro_in_dim': random.choice(param_ranges['macro_in_dim']),
            'micro_in_dim': random.choice(param_ranges['micro_in_dim']),
            'rl_gamma': random.choice(param_ranges['rl_gamma']),
            'epsilon': random.choice(param_ranges['epsilon']),
            'rl_batch_size': random.choice(param_ranges['rl_batch_size']),
            'rl_buffer_capacity': random.choice(param_ranges['rl_buffer_capacity']),
            'rl_target_update_freq': random.choice(param_ranges['rl_target_update_freq']),
            'tau': random.choice(param_ranges['tau']),
            
            # Gradient clipping
            'classifier_clip_norm': random.choice(param_ranges['classifier_clip_norm']),
            'macroq_clip_norm': random.choice(param_ranges['macroq_clip_norm']),
            'microq_clip_norm': random.choice(param_ranges['microq_clip_norm']),
            
            # Feature preprocessing
            'feature_dim': random.choice(param_ranges['feature_dim']),
            'pca_dim': random.choice(param_ranges['pca_dim']),
            'lda_dim': random.choice(param_ranges['lda_dim']),
            'threshold': random.choice(param_ranges['threshold']),
            
            # Focal Loss
            'focal_alpha': random.choice(param_ranges.get('focal_alpha', [0.75])),
            'focal_gamma': random.choice(param_ranges.get('focal_gamma', [3.0])),
            
            # Reward shaping
            'diversity_bonus_weight': random.choice(param_ranges['diversity_bonus_weight']),
            'max_diversity_history_len': random.choice(param_ranges['max_diversity_history_len']),
            'max_baseline_history_len': random.choice(param_ranges['max_baseline_history_len']),
            'reward_nc_penalty_effective': random.choice(param_ranges['reward_nc_penalty_effective']),
        }
    else:
        # Baseline mode: original parameters
        hparams = {
            # Classifier selection
            'classifier_type': random.choice(param_ranges.get('classifier_type', ['crnn'])),
            'hidden_dim':   random.choice(param_ranges['hidden_dim']),
            'num_layers':   random.choice(param_ranges['num_layers']),
            'dropout':      random.choice(param_ranges['dropout']),
            'lr':           random.choice(param_ranges['lr']),
            'wd':           random.choice(param_ranges['wd']),
            'window_size':  random.choice(param_ranges['window_size']),
            'step_size':    random.choice(param_ranges['step_size']),
            
            # CRNN-specific
            'crnn_out_channels': random.choice(param_ranges['crnn_out_channels']),
            'crnn_kernel_size': random.choice(param_ranges['crnn_kernel_size']),
            'crnn_pool_size': random.choice(param_ranges['crnn_pool_size']),
            
            # Feature preprocessing
            'feature_dim': random.choice(param_ranges['feature_dim']),
            'pca_dim': random.choice(param_ranges['pca_dim']),
            'lda_dim': random.choice(param_ranges['lda_dim']),
            'threshold': random.choice(param_ranges['threshold']),
            
            # Focal Loss
            'focal_alpha': random.choice(param_ranges.get('focal_alpha', [0.75])),
            'focal_gamma': random.choice(param_ranges.get('focal_gamma', [3.0])),
            
            # Gradient clipping
            'classifier_clip_norm': random.choice(param_ranges['classifier_clip_norm']),
        }
        
        # Add transformer-specific parameters if transformer is selected
        if hparams['classifier_type'] == 'transformer':
            hparams['transformer_num_heads'] = random.choice(param_ranges.get('transformer_num_heads', [8]))
            hparams['transformer_num_layers'] = random.choice(param_ranges.get('transformer_num_layers', [4]))
            hparams['dim_feedforward'] = random.choice(param_ranges.get('dim_feedforward', [512]))
            hparams['transformer_dropout'] = random.choice(param_ranges.get('transformer_dropout', [0.1]))
    
    return hparams

def mutate_hparams(hparams, param_ranges, mutation_prob=0.2, use_hrl=False):
    """
    Hyperparameter mutation operation
    
    Mutates each hyperparameter with certain probability.
    
    Args:
        hparams (dict): Original hyperparameters
        param_ranges (dict): Possible value ranges
        mutation_prob (float): Mutation probability (default: 0.2)
        use_hrl (bool): Whether HRL mode
        
    Returns:
        dict: Mutated hyperparameters
    """
    keys = list(hparams.keys())
    
    if not use_hrl:
        old_classifier_type = hparams.get('classifier_type', 'crnn')
    
    for k in keys:
        if k in param_ranges and random.random() < mutation_prob:
            hparams[k] = random.choice(param_ranges[k])
    
    # Baseline mode: handle transformer params
    if not use_hrl:
        new_classifier_type = hparams.get('classifier_type', 'crnn')
        if new_classifier_type == 'transformer' and old_classifier_type != 'transformer':
            # Add transformer parameters
            hparams['num_heads'] = random.choice(param_ranges.get('num_heads', [8]))
            hparams['dim_feedforward'] = random.choice(param_ranges.get('dim_feedforward', [512]))
            hparams['transformer_dropout'] = random.choice(param_ranges.get('transformer_dropout', [0.1]))
        elif new_classifier_type != 'transformer' and old_classifier_type == 'transformer':
            # Remove transformer parameters
            hparams.pop('num_heads', None)
            hparams.pop('dim_feedforward', None)
            hparams.pop('transformer_dropout', None)
    
    return hparams

def crossover_hparams(parent1, parent2, use_hrl=False):
    """
    Hyperparameter crossover operation
    
    Creates child by crossing two parent hyperparameters.
    
    Args:
        parent1 (dict): First parent hyperparameters
        parent2 (dict): Second parent hyperparameters
        use_hrl (bool): Whether HRL mode
        
    Returns:
        dict: Crossed child hyperparameters
    """
    # Get common keys
    common_keys = set(parent1.keys()) & set(parent2.keys())
    all_keys = list(common_keys)
    
    if len(all_keys) == 0:
        return parent1.copy()
    
    idx = random.randint(1, max(1, len(all_keys)-1))
    child = {}
    
    for i, k in enumerate(all_keys):
        if i < idx:
            child[k] = parent1[k]
        else:
            child[k] = parent2[k]
    
    # Baseline mode: ensure classifier_type specific params are correct
    if not use_hrl:
        classifier_type = child.get('classifier_type', 'crnn')
        if classifier_type == 'transformer':
            if 'num_heads' not in child:
                child['num_heads'] = parent1.get('num_heads', parent2.get('num_heads', 8))
            if 'dim_feedforward' not in child:
                child['dim_feedforward'] = parent1.get('dim_feedforward', parent2.get('dim_feedforward', 512))
            if 'transformer_dropout' not in child:
                child['transformer_dropout'] = parent1.get('transformer_dropout', parent2.get('transformer_dropout', 0.1))
        else:
            # Remove transformer params if not transformer
            child.pop('num_heads', None)
            child.pop('dim_feedforward', None)
            child.pop('transformer_dropout', None)
    
    return child

def crossval_fitness(hparams, fold_loaders, device, epochs=5, use_hrl=False):
    """
    5-Fold Cross-Validation based fitness evaluation
    
    Trains and validates on 5 folds with given hyperparameters
    and returns average performance.
    
    Args:
        hparams (dict): Hyperparameters to evaluate
        fold_loaders (list): [(train_loader, val_loader), ...] 5 folds
        device (torch.device): Training device
        epochs (int): Training epochs per fold
        use_hrl (bool): Whether HRL mode
        
    Returns:
        dict: 5-fold average performance metrics
            - acc, sen, spec, f1, auc
    """
    metrics_sum = {'acc':0.0, 'sen':0.0, 'spec':0.0, 'f1':0.0, 'auc':0.0}
    num_folds = len(fold_loaders)

    for (train_loader, val_loader) in fold_loaders:
        if use_hrl:
            val_metrics, _ = train_evaluate_hrl_with_minibatch(hparams, train_loader, val_loader, device, epochs)
        else:
            val_metrics, _ = train_evaluate_classifier_with_minibatch(hparams, train_loader, val_loader, device, epochs)
        for k in metrics_sum.keys():
            metrics_sum[k] += val_metrics[k]

    # average
    for k in metrics_sum.keys():
        metrics_sum[k] /= num_folds

    return metrics_sum

def evolutionary_search_with_cv(fold_loaders, device, param_ranges,
                               pop_size=8, generations=5, epochs=5, use_hrl=False):
    """
    Evolutionary algorithm based hyperparameter optimization
    
    Searches for optimal hyperparameters using genetic algorithm.
    Evaluates fitness of each individual (hyperparameter combination) with 5-fold cross-validation.
    
    Algorithm:
        1. Generate initial population (random)
        2. Evaluate fitness (5-fold CV)
        3. Selection: top 50% survive
        4. Generate children with crossover and mutation
        5. Repeat
    
    Args:
        fold_loaders (list): 5-fold data loaders
        device (torch.device): Training device
        param_ranges (dict): Hyperparameter search space
        pop_size (int): Population size (default: 8)
        generations (int): Number of generations (default: 5)
        epochs (int): Training epochs per individual
        use_hrl (bool): Whether HRL mode
        
    Returns:
        tuple: (best_hparams, best_val_metrics)
            - best_hparams: optimal hyperparameters
            - best_val_metrics: validation performance of optimal hyperparameters
    """
    population = [create_random_hparams(param_ranges, use_hrl=use_hrl) for _ in range(pop_size)]

    best_acc = 0.0
    best_hparams = None
    best_val_metrics = None

    for gen in range(generations):
        fitness_list = []
        for hparams in population:
            avg_val_metrics = crossval_fitness(hparams, fold_loaders, device, epochs, use_hrl=use_hrl)
            fitness_list.append((avg_val_metrics['acc'], hparams, avg_val_metrics))

        # Sort by average accuracy
        fitness_list.sort(key=lambda x: x[0], reverse=True)

        # Update global best
        if fitness_list[0][0] > best_acc:
            best_acc = fitness_list[0][0]
            best_hparams = fitness_list[0][1]
            best_val_metrics = fitness_list[0][2]

        model_type = 'HRL' if use_hrl else fitness_list[0][1].get('classifier_type', 'crnn')
        print(f"[Gen {gen+1}] Best val acc={fitness_list[0][0]:.4f} | Type: {model_type} | "
              f"metrics={fitness_list[0][2]}")

        # Survivor selection
        survivors = [h for (_, h, _) in fitness_list[: len(population)//2]]

        # Offspring
        new_offspring = []
        while len(new_offspring) < (pop_size - len(survivors)):
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            child = crossover_hparams(p1, p2, use_hrl=use_hrl)
            child = mutate_hparams(child, param_ranges, mutation_prob=0.2, use_hrl=use_hrl)
            new_offspring.append(child)

        population = survivors + new_offspring

    print("\n=== Finished Evolutionary Search ===")
    print(f"Best val acc (avg over folds): {best_acc:.4f}")
    print(f"Best hyperparameters: {best_hparams}")
    print(f"Best val metrics (avg over folds): {best_val_metrics}")
    return best_hparams, best_val_metrics


##########################################
# 5. Full 5-fold Example with Final Test
##########################################

def build_folds(all_data, fold_indices, batch_size, device_if_gpu=None, use_hrl=False, possible_window_sizes=None, possible_shift_ratios=None):
    """
    Create 5-Fold data loaders
    
    Creates (train_loader, val_loader) pair for each fold.
    Supports both HRL mode and Baseline mode.
    
    Args:
        all_data (dict): All data (result from util.load_signal_data())
        fold_indices (list): Fold index list [1, 2, 3, 4, 5]
        batch_size (int): Batch size
        device_if_gpu (torch.device): GPU device (for Baseline)
        use_hrl (bool): True for DFCSequenceDataset, False for BOLDDataset
        possible_window_sizes (list): Window size options for HRL
        possible_shift_ratios (list): Shift ratio options for HRL
        
    Returns:
        list: [(train_loader, val_loader), ...] 5 folds
    """
    folds = []
    for fold_idx in fold_indices:
        train_bd = all_data[f'fold_{fold_idx}_train_data.npy']
        train_lb = all_data[f'fold_{fold_idx}_train_labels.npy']
        train_pid= all_data[f'fold_{fold_idx}_train_patient_ids.npy']

        val_bd   = all_data[f'fold_{fold_idx}_val_data.npy']
        val_lb   = all_data[f'fold_{fold_idx}_val_labels.npy']
        val_pid  = all_data[f'fold_{fold_idx}_val_patient_ids.npy']

        if use_hrl:
            # HRL mode: use DFCSequenceDataset
            train_ds = DFCSequenceDataset(
                train_bd, train_lb, train_pid,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                epsilon=0.15,
                macro_in_dim=128,
                micro_in_dim=128,
                ablation_mode='full'
            )
            val_ds = DFCSequenceDataset(
                val_bd, val_lb, val_pid,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                epsilon=0.15,
                macro_in_dim=128,
                micro_in_dim=128,
                ablation_mode='full'
            )
        else:
            # Baseline mode: use BOLDDataset
            train_ds = BOLDDataset(train_bd, train_lb, train_pid, device_if_gpu=device_if_gpu)
            val_ds   = BOLDDataset(val_bd,   val_lb,   val_pid,   device_if_gpu=device_if_gpu)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        folds.append((train_loader, val_loader))
    return folds

if __name__ == "__main__":
    """
    Main execution block
    
    Uses evolutionary algorithm to search for optimal hyperparameters
    and evaluates performance on final test set.
    
    Usage:
        1. Set USE_HRL variable to True/False
        2. Run python main/EVO.py
        3. Check results
    """
    
    # ====================================
    # Configuration: Switch between HRL and Baseline mode
    # ====================================
    USE_HRL = True  # Set to True for HRL optimization, False for baseline classifiers
    
    # HRL hyperparameter ranges (based on README.md defaults)
    hrl_param_ranges = {
        # Classifier (CRNN) hyperparameters
        'crnn_hidden_dim': [64, 96, 128, 192, 256],  # Default: 128
        'crnn_num_layers': [1, 2, 3, 4],            # Default: 2
        'crnn_dropout': [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],  # Default: 0.2
        'crnn_lr': [5e-5, 1e-5, 5e-6, 1e-6, 5e-7],  # Around default: 1e-5
        'crnn_wd': [5e-5, 1e-5, 5e-6, 1e-6, 0.0],   # Around default: 1e-5
        'crnn_out_channels': [16, 24, 32, 48],       # Default: 32
        'crnn_kernel_size': [3, 5, 7],               # Default: 3
        'crnn_pool_size': [2, 3, 4],                 # Default: 2
        
        # Macro-Agent (DQN) hyperparameters
        'macro_lr': [5e-5, 1e-5, 5e-6, 1e-6, 5e-7], # Around default: 1e-5
        'macro_wd': [5e-5, 1e-5, 5e-6, 1e-6, 0.0],  # Around default: 1e-5
        
        # Micro-Agent (Contextual Bandit) hyperparameters
        'micro_lr': [5e-5, 1e-5, 5e-6, 1e-6, 5e-7], # Around default: 1e-5
        'micro_wd': [5e-5, 1e-5, 5e-6, 1e-6, 0.0],  # Around default: 1e-5
        
        # RL network architecture
        'rl_hidden_dim': [64, 96, 128, 192, 256],    # Default: 128
        'rl_embed_dim': [64, 96, 128, 192, 256],    # Default: 128
        'macro_in_dim': [64, 96, 128, 192],          # Default: 128
        'micro_in_dim': [64, 96, 128, 192],          # Default: 128
        
        # RL training hyperparameters
        'rl_gamma': [0.90, 0.93, 0.95, 0.97, 0.99], # Default: 0.95
        'epsilon': [0.05, 0.1, 0.15, 0.2, 0.25],    # Default: 0.15
        'rl_batch_size': [1, 2, 4],                  # Default: 1
        'rl_buffer_capacity': [50000, 100000, 200000, 500000], # Default: 100000
        'rl_target_update_freq': [1, 2, 3, 5],       # Default: 1
        'tau': [0.0005, 0.001, 0.002, 0.005],       # Default: 0.001
        
        # Gradient clipping
        'classifier_clip_norm': [1.0, 2.0, 4.0, 6.0, 8.0],  # Default: 6.0
        'macroq_clip_norm': [1.0, 2.0, 4.0, 6.0, 8.0],      # Default: 6.0
        'microq_clip_norm': [1.0, 2.0, 4.0, 6.0, 8.0],      # Default: 6.0
        
        # Feature preprocessing
        'feature_dim': [32, 48, 64, 96, 128],         # Default: 64
        'pca_dim': [128, 192, 256, 384],             # Default: 256
        'lda_dim': [128, 192, 256, 384],             # Default: 256
        'threshold': [0.01, 0.03, 0.05, 0.07, 0.1], # Default: 0.05
        
        # Focal Loss hyperparameters
        'focal_alpha': [0.5, 0.6, 0.75, 0.85, 0.9], # Default: 0.75
        'focal_gamma': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], # Default: 3.0
        
        # Reward shaping parameters
        'diversity_bonus_weight': [0.01, 0.025, 0.05, 0.075, 0.1],  # Default: 0.05
        'max_diversity_history_len': [10, 15, 20, 30, 40],          # Default: 20
        'max_baseline_history_len': [50, 75, 100, 150, 200],        # Default: 100
        'reward_nc_penalty_effective': [-0.5, -1.0, -1.5, -2.0],   # Default: -1.0
    }
    
    # Baseline hyperparameter ranges (based on README.md defaults)
    baseline_param_ranges = {
        'classifier_type': ['crnn', 'cbgru', 'transformer'],
        'hidden_dim':   [64, 96, 128, 192, 256],    # Around CRNN default: 128
        'num_layers':   [1, 2, 3, 4],              # Around CRNN default: 2
        'dropout':      [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],  # Around CRNN default: 0.2
        'lr':           [5e-5, 1e-5, 5e-6, 1e-6, 5e-7],  # Around default: 1e-5
        'wd':           [5e-5, 1e-5, 5e-6, 1e-6, 0.0],   # Around default: 1e-5
        'window_size':  [20, 30, 40, 50, 60, 75, 90, 100], # Common window sizes
        'step_size':    [5, 10, 15, 20, 25, 30, 40],      # Common step sizes
        
        # CRNN-specific parameters
        'crnn_out_channels': [16, 24, 32, 48],       # Default: 32
        'crnn_kernel_size': [3, 5, 7],               # Default: 3
        'crnn_pool_size': [2, 3, 4],                 # Default: 2
        
        # Feature preprocessing
        'feature_dim': [32, 48, 64, 96, 128],         # Default: 64
        'pca_dim': [128, 192, 256, 384],             # Default: 256
        'lda_dim': [128, 192, 256, 384],             # Default: 256
        'threshold': [0.01, 0.03, 0.05, 0.07, 0.1], # Default: 0.05
        
        # Focal Loss hyperparameters
        'focal_alpha': [0.5, 0.6, 0.75, 0.85, 0.9], # Default: 0.75
        'focal_gamma': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], # Default: 3.0
        
        # Gradient clipping
        'classifier_clip_norm': [1.0, 2.0, 4.0, 6.0, 8.0],  # Default: 6.0
        
        # Transformer-specific (used only when classifier_type='transformer')
        'transformer_num_heads': [4, 8, 12, 16],                   # Default: 8
        'transformer_num_layers': [2, 4, 6, 8],                    # Default: 4
        'dim_feedforward': [256, 384, 512, 768, 1024],             # Default: 512
        'transformer_dropout': [0.0, 0.1, 0.15, 0.2, 0.3],        # Default: 0.1
    }
    
    # Select parameter ranges based on mode
    param_ranges = hrl_param_ranges if USE_HRL else baseline_param_ranges

    # GPU or CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU_DATA = True if not USE_HRL else False  # HRL mode doesn't use GPU data

    # Load your data with 5 folds
    all_data = util.load_signal_data()
    fold_indices = [1, 2, 3, 4, 5]  # 5 folds

    # HRL-specific: Define action spaces
    possible_window_sizes = list(range(10, 233, 10))  # [10, 20, 30, ..., 230] TRs
    possible_shift_ratios = [i / 100 for i in range(1, 101)]  # 0.01~1.00

    # 1) Build a list of (train_loader, val_loader) for cross-validation
    batch_size = 1 if USE_HRL else 2  # HRL uses batch_size=1
    fold_loaders = build_folds(
        all_data, fold_indices, batch_size,
        device_if_gpu=DEVICE if USE_GPU_DATA else None,
        use_hrl=USE_HRL,
        possible_window_sizes=possible_window_sizes if USE_HRL else None,
        possible_shift_ratios=possible_shift_ratios if USE_HRL else None
    )

    # 2) Evolutionary search with cross-validation
    mode_name = "HRL" if USE_HRL else "Baseline (CRNN/CBGRU/Transformer)"
    print(f"\n{'='*60}")
    print(f"Starting Evolutionary Search for {mode_name}")
    print(f"{'='*60}\n")
    
    best_hparams, best_val_metrics = evolutionary_search_with_cv(
        fold_loaders,
        DEVICE,
        param_ranges,
        pop_size=4,       # Population size (reduce for faster testing)
        generations=3,    # Number of generations
        epochs=10 if USE_HRL else 20,  # Training epochs per individual
        use_hrl=USE_HRL
    )

    # 3) Final evaluation on each fold's test set using best hyperparams
    #    We train on (train+val) combined for each fold, then test on test set
    print(f"\n{'='*60}")
    print("Final Test Evaluation with Best Hyperparameters")
    print(f"{'='*60}\n")
    
    test_metrics_list = []
    for fold_idx in fold_indices:
        print(f"\n=== Final evaluation for fold {fold_idx} ===")

        # Combine train+val
        train_bd = all_data[f'fold_{fold_idx}_train_data.npy']
        train_lb = all_data[f'fold_{fold_idx}_train_labels.npy']
        train_pid= all_data[f'fold_{fold_idx}_train_patient_ids.npy']

        val_bd   = all_data[f'fold_{fold_idx}_val_data.npy']
        val_lb   = all_data[f'fold_{fold_idx}_val_labels.npy']
        val_pid  = all_data[f'fold_{fold_idx}_val_patient_ids.npy']

        combined_bd = np.concatenate([train_bd, val_bd], axis=0)
        combined_lb = np.concatenate([train_lb, val_lb], axis=0)
        combined_pid= np.concatenate([train_pid, val_pid], axis=0)
        
        if USE_HRL:
            combined_ds = DFCSequenceDataset(
                combined_bd, combined_lb, combined_pid,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                epsilon=0.15,
                macro_in_dim=128,
                micro_in_dim=128,
                ablation_mode='full'
            )
        else:
            combined_ds = BOLDDataset(combined_bd, combined_lb, combined_pid,
                                      device_if_gpu=DEVICE if USE_GPU_DATA else None)
        combined_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)

        # Test set
        test_bd  = all_data[f'fold_{fold_idx}_test_data.npy']
        test_lb  = all_data[f'fold_{fold_idx}_test_labels.npy']
        test_pid = all_data[f'fold_{fold_idx}_test_patient_ids.npy']
        
        if USE_HRL:
            test_ds = DFCSequenceDataset(
                test_bd, test_lb, test_pid,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                epsilon=0.15,
                macro_in_dim=128,
                micro_in_dim=128,
                ablation_mode='full'
            )
        else:
            test_ds = BOLDDataset(test_bd, test_lb, test_pid,
                                   device_if_gpu=DEVICE if USE_GPU_DATA else None)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Train final model on combined data
        if USE_HRL:
            final_metrics, final_model = train_evaluate_hrl_with_minibatch(
                best_hparams,
                combined_loader,
                combined_loader,  # Here we use train+val merged data only without val_loader
                DEVICE,
                epochs=10  # Example: final model training epoch
            )
            # Evaluate on test
            criterion = nn.CrossEntropyLoss()
            fold_test_results = evaluate_dataset(
                final_model, test_loader, criterion, DEVICE, 'full',
                fixed_window_size=30,
                fixed_shift_ratio=0.5,
                macro_in_dim=128,
                micro_in_dim=128,
                possible_window_sizes=possible_window_sizes,
                possible_shift_ratios=possible_shift_ratios,
                max_steps_per_trial=20,
                fold_idx=fold_idx,
                dataset_type='test',
                track_usage=False,
                verbose=False,
                save_dir=None
            )
            fold_test_metrics = {
                'acc': fold_test_results['acc'],
                'sen': fold_test_results['sen'],
                'spec': fold_test_results['spec'],
                'f1': fold_test_results['f1'],
                'auc': fold_test_results['auc']
            }
        else:
            final_metrics, final_model = train_evaluate_classifier_with_minibatch(
                best_hparams,
                combined_loader,
                combined_loader,  # Here we use train+val merged data only without val_loader
                DEVICE,
                epochs=10  # Example: final model training epoch
            )
            # Evaluate on test
            fold_test_metrics = evaluate_dataset_with_minibatch(
                final_model,
                test_loader,
                DEVICE,
                best_hparams['window_size'],
                best_hparams['step_size']
            )
        
        print("Fold test metrics:", fold_test_metrics)
        test_metrics_list.append(fold_test_metrics)

    # 4) Average test metrics across the 5 folds
    avg_test = {'acc': 0.0, 'sen': 0.0, 'spec': 0.0, 'f1': 0.0, 'auc': 0.0}
    for m in test_metrics_list:
        for k in avg_test.keys():
            avg_test[k] += m[k]
    for k in avg_test.keys():
        avg_test[k] /= len(test_metrics_list)

    print(f"\n{'='*60}")
    print("Final 5-Fold Average Test Metrics")
    print(f"{'='*60}")
    print(f"Model Type: {mode_name}")
    if not USE_HRL:
        print(f"Best Classifier Type: {best_hparams.get('classifier_type', 'crnn')}")
    print(f"Avg Accuracy:    {avg_test['acc']:.4f}")
    print(f"Avg Sensitivity: {avg_test['sen']:.4f}")
    print(f"Avg Specificity: {avg_test['spec']:.4f}")
    print(f"Avg F1-Score:    {avg_test['f1']:.4f}")
    print(f"Avg AUC:         {avg_test['auc']:.4f}")
    print(f"{'='*60}")
    print("\nBest Hyperparameters:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")
    print(f"\n{'='*60}")
    print("Done.")