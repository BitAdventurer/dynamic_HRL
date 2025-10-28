"""
train.py: Hierarchical RL training execution script

- Contains the complete code from HRL.py's train_hierarchical_dqn()
- Runs independently without importing HRL.py
"""
import sys
import os
# Ensure project root is in sys.path BEFORE local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now import local modules
from main import save_results
from main.HRL import DFCSequenceDataset, ReplayBuffer, evaluate_dataset, compute_macro_dqn_loss, compute_micro_bandit_loss, HRL, update_metrics, calc_scores
from main.models import CRNNClassifier, MacroQNet, MicroQNet
import data.Signal_cv as Signal_cv
from utils.preprocessing import preprocess_matrix
import utils.util as util
from utils.seed import seed_everything
from config import get_config

# Standard library and third-party imports
import csv
from collections import defaultdict
import json
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import logging
from datetime import datetime
# === [Recommended] Fix seed only once for the entire experiment ===

args = get_config()
# Override focal_loss_gamma if --focal_gamma provided
if getattr(args, 'focal_gamma', None) is not None:
    args.focal_loss_gamma = args.focal_gamma
# Override NC penalty if --reward_nc_penalty provided
if getattr(args, 'reward_nc_penalty', None) is not None:
    args.reward_nc_penalty_effective = args.reward_nc_penalty
else:
    args.reward_nc_penalty_effective = -1.0  # default used in reward calculation
# Non-deterministic epsilon-greedy exploration: do NOT fix random.seed
seed_everything(args.seed, skip_random=True)

print(f"[CONFIG] Focal Loss gamma: {args.focal_loss_gamma}")
print(f"[CONFIG] NC penalty reward: {args.reward_nc_penalty_effective}")

DATA_PATH = 'data'


# Convenience variable definitions - these values are already parsed in config.py
if __name__ == "__main__":
    # config.py's get_config() parses command line arguments
    INFERENCE_ONLY = args.inference_only
    MODEL_PATH = args.model_path
    MODEL_FOLD = args.model_fold

# Main hyperparameters and constants definition (same as HRL.py)
DEVICE = torch.device(
    f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
EPOCHS = args.epochs
N_FOLDS = args.n_folds
DATA_BATCH_SIZE = args.batch_size
NUM_WORKER = args.num_worker
USE_GPU_DATA = False
FEATURE_DIM = args.feature_dim
CRNN_HIDDEN_DIM = args.crnn_hidden_dim
CRNN_NUM_LAYERS = args.crnn_num_layers
CRNN_DROPOUT = args.dropout_rate  # This line is maintained, but args.dropout_rate is used directly when creating models
CRNN_NUM_CLASSES = args.crnn_num_classes
CRNN_OUT_CHANNELS = args.crnn_out_channels
CRNN_KERNEL_SIZE = args.crnn_kernel_size
CRNN_POOL_SIZE = args.crnn_pool_size
CRNN_LR = args.lr
CRNN_WD = args.wd
MACRO_IN_DIM = args.macro_in_dim
MICRO_IN_DIM = args.micro_in_dim
RL_HIDDEN_DIM = args.rl_hidden_dim
RL_EMBED_DIM = args.rl_embed_dim
RL_BATCH_SIZE = args.rl_batch_size
RL_BUFFER_CAPACITY = args.buffer_capacity
RL_GAMMA = args.discount_factor
RL_TARGET_UPDATE_FREQ = args.rl_target_update_freq
VAL_INTERVAL = args.val_interval
MACRO_LR = args.lr
MICRO_LR = args.lr
MACRO_WD = args.wd if hasattr(args, 'wd') else 1e-5
MICRO_WD = args.wd if hasattr(args, 'wd') else 1e-5
# Paper-aligned: absolute window sizes (1 TRs interval)
possible_window_sizes = list(range(10, 233, 1))  #TRs
possible_shift_ratios = [i / 100 for i in range(1, 101)]  # 0.01~1.00
ES_PATIENCE = args.patience
MAX_STEPS_PER_TRIAL = 50
DROP_RATE = args.dropout_rate

# Logging system setup


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # Weighting factor for balancing classes (e.g., 0.25 for positive
        # class)
        self.alpha = alpha
        self.gamma = gamma  # Focusing parameter (e.g., 2.0)
        self.reduction = reduction

    def forward(self, logits, labels):
        # logits: model predictions (tensor of shape [N, C])
        # labels: ground truth labels (tensor of shape [N])

        # Calculate Cross Entropy loss for each instance (without reduction)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # Calculate pt (probability of the true class)
        # pt = exp(-ce_loss) because ce_loss = -log(pt)
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss: alpha_t * (1-pt)^gamma * ce_loss
        # For simplicity, using scalar alpha. If per-class alpha is needed,
        # self.alpha should be a tensor and gathered using labels.
        f_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:  # 'none'
            return f_loss


def setup_logger(args=None):
    # Create log directory
    os.makedirs('logs', exist_ok=True)

    # Include current time in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Include ablation mode in filename if available
    if args is not None and hasattr(args, 'ablation_mode'):
        log_file = f'logs/train_{args.ablation_mode}_{timestamp}.log'
    else:
        log_file = f'logs/train_{timestamp}.log'

    # Logger setup
    logger = logging.getLogger('hrl_training')
    logger.setLevel(logging.INFO)

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Format setup
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger declaration (initialized later)
logger = None


def calculate_reward_for_step(classifier_output, true_label, device, ablation_mode):
    """
    Calculates the reward for a single step based on classifier output and true label.
    Incorporates class-specific base rewards and a confidence-based multiplier.

    Args:
        classifier_output (torch.Tensor): Raw output from the classifier for a single instance (shape [1, num_classes]).
        true_label (int): The true label for the instance (0 for NC, 1 for MDD).
        device (torch.device): The device to perform calculations on.
        ablation_mode (str): Ablation mode for reward logic.

    Returns:
        float: Reward value for the step.
    """
    # Softmax to get class probabilities
    prob = F.softmax(classifier_output, dim=1)
    confidence_value, predicted_class = torch.max(prob, dim=1)
    confidence_value = confidence_value.item()
    predicted_class = predicted_class.item()

    # Class-specific base rewards (with override for NC penalty)
    nc_penalty = getattr(args, 'reward_nc_penalty_effective', -1.0)
    # True Negative (NC): correct NC prediction
    if predicted_class == 0 and true_label == 0:
        base_reward = 1.0
    # True Positive (MDD): correct MDD prediction
    elif predicted_class == 1 and true_label == 1:
        base_reward = 0.5
    # False Positive: Incorrectly predicting MDD for an NC patient (label 0, pred 1)
    elif predicted_class == 1 and true_label == 0:
        base_reward = nc_penalty
    # False Negative: Incorrectly predicting NC for an MDD patient (label 1, pred 0)
    elif predicted_class == 0 and true_label == 1:
        base_reward = -0.3

    # Confidence multiplier: scales from 0.5 (for 0% confidence) to 1.5 (for 100% confidence)
    confidence_multiplier = 0.5 + confidence_value

    final_reward = base_reward * confidence_multiplier
    return final_reward



def calculate_adaptive_clip_norm(model_type, current_step, total_steps, model):
    """
    Calculate adaptive gradient clipping norm based on various factors

    Args:
        model_type: Model type ('classifier', 'macro', 'micro')
        current_step: Current iteration count (step) or epoch
        total_steps: Total expected iteration count or maximum epoch
        model: Model object

    Returns:
        Adaptively calculated clip_norm value
    """
    # Basic setup values (defined in config.py)
    base_clip_norm = {
        'classifier': args.classifier_clip_norm,
        'macro': args.macroq_clip_norm,
        'micro': args.microq_clip_norm
    }[model_type]

    # 1. Model size-based scaling
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Scaling factor based on model size
    # More parameters may require larger clip_norm
    size_factor = 1.0 + (np.log(param_count) / 20)  # Log scaling

    # 2. Learning rate-based scaling
    # Apply stronger gradient clipping when learning rate is higher
    lr = args.lr
    lr_factor = 1.0 / (lr * 10000)  # Inversely proportional to learning rate

    # 3. Current progress-based scaling (larger initially, gradually decreases)
    # Set larger in early training, smaller in later training for stability
    # Ensure progress_ratio is between 0 and 1
    progress_ratio = min(1.0, max(0.0, current_step / max(1, total_steps)))
    progress_factor = 1.0 - (0.5 * progress_ratio)  # Decrease by up to 50%

    # Combine all factors to calculate final clip_norm
    adaptive_clip_norm = base_clip_norm * size_factor * lr_factor * progress_factor

    # Range limits to prevent too small or large values
    min_clip_norm = base_clip_norm * 0.5  # 50% of base value
    max_clip_norm = base_clip_norm * 2.0  # 200% of base value

    # Logging (every 100 steps or every epoch)
    # if current_step % 100 == 0 or current_step % total_steps == 0:
    #     logger.info(f"Adaptive {model_type}_clip_norm: {adaptive_clip_norm:.2f} "
    #                f"(base: {base_clip_norm}, size_factor: {size_factor:.2f}, "
    # f"lr_factor: {lr_factor:.2f}, progress_factor: {progress_factor:.2f})")

    return np.clip(adaptive_clip_norm, min_clip_norm, max_clip_norm)


def train_hierarchical_dqn(args, fold_idx, train_loader, val_loader, test_loader, pids_train, pids_val, pids_test):
    # 결과 저장을 위한 디렉토리 생성
    if getattr(args, 'result_dir', None):
        save_dir = args.result_dir
    else:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        ablation_suffix = f"_{args.ablation_mode}" if args.ablation_mode != 'none' else ""
        save_dir = f'results/run_{current_time}{ablation_suffix}'
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"[EXEC_TRACE] Save directory created: {save_dir}")

    # === Epoch별 metric 로그 저장용 CSV 초기화 ===
    epoch_metrics_log_path = os.path.join(save_dir, 'epoch_metrics_log.csv')
    with open(epoch_metrics_log_path, 'w') as f:
        f.write('Fold,Epoch,Phase,Loss,Acc,Sen,Spec,F1,AUC\n')

    # 전역 변수 접근
    global INFERENCE_ONLY, MODEL_PATH, MODEL_FOLD
    inference_mode = INFERENCE_ONLY
    model_path = MODEL_PATH
    selected_fold = MODEL_FOLD
    
    # Determine classifier type from args
    if args.classifier_type == 'transformer':
        classifier_name = "Spatio-Temporal Transformer"
    elif args.classifier_type == 'cbgru':
        classifier_name = "CBGRU (Bidirectional GRU)"
    else:  # 'crnn'
        classifier_name = "CRNN" + (" (Bidirectional)" if args.crnn_bidirectional else "")

    logger.info(f"Starting train_hierarchical_dqn. Device: {DEVICE}, Ablation: {args.ablation_mode}")
    logger.info(f"Classifier: {classifier_name}, Epochs: {EPOCHS}, Batch Size: {DATA_BATCH_SIZE}, LR: {CRNN_LR}")

    # Loss Function
    criterion = FocalLoss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma, reduction='none').to(DEVICE)
    logger.info(f"Using FocalLoss with alpha={args.focal_loss_alpha}, gamma={args.focal_loss_gamma}")

    # Determine which folds to run
    if args.run_fold is not None:
        if 0 <= args.run_fold < N_FOLDS:
            folds_to_run = [args.run_fold + 1]
            logger.info(f"Running a single specified fold: {args.run_fold} (data fold {folds_to_run[0]})")
        else:
            logger.error(f"Invalid fold index: {args.run_fold}. Must be between 0 and {N_FOLDS - 1}.")
            return
    else:
        folds_to_run = range(1, N_FOLDS + 1)
        logger.info(f"Running full {N_FOLDS}-fold cross-validation.")

    # Lists to aggregate results across all folds
    all_results = []
    early_stop_test_metrics = []

    logger.info("[EXEC_TRACE] Entering main cross-validation loop...")

    # === Policy selection logging ===
    policy_log = []
    for fold_idx in folds_to_run:
        logger.info(f'\n==================== Fold {fold_idx}/{N_FOLDS} ====================')
        logger.info(f"[EXEC_TRACE] Start of Fold {fold_idx}.")
        model_save_path = os.path.join(save_dir, f'best_model_fold_{fold_idx}.pth')
        
        # --- Per-fold variables ---
        best_val_auc = 0.0
        patience_counter = 0
        fold_best_metrics_dict = {}

        try:
            logger.info("[EXEC_TRACE] Loading data for fold.")
            split_data_path = os.path.join(DATA_PATH, 'Signal_split')
            try:
                train_data = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_train_data.npy'), allow_pickle=True)
                train_labels = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_train_labels.npy'), allow_pickle=True)
                train_pids = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_train_patient_ids.npy'), allow_pickle=True)
                val_data = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_val_data.npy'), allow_pickle=True)
                val_labels = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_val_labels.npy'), allow_pickle=True)
                val_pids = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_val_patient_ids.npy'), allow_pickle=True)
                test_data = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_test_data.npy'), allow_pickle=True)
                test_labels = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_test_labels.npy'), allow_pickle=True)
                test_pids = np.load(os.path.join(split_data_path, f'fold_{fold_idx}_test_patient_ids.npy'), allow_pickle=True)
            except FileNotFoundError as e:
                logger.error(f"Data loading failed for fold {fold_idx}. File not found: {e}. Skipping fold.")
                continue

            train_dataset = DFCSequenceDataset(train_data, train_labels, train_pids, use_pca=True, pca_dim=args.feature_dim)
            val_dataset = DFCSequenceDataset(val_data, val_labels, val_pids, use_pca=True, pca_dim=args.feature_dim)
            test_dataset = DFCSequenceDataset(test_data, test_labels, test_pids, use_pca=True, pca_dim=args.feature_dim)

            # === Input dimension debugging/logging ===
            # Check shape of one sample (train)
            sample_data, _, _ = train_dataset[0]
            logger.info(f"[DEBUG] train_dataset[0] shape: {sample_data.shape} (should be [T, feature_dim])")
            logger.info(f"[DEBUG] args.feature_dim: {args.feature_dim}")
            # Assert feature_dim matches
            if hasattr(sample_data, 'shape') and len(sample_data.shape) == 2:
                actual_feature_dim = sample_data.shape[-1]
                assert actual_feature_dim == args.feature_dim, (
                    f"Input dimension mismatch: DFCSequenceDataset feature_dim={actual_feature_dim}, "
                    f"model expects feature_dim={args.feature_dim}. "
                    "Check PCA/LDA config and set feature_dim accordingly.")
            else:
                logger.warning("[DEBUG] Could not determine feature_dim from sample_data shape.")

            train_loader = DataLoader(train_dataset, batch_size=DATA_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
            val_loader = DataLoader(val_dataset, batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
            test_loader = DataLoader(test_dataset, batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
            logger.info(f"DataLoaders created for fold {fold_idx}.")

            # --- Model, Optimizer, Scheduler Initialization ---
            logger.info("[EXEC_TRACE] Initializing models and optimizers.")
            hrl_model = HRL(
                feature_dim=args.feature_dim,
                crnn_hidden_dim=args.crnn_hidden_dim,
                crnn_num_layers=args.crnn_num_layers,
                crnn_dropout=args.crnn_dropout,
                crnn_num_classes=args.crnn_num_classes,
                crnn_out_channels=args.crnn_out_channels,
                crnn_kernel_size=args.crnn_kernel_size,
                crnn_pool_size=args.crnn_pool_size,
                macro_in_dim=MACRO_IN_DIM,
                micro_in_dim=MICRO_IN_DIM,
                rl_hidden_dim=args.rl_hidden_dim,
                rl_embed_dim=args.rl_embed_dim,
                num_macro_actions=len(possible_window_sizes),
                num_micro_actions=len(possible_shift_ratios),
                rl_dropout=args.rl_dropout,
                crnn_bidirectional=args.crnn_bidirectional,
                classifier_type=args.classifier_type,
                transformer_num_heads=args.transformer_num_heads,
                transformer_dim_feedforward=args.transformer_dim_feedforward,
                transformer_dropout=args.transformer_dropout
            ).to(DEVICE)
            
            # Create separate optimizers for each component
            classifier_optimizer = optim.Adam(hrl_model.classifier.parameters(), lr=CRNN_LR, weight_decay=CRNN_WD)
            macro_optimizer = optim.Adam(hrl_model.macroQ.parameters(), lr=MACRO_LR, weight_decay=MACRO_WD)
            micro_optimizer = optim.Adam(hrl_model.microQ.parameters(), lr=MICRO_LR, weight_decay=MICRO_WD)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, 'max', factor=0.5, patience=5)
            
            # Create target networks for Macro Q-learning (Double DQN)
            macroQ_target1 = MacroQNet(MACRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, len(possible_window_sizes), dropout=args.rl_dropout).to(DEVICE)
            macroQ_target2 = MacroQNet(MACRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, len(possible_window_sizes), dropout=args.rl_dropout).to(DEVICE)
            macroQ_target1.load_state_dict(hrl_model.macroQ.state_dict())
            macroQ_target2.load_state_dict(hrl_model.macroQ.state_dict())
            macroQ_target1.eval()
            macroQ_target2.eval()
            
            # Create replay buffers
            macro_buffer = ReplayBuffer(RL_BUFFER_CAPACITY)
            micro_buffer = ReplayBuffer(RL_BUFFER_CAPACITY)
            
            logger.info("Optimizers, target networks, and replay buffers are ready.")

            # === Epoch Loop ===
            logger.info(f"[EXEC_TRACE] Starting epoch loop for fold {fold_idx}.")
            # === Gradient Norm Logging Initialization ===
            gradnorm_csv_path = os.path.join(save_dir, 'gradient_norms.csv')
            gradnorm_header_written = False
            gradnorms_per_epoch = []

            for epoch in range(EPOCHS):
                logger.info(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
                hrl_model.train()
                epoch_loss = 0.0
                # Gradient norm accumulators for this epoch
                gradnorm_classifier = []
                gradnorm_macroQ = []
                gradnorm_microQ = []
                
                # --- Per-label loss accumulators ---
                per_class_loss_sum = {0: 0.0, 1: 0.0}
                per_class_count = {0: 0, 1: 0}

                for i, (batch_data, batch_labels, batch_pids) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
                    batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
                    classifier_optimizer.zero_grad()
                    outputs = hrl_model.classifier(batch_data)
                    # Compute unreduced loss for per-sample analysis
                    loss_all = criterion(outputs, batch_labels)
                    if loss_all.dim() == 0 or loss_all.shape[0] == 1:
                        # Scalar loss (should not happen with reduction='none')
                        loss = loss_all
                        loss_per_sample = loss_all.expand_as(batch_labels)
                    else:
                        loss = loss_all.mean()
                        loss_per_sample = loss_all
                    loss.backward()
                    # --- Policy selection logging (train RL step) ---
                    if args.ablation_mode == 'full':
                        # For each sample in batch, log the policy (window/step) selection
                        for j in range(batch_data.size(0)):
                            # Here we log only the initial window/step selection for each episode (can be extended to per-step if multi-step RL is used)
                            # For more granular logging, RL step loop should be instrumented similarly to evaluate_dataset
                            # For now, we log batch index as step, and don't have access to actual RL agent's window/step here
                            policy_log.append({
                                'fold': fold_idx,
                                'epoch': epoch+1,
                                'phase': 'train',
                                'patient_id': batch_pids[j] if hasattr(batch_pids, '__getitem__') else 'NA',
                                'label': int(batch_labels[j].item()),
                                'step': i,
                                'window': 'NA',
                                'step_ratio': 'NA',
                                'reward': 'NA',
                                'prediction': int(outputs[j].argmax().item()),
                                'confidence': float(torch.softmax(outputs[j], dim=0)[int(outputs[j].argmax().item())].item())
                            })
                    # --- Per-label loss accumulation ---
                    for lbl in [0, 1]:
                        mask = (batch_labels == lbl)
                        if mask.sum().item() > 0:
                            per_class_loss_sum[lbl] += loss_per_sample[mask].sum().item()
                            per_class_count[lbl] += mask.sum().item()
                    # --- Gradient Norm Calculation (before optimizer.step) ---
                    def grad_norm(module):
                        total_norm = 0.0
                        for p in module.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2).item()
                                total_norm += param_norm ** 2
                        return total_norm ** 0.5
                    gradnorm_classifier.append(grad_norm(hrl_model.classifier))
                    
                    # Classifier gradient clipping and step
                    torch.nn.utils.clip_grad_norm_(hrl_model.classifier.parameters(), args.classifier_clip_norm)
                    classifier_optimizer.step()
                    epoch_loss += loss.item()
                    
                    # === RL Training: Macro Q-Network ===
                    if len(macro_buffer) >= RL_BATCH_SIZE and args.ablation_mode == 'full':
                        macro_batch = macro_buffer.sample(RL_BATCH_SIZE)
                        macro_loss = compute_macro_dqn_loss(
                            macro_batch, hrl_model.macroQ, macroQ_target1, macroQ_target2, RL_GAMMA, DEVICE
                        )
                        macro_optimizer.zero_grad()
                        macro_loss.backward()
                        torch.nn.utils.clip_grad_norm_(hrl_model.macroQ.parameters(), args.macroq_clip_norm)
                        macro_optimizer.step()
                        gradnorm_macroQ.append(grad_norm(hrl_model.macroQ))
                    
                    # === RL Training: Micro Q-Network ===
                    if len(micro_buffer) >= RL_BATCH_SIZE and args.ablation_mode == 'full':
                        micro_batch = micro_buffer.sample(RL_BATCH_SIZE)
                        micro_loss = compute_micro_bandit_loss(micro_batch, hrl_model.microQ, DEVICE)
                        micro_optimizer.zero_grad()
                        micro_loss.backward()
                        torch.nn.utils.clip_grad_norm_(hrl_model.microQ.parameters(), args.microq_clip_norm)
                        micro_optimizer.step()
                        gradnorm_microQ.append(grad_norm(hrl_model.microQ))
                # --- End of batch loop ---
                # Compute average per-label loss for the epoch
                avg_loss_nc = per_class_loss_sum[0] / per_class_count[0] if per_class_count[0] > 0 else 0.0
                avg_loss_mdd = per_class_loss_sum[1] / per_class_count[1] if per_class_count[1] > 0 else 0.0
                logger.info(f"[EPOCH {epoch}] Fold {fold_idx} | Train Loss NC: {avg_loss_nc:.4f} | Train Loss MDD: {avg_loss_mdd:.4f}")

                # --- Train phase metric 계산 (epoch 끝나고) ---
                from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
                all_train_preds = []
                all_train_labels = []
                all_train_probs = []
                hrl_model.classifier.eval()
                with torch.no_grad():
                    for batch_data, batch_labels, _ in train_loader:
                        batch_data = batch_data.to(DEVICE)
                        outputs = hrl_model.classifier(batch_data)
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        all_train_preds.extend(preds)
                        all_train_labels.extend(batch_labels.numpy())
                        all_train_probs.extend(probs)
                # metric 계산
                acc_train = accuracy_score(all_train_labels, all_train_preds)
                sen_train = recall_score(all_train_labels, all_train_preds, pos_label=1)
                spec_train = recall_score(all_train_labels, all_train_preds, pos_label=0)
                f1_train = f1_score(all_train_labels, all_train_preds)
                try:
                    auc_train = roc_auc_score(all_train_labels, all_train_probs)
                except:
                    auc_train = 0.5
                # Confusion matrix 저장
                cm = confusion_matrix(all_train_labels, all_train_preds, labels=[0,1])
                cm_path = os.path.join(save_dir, f'train_confusion_matrix_epoch{epoch+1}_fold{fold_idx}.csv')
                pd.DataFrame(cm, index=["NC (0)", "MDD (1)"], columns=["Pred NC (0)", "Pred MDD (1)"]).to_csv(cm_path)
                logger.info(f"[EPOCH {epoch}] Fold {fold_idx} | Train confusion matrix saved to {cm_path}")

                mean_classifier = np.mean(gradnorm_classifier) if gradnorm_classifier else 0.0
                mean_macroQ = np.mean(gradnorm_macroQ) if gradnorm_macroQ else 0.0
                mean_microQ = np.mean(gradnorm_microQ) if gradnorm_microQ else 0.0
                gradnorms_per_epoch.append({
                    'Epoch': epoch+1,
                    'Classifier': mean_classifier,
                    'MacroQ': mean_macroQ,
                    'MicroQ': mean_microQ
                })
                # Write/update CSV every epoch
                if not gradnorm_header_written:
                    with open(gradnorm_csv_path, 'w') as f:
                        f.write('Epoch,Classifier,MacroQ,MicroQ\n')
                    gradnorm_header_written = True
                with open(gradnorm_csv_path, 'a') as f:
                    f.write(f"{epoch+1},{mean_classifier},{mean_macroQ},{mean_microQ}\n")

                # 평균 loss로 변환 (batch 개수로 나눔)
                num_batches = len(train_loader)
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                
                # === Collect RL Transitions from Training Data ===
                if args.ablation_mode == 'full' and (epoch + 1) % 5 == 0:  # Every 5 epochs to save computation
                    logger.info(f"[EPOCH {epoch}] Collecting RL transitions from training data...")
                    train_eval_results = hrl_model.evaluate_and_get_rewards(
                        train_loader, criterion, DEVICE, args.ablation_mode,
                        fixed_window_size=args.fixed_window_size,
                        fixed_shift_ratio=args.fixed_shift_ratio,
                        macro_in_dim=MACRO_IN_DIM,
                        micro_in_dim=MICRO_IN_DIM,
                        possible_window_sizes=possible_window_sizes,
                        possible_shift_ratios=possible_shift_ratios,
                        max_steps_per_trial=MAX_STEPS_PER_TRIAL,
                        fold_idx=fold_idx,
                        dataset_type='train',
                        track_usage=False,
                        verbose=False,
                        save_dir=None
                    )
                    # Add transitions to replay buffers
                    for transition in train_eval_results.get('macro_transitions', []):
                        macro_buffer.push(transition)
                    for transition in train_eval_results.get('micro_transitions', []):
                        micro_buffer.push(transition)
                    logger.info(f"[EPOCH {epoch}] Added {len(train_eval_results.get('macro_transitions', []))} macro transitions, "
                               f"{len(train_eval_results.get('micro_transitions', []))} micro transitions. "
                               f"Buffer sizes: Macro={len(macro_buffer)}, Micro={len(micro_buffer)}")
                
                # === Target Network Update ===
                if (epoch + 1) % RL_TARGET_UPDATE_FREQ == 0 and args.ablation_mode == 'full':
                    macroQ_target1.load_state_dict(hrl_model.macroQ.state_dict())
                    macroQ_target2.load_state_dict(hrl_model.macroQ.state_dict())
                    logger.info(f"[EPOCH {epoch}] Target networks updated")

                # --- Validation, Early Stopping, Model Saving ---
                if (epoch + 1) % VAL_INTERVAL == 0:
                    val_results = hrl_model.evaluate_and_get_rewards(
                        val_loader, criterion, DEVICE, args.ablation_mode,
                        fixed_window_size=args.fixed_window_size,
                        fixed_shift_ratio=args.fixed_shift_ratio,
                        macro_in_dim=MACRO_IN_DIM,
                        micro_in_dim=MICRO_IN_DIM,
                        possible_window_sizes=possible_window_sizes,
                        possible_shift_ratios=possible_shift_ratios,
                        max_steps_per_trial=MAX_STEPS_PER_TRIAL,
                        fold_idx=fold_idx,
                        dataset_type='validation',
                        track_usage=args.track_usage,
                        verbose=True,
                        save_dir=save_dir
                    )
                    logger.info(
    f"Validation | Acc: {val_results['acc']:.4f}, Sen: {val_results['sen']:.4f}, "+
    f"Spec: {val_results['spec']:.4f}, F1: {val_results['f1']:.4f}, AUC: {val_results['auc']:.4f}")

                    scheduler.step(val_results['auc'])

                    if val_results['auc'] > best_val_auc:
                        best_val_auc = val_results['auc']
                        patience_counter = 0
                        logger.info(f"✓ New best validation AUC: {best_val_auc:.4f}. Saving model to {model_save_path}")
                        torch.save(hrl_model.state_dict(), model_save_path)

                        logger.info("Evaluating on test set with the new best model...")
                        test_results = hrl_model.evaluate_and_get_rewards(
                            test_loader, criterion, DEVICE, args.ablation_mode,
                            fixed_window_size=args.fixed_window_size,
                            fixed_shift_ratio=args.fixed_shift_ratio,
                            macro_in_dim=MACRO_IN_DIM,
                            micro_in_dim=MICRO_IN_DIM,
                            possible_window_sizes=possible_window_sizes,
                            possible_shift_ratios=possible_shift_ratios,
                            max_steps_per_trial=MAX_STEPS_PER_TRIAL,
                            fold_idx=fold_idx,
                            dataset_type='test',
                            track_usage=args.track_usage,
                            verbose=False,
                            save_dir=save_dir
                        )
                        fold_best_metrics_dict = {
                            'fold': fold_idx,
                            'val_auc': val_results['auc'], 'val_acc': val_results['acc'],
                            'test_auc': test_results['auc'], 'test_acc': test_results['acc'],
                            'test_sen': test_results['sen'], 'test_spec': test_results['spec']
                        }
                        logger.info(f"Test results for best model in fold {fold_idx}: {fold_best_metrics_dict}")

                    else:
                        patience_counter += 1
                        logger.info(f"Validation AUC did not improve from {best_val_auc:.4f}. Patience: {patience_counter}/{ES_PATIENCE}")
                        if patience_counter >= ES_PATIENCE:
                            logger.info("Early stopping triggered.")
                            break

                # === [Epoch별 로그/저장] ===
                # 평균 loss로 변환 (batch 개수로 나눔)
                num_batches = len(train_loader)
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                logger.info(f"[EPOCH {epoch}] Fold {fold_idx} | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_results['loss']:.4f} | Test Loss: {test_results['loss']:.4f}")
                logger.info(f"[EPOCH {epoch}] Fold {fold_idx} | Train Acc: {acc_train:.4f} | Val Acc: {val_results['acc']:.4f} | Test Acc: {test_results['acc']:.4f}")
                logger.info(f"[EPOCH {epoch}] Fold {fold_idx} | Train AUC: {auc_train:.4f} | Val AUC: {val_results['auc']:.4f} | Test AUC: {test_results['auc']:.4f}")

                # === [Epoch별 metric 로그 CSV 저장] ===
                with open(epoch_metrics_log_path, 'a') as f:
                    # Log per-label train loss as extra columns
                    f.write(f'{fold_idx},{epoch},train,{avg_epoch_loss},{acc_train},{sen_train},{spec_train},{f1_train},{auc_train},{avg_loss_nc},{avg_loss_mdd}\n')
                    f.write(f'{fold_idx},{epoch},val,{val_results["loss"]},{val_results["acc"]},{val_results["sen"]},{val_results["spec"]},{val_results["f1"]},{val_results["auc"]},,\n')
                    f.write(f'{fold_idx},{epoch},test,{test_results["loss"]},{test_results["acc"]},{test_results["sen"]},{test_results["spec"]},{test_results["f1"]},{test_results["auc"]},,\n')

                if patience_counter >= ES_PATIENCE:
                    logger.warning(f"Early stopping triggered at epoch {epoch+1}.")
                    break # Exit epoch loop

        except Exception as e:
            logger.error(f"An error occurred during training for fold {fold_idx}: {e}", exc_info=True)
        
        finally:
            if fold_best_metrics_dict:
                all_results.append(fold_best_metrics_dict)
                early_stop_test_metrics.append(fold_best_metrics_dict) # Assuming we save the best
            logger.info(f"[EXEC_TRACE] End of Fold {fold_idx}.")

    # === End of Cross-Validation Loop ===
    logger.info("[EXEC_TRACE] Exited main cross-validation loop.")

    # --- Save policy selection log ---
    if len(policy_log) > 0:
        policy_log_path = os.path.join(save_dir, 'policy_log.csv')

        pd.DataFrame(policy_log).to_csv(policy_log_path, index=False)
        logger.info(f"Policy selection log saved to {policy_log_path}")

    # --- Result Aggregation and Saving ---
    if all_results:
        logger.info("Aggregating and saving results from all folds...")
        df_all_results = pd.DataFrame(all_results)
        results_csv_path = os.path.join(save_dir, 'training_val_test_results_allfolds.csv')
        df_all_results.to_csv(results_csv_path, index=False)
        logger.info(f"✓ All fold results saved to {results_csv_path}")

    else:
        logger.warning("No results were generated to save.")

    logger.info("[EXEC_TRACE] End of train_hierarchical_dqn function.")

if __name__ == "__main__":

    # 로거 초기화 (이제 ablation_mode가 로그 파일명에 포함됨)
    logger = setup_logger(args)
    
    # 시작 로그 출력
    logger.info("Starting Hierarchical RL training script")
    logger.info(f"Ablation Mode: {args.ablation_mode}")


    
    try:
        # --- Prepare data for the specified fold ---
        fold_idx = 1  # Hardcoded for immediate test, since user requested to use fold 1
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'Signal_split')
        
        def npy_path(name):
            return os.path.join(data_dir, f'fold_{fold_idx}_{name}.npy')
        
        train_data = np.load(npy_path('train_data'))
        train_labels = np.load(npy_path('train_labels'))
        train_pids = np.load(npy_path('train_patient_ids'))
        val_data = np.load(npy_path('val_data'))
        val_labels = np.load(npy_path('val_labels'))
        val_pids = np.load(npy_path('val_patient_ids'))
        test_data = np.load(npy_path('test_data'))
        test_labels = np.load(npy_path('test_labels'))
        test_pids = np.load(npy_path('test_patient_ids'))

        # Create datasets and loaders
        train_dataset = DFCSequenceDataset(train_data, train_labels, train_pids, device_if_gpu=DEVICE if USE_GPU_DATA else None)
        val_dataset = DFCSequenceDataset(val_data, val_labels, val_pids, device_if_gpu=DEVICE if USE_GPU_DATA else None)
        test_dataset = DFCSequenceDataset(test_data, test_labels, test_pids, device_if_gpu=DEVICE if USE_GPU_DATA else None)
        
        train_loader = DataLoader(train_dataset, batch_size=DATA_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
        val_loader = DataLoader(val_dataset, batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)
        test_loader = DataLoader(test_dataset, batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

        # Call training function with all required arguments
        train_hierarchical_dqn(args, fold_idx, train_loader, val_loader, test_loader, train_pids, val_pids, test_pids)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Error details:", exc_info=True)
        raise
