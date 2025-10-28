import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser(description='Argparse')
timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")


path = os.getcwd()
# PATH
# Keep only the main parameters actually used in environment and training
parser.add_argument("--stop_point", default=115, type=int, help="Terminal point")
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument("--batch_size", default=72, type=int, help="batch size")
parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--dropout_rate', default=0.2, type=float, help='General dropout rate (may be used for RL or other components)')
parser.add_argument('--crnn_dropout', default=0.2, type=float, help='Dropout rate for CRNN classifier (default: 0.4)')
parser.add_argument("--discount_factor", default=0.95, type=float, help="Discount factor")
parser.add_argument("--epochs", default=2000, type=int, help="Classifier train epochs")
parser.add_argument("--patience", default=1500, type=int, help="Early stop point")
parser.add_argument('--val_interval', default=1, type=int, help='Epoch interval for running validation')
# Additional hyperparameters used in HRL.py
parser.add_argument('--epsilon_start', default=1.0, type=float, help='Starting epsilon for epsilon-greedy exploration')
parser.add_argument('--epsilon_end', default=0.05, type=float, help='Ending epsilon for epsilon-greedy exploration')
parser.add_argument('--epsilon_decay_ratio', default=0.2, type=float, help='Ratio of episodes for epsilon decay (20% as per paper)')
parser.add_argument('--rl_dropout', default=0.2, type=float, help='Dropout rate for RL modules (default: 0.2)')
# Reward shaping parameters
parser.add_argument('--max_diversity_history_len', default=20, type=int, help='Max history length for diversity bonus entropy calculation (default: 20)')
parser.add_argument('--diversity_bonus_weight', default=0.05, type=float, help='Weight for the diversity bonus (default: 0.05)')
parser.add_argument('--max_baseline_history_len', default=100, type=int, help='Max history length for reward baseline calculation (default: 100)')
parser.add_argument('--reward_min_val', default=-2.0, type=float, help='Minimum clipped reward value (default: -1.0)')
parser.add_argument('--reward_max_val', default=2.0, type=float, help='Maximum clipped reward value (default: 1.0)')
parser.add_argument('--track_usage', action='store_true', default=False, help='Log and save agent window/step/reward usage statistics')
parser.add_argument('--n_folds', default=5, type=int, help='Number of folds')
parser.add_argument('--focal_loss_alpha', default=0.75, type=float, help='Alpha parameter for Focal Loss (ensemble/ablation: try 0.25, 0.5, 0.75)')
parser.add_argument('--focal_loss_gamma', default=3.0, type=float, help='Gamma parameter for Focal Loss (ensemble/ablation: try 1.0, 2.0, 3.0)')
parser.add_argument('--focal_gamma', default=None, type=float, help='Override Focal Loss gamma (for experiment automation)')
parser.add_argument('--reward_nc_penalty', default=None, type=float, help='Override NC penalty reward (for experiment automation)')
parser.add_argument('--run_fold', type=int, default=None, help='Specify a single fold to run (0-indexed). If not provided, runs all folds.')
parser.add_argument('--seed', default=920507, type=int, help='Random seed for reproducibility')
parser.add_argument('--result_dir', type=str, default=None, help='Directory to save experiment results (for automation)')
parser.add_argument('--crnn_out_channels', default=32, type=int, help='CRNN out channels')
parser.add_argument('--crnn_kernel_size', default=3, type=int, help='CRNN kernel size')
parser.add_argument('--crnn_pool_size', default=2, type=int, help='CRNN pool size')
parser.add_argument('--crnn_hidden_dim', default=128, type=int, help='CRNN hidden dim')
parser.add_argument('--crnn_num_layers', default=2, type=int, help='CRNN num layers')
parser.add_argument('--crnn_num_classes', default=2, type=int, help='CRNN num classes')
parser.add_argument('--crnn_bidirectional', default=False, type=bool, help='Use bidirectional GRU (CBGRU) if True, else unidirectional (CRNN)')
parser.add_argument('--classifier_type', default='crnn', type=str, choices=['crnn', 'cbgru', 'transformer'], 
                    help='Classifier architecture: crnn (default), cbgru (bidirectional GRU), transformer (Spatio-Temporal Transformer)')
# Transformer-specific parameters
parser.add_argument('--transformer_num_heads', default=8, type=int, help='Number of attention heads in Transformer')
parser.add_argument('--transformer_num_layers', default=4, type=int, help='Number of Transformer encoder layers')
parser.add_argument('--transformer_dim_feedforward', default=512, type=int, help='Dimension of feedforward network in Transformer')
parser.add_argument('--transformer_dropout', default=0.1, type=float, help='Dropout rate for Transformer')
parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim (ensemble/ablation: try 64, 116, 128)')
parser.add_argument('--macro_in_dim', default=1024, type=int, help='Macro input dim (ensemble/ablation: try 64, 128, 1024)')
parser.add_argument('--micro_in_dim', default=1024, type=int, help='Micro input dim (ensemble/ablation: try 64, 128, 1024)')
parser.add_argument('--rl_hidden_dim', default=1024, type=int, help='RL hidden dim (5×1024 as per paper)')
parser.add_argument('--rl_embed_dim', default=1024, type=int, help='RL embed dim (5×1024 as per paper)')
parser.add_argument('--rl_target_update_freq', default=100, type=int, help='RL target update freq (every 100 steps as per paper)')
parser.add_argument('--buffer_capacity', default=10000, type=int, help='Replay buffer capacity (10k as per paper)')
parser.add_argument('--rl_batch_size', default=72, type=int, help='RL batch size (same as classifier batch size)')
parser.add_argument('--num_worker', default=0, type=int, help='Number of workers for DataLoader')
# Additional: PCA dimension, threshold for preprocessing
parser.add_argument('--pca_dim', default=256, type=int, help='PCA dimension reduction size')
parser.add_argument('--pca_use', default=False, type=bool, help='Use PCA dimension reduction')
parser.add_argument('--lda_dim', default=256, type=int, help='LDA dimension reduction size')
parser.add_argument('--lda_use', default=True, type=bool, help='Use LDA dimension reduction')
parser.add_argument('--threshold', default=0.05, type=float, help='z-score thresholding threshold value')
parser.add_argument('--tau', default=0.001, type=float, help='Soft update coefficient for target network (EMA)')
parser.add_argument('--classifier_clip_norm', default=6.0, type=float, help='Gradient clipping norm for Classifier')
parser.add_argument('--macroq_clip_norm', default=6.0, type=float, help='Gradient clipping norm for MacroQ')
parser.add_argument('--microq_clip_norm', default=6.0, type=float, help='Gradient clipping norm for MicroQ')

# Ablation study related parameters
parser.add_argument('--ablation_mode', type=str, default='full', 
                    choices=['full', 'no_macro', 'no_micro', 'fixed'], 
                    help='Ablation study mode: full(default), no_macro, no_micro, fixed')
parser.add_argument('--fixed_window_size', type=int, default=30, 
                    help='Fixed window size(in timepoints) when macro agent is disabled')
parser.add_argument('--fixed_shift_ratio', type=float, default=0.5, 
                    help='Fixed shift ratio when micro agent is disabled')

# Inference related parameters
parser.add_argument('--inference_only', action='store_true',
                    help='Run only inference without training')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to pretrained model weights for inference')
parser.add_argument('--model_fold', type=int, default=1,
                    help='Fold number of the model to use for inference (1-5)')

def get_config():
    return parser.parse_args()

# device
parser.add_argument("--cuda_device", default=0, type=int, help="Cuda")

