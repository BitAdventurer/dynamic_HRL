import torch
import os

# Model classes import (must match classes defined in HRL.py and models.py)
from HRL import MacroQNet, MicroQNet, CRNNClassifier
from config import get_config
from models import TemporalEmbedding, ROIEmbedding, MacroQNet, MicroQNet, CRNNClassifier, SpatioTemporalTransformer

# Environment/hyperparameter settings (same as HRL.py)
args = get_config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = args.feature_dim
MACRO_IN_DIM = args.macro_in_dim
MICRO_IN_DIM = args.micro_in_dim
RL_HIDDEN_DIM = args.rl_hidden_dim
RL_EMBED_DIM = args.rl_embed_dim
CRNN_OUT_CHANNELS = args.crnn_out_channels
CRNN_KERNEL_SIZE = args.crnn_kernel_size
CRNN_POOL_SIZE = args.crnn_pool_size
CRNN_HIDDEN_DIM = args.crnn_hidden_dim
CRNN_NUM_LAYERS = args.crnn_num_layers
CRNN_DROPOUT = args.dropout_rate
CRNN_NUM_CLASSES = args.crnn_num_classes

# Transformer-specific parameters
TRANSFORMER_NUM_HEADS = args.transformer_num_heads
TRANSFORMER_NUM_LAYERS = args.transformer_num_layers
TRANSFORMER_DIM_FEEDFORWARD = args.transformer_dim_feedforward
TRANSFORMER_DROPOUT = args.transformer_dropout

# Action space sizes for Q-Networks (not actual ratios)
macro_action_space_size = 100  # Number of possible window size actions
micro_action_space_size = 100  # Number of possible shift ratio actions

# Helper function to create appropriate classifier
def create_classifier(filename):
    """
    Create appropriate classifier based on filename pattern
    Supports CRNN, CBGRU, and Transformer architectures
    """
    if 'transformer' in filename.lower():
        return SpatioTemporalTransformer(
            input_dim=FEATURE_DIM,
            hidden_dim=CRNN_HIDDEN_DIM,
            num_layers=TRANSFORMER_NUM_LAYERS,
            num_heads=TRANSFORMER_NUM_HEADS,
            dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
            dropout=TRANSFORMER_DROPOUT,
            num_classes=CRNN_NUM_CLASSES
        )
    elif 'cbgru' in filename.lower():
        return CRNNClassifier(
            input_dim=FEATURE_DIM,
            hidden_dim=CRNN_HIDDEN_DIM,
            num_layers=CRNN_NUM_LAYERS,
            dropout=CRNN_DROPOUT,
            num_classes=CRNN_NUM_CLASSES,
            out_channels=CRNN_OUT_CHANNELS,
            kernel_size=CRNN_KERNEL_SIZE,
            pool_size=CRNN_POOL_SIZE,
            bidirectional=True
        )
    else:  # Default: CRNN
        return CRNNClassifier(
            input_dim=FEATURE_DIM,
            hidden_dim=CRNN_HIDDEN_DIM,
            num_layers=CRNN_NUM_LAYERS,
            dropout=CRNN_DROPOUT,
            num_classes=CRNN_NUM_CLASSES,
            out_channels=CRNN_OUT_CHANNELS,
            kernel_size=CRNN_KERNEL_SIZE,
            pool_size=CRNN_POOL_SIZE,
            bidirectional=False
        )

# Target file list for conversion
model_dir = os.path.join(os.path.dirname(__file__), '../models/weight')
file_map = []

for i in range(1, 6):
    # HRL components
    file_map.append((f"macroQ_fold{i}.pt", MacroQNet, (MACRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, macro_action_space_size)))
    file_map.append((f"microQ_fold{i}.pt", MicroQNet, (MICRO_IN_DIM, RL_HIDDEN_DIM, RL_EMBED_DIM, micro_action_space_size)))
    
    # Multiple classifier types
    file_map.append((f"classifier_fold{i}.pt", create_classifier(f"classifier_fold{i}.pt"), None))  # Will be created dynamically
    file_map.append((f"crnn_classifier_fold{i}.pt", create_classifier(f"crnn_classifier_fold{i}.pt"), None))
    file_map.append((f"cbgru_classifier_fold{i}.pt", create_classifier(f"cbgru_classifier_fold{i}.pt"), None))
    file_map.append((f"transformer_classifier_fold{i}.pt", create_classifier(f"transformer_classifier_fold{i}.pt"), None))

for fname, cls, ctor_args in file_map:
    fpath = os.path.join(model_dir, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] {fpath} (not found)")
        continue
    print(f"[CONVERT] {fpath}")
    model = torch.load(fpath, map_location=DEVICE)
    # New state_dict filename
    state_dict_path = fpath.replace('.pt', '_state.pt')
    # Skip if state_dict file already exists
    if os.path.exists(state_dict_path):
        print(f"[SKIP] {state_dict_path} (already exists)")
        continue
    # Save state_dict
    torch.save(model.state_dict(), state_dict_path)
    print(f"[SAVED] {state_dict_path}")

print("Conversion completed.")
