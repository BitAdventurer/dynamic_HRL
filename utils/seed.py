import os
import random
import torch
import numpy as np

def seed_everything(seed=42, skip_random=False):
    """
    Set random seeds for Python, NumPy, Torch, and CUDNN for reproducibility.
    If skip_random=True, do NOT set random.seed() (for non-deterministic exploration).
    Call this ONCE at the start of your script. Do NOT call repeatedly.
    """
    if not skip_random:
        random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False