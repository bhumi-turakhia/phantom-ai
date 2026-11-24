import random
import os
import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """
    Sets reproducible random seeds across Python, NumPy, and PyTorch (CPU, CUDA, MPS).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS deterministic settings
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass

