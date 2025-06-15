import aegnn.utils.bounding_box
import aegnn.utils.io
from aegnn.utils.multiprocessing import TaskManager

import aegnn.utils.callbacks
import aegnn.utils.loggers

import torch


def default_device() -> torch.device:
    """Return the best available torch device.

    The function prefers CUDA if available, then falls back to the MPS backend
    on Apple silicon and finally to the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
