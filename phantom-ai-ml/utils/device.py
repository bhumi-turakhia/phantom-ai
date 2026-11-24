import os
import sys
import torch
from typing import Dict, Any


def resolve_device() -> torch.device:
    """
    Automatically selects computational hardware according to priority:
    1. Apple Silicon MPS (Metal Performance Shaders)
    2. NVIDIA CUDA
    3. CPU fallback
    """
    # 1. Check Apple Silicon MPS
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")

    # 2. Check NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 3. CPU Fallback
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Gathers detailed hardware acceleration metadata.
    """
    device = resolve_device()
    device_type = device.type

    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())

    if device_type == "mps":
        device_name = "Apple Silicon MPS (Metal Performance Shaders)"
    elif device_type == "cuda":
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else "Generic CUDA"
        device_name = f"NVIDIA CUDA ({cuda_name})"
    else:
        device_name = "CPU (Central Processing Unit)"

    return {
        "device_type": device_type,
        "device_name": device_name,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
    }


def print_device_banner() -> None:
    """
    Logs hardware detection status to stdout.
    """
    info = get_device_info()
    sys.stdout.write("================================================================\n")
    sys.stdout.write(f"[HARDWARE ACCELERATION DETECTED]: {info['device_name']}\n")
    sys.stdout.write(f"PyTorch Version: {info['torch_version']} | Device Type: {info['device_type']}\n")
    sys.stdout.write(f"MPS Available: {info['mps_available']} | CUDA Available: {info['cuda_available']}\n")
    sys.stdout.write("================================================================\n")
    sys.stdout.flush()

