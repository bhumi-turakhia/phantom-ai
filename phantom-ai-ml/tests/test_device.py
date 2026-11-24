import torch
from utils.device import resolve_device, get_device_info


def test_device_resolution_order():
    device = resolve_device()
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cuda", "cpu"]


def test_device_info_structure():
    info = get_device_info()
    assert "device_type" in info
    assert "device_name" in info
    assert "mps_built" in info
    assert "mps_available" in info
    assert "cuda_available" in info
    assert "torch_version" in info


def test_mps_priority_on_apple_silicon():
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    device = resolve_device()
    if mps_available:
        assert device.type == "mps"

