import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINTS_DIR = os.getenv("PHANTOM_ML_CHECKPOINTS_DIR", str(BASE_DIR / "checkpoints"))
DATASETS_DIR = os.getenv("PHANTOM_ML_DATASETS_DIR", str(BASE_DIR / "datasets"))

HOST = os.getenv("PHANTOM_ML_HOST", "127.0.0.1")
PORT = int(os.getenv("PHANTOM_ML_PORT", "8000"))

SERVICE_NAME = "Phantom AI Custom ML Service"
VERSION = "3.0.0-phase2-foundation"

