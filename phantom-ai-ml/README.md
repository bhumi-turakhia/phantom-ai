# Phantom AI 3.0 — Custom Python ML Service Foundation (Phase 2)

A zero-paid-API, zero-external-inference local forensic machine learning microservice built with **Python 3.13**, **PyTorch**, and **FastAPI**.

## Core Constraints & Principles
1. **Zero External Inference APIs**: No Gemini API, Hugging Face Inference API, OpenAI, or paid external services.
2. **Zero Pretrained Weights**: All architectures are defined from scratch using PyTorch with random parameter initialization. Pretrained embeddings or weights are strictly forbidden.
3. **Data Confidentiality**: Sensitive witness testimony and investigative evidence remain 100% on-premises and are never transmitted externally.
4. **Honest Reporting**: No simulated or fabricated predictions or evaluation metrics. If a model checkpoint does not exist, the API returns an explicit `model_not_trained` status.
5. **Multi-Hardware Acceleration**: Automatically prioritizes Apple Silicon MPS (Metal Performance Shaders), NVIDIA CUDA, and CPU.

---

## Directory Structure
```
phantom-ai-ml/
├── app/
│   ├── config.py             # Service configuration (host, port, dirs)
│   └── main.py               # FastAPI application and route endpoints
├── checkpoints/
│   └── README.md             # Checkpoint storage documentation
├── datasets/
│   └── README.md             # Dataset specifications & privacy policy
├── evaluation/
│   ├── evaluate_generation.py# DCGAN generator evaluation
│   ├── evaluate_metrics.py   # Strict precision, recall, F1 computation
│   └── evaluate_nlp.py       # BiLSTM sequence labeler evaluation
├── models/
│   ├── cnn_model.py          # Custom 4-block CNN feature extractor
│   ├── dcgan_model.py        # Conditional DCGAN Generator & Discriminator
│   ├── nlp_model.py          # Custom BiLSTM with CRF emission layer
│   └── uncertainty_model.py  # Evidential deep learning uncertainty classifier
├── schemas/
│   ├── attributes.py         # Forensic evidence states & attribute categories
│   └── requests.py           # Pydantic request & response schemas
├── services/
│   ├── evaluation_service.py # Honest evaluation handler
│   ├── evidence_service.py   # Forensic attribute validation engine
│   ├── nlp_service.py        # Sequence labeling & description analysis
│   └── uncertainty_service.py# Linguistic hedging & contradiction detection
├── tests/
│   ├── test_api.py           # FastAPI endpoints tests
│   ├── test_device.py        # Hardware detection tests
│   ├── test_models.py        # Model initialization & forward pass tests
│   ├── test_schemas.py       # Pydantic validation tests
│   └── test_services.py      # Linguistic analysis & contradiction tests
├── training/
│   ├── train_cnn.py          # CNN feature extractor training loop
│   ├── train_dcgan.py        # Conditional DCGAN training loop
│   ├── train_nlp.py          # BiLSTM sequence labeler training loop
│   └── train_uncertainty.py  # Evidential Dirichlet model training loop
├── utils/
│   ├── device.py             # Automatic MPS / CUDA / CPU resolution
│   ├── preprocessing.py      # Vocabulary, tokenizer, BIO tags
│   └── seed.py               # Global reproducible seed utility
├── requirements.txt
└── README.md
```

---

## Hardware Acceleration Detection Priority
Implemented in `utils/device.py`:
1. **Apple Silicon MPS**: `torch.backends.mps.is_available() and torch.backends.mps.is_built()`
2. **NVIDIA CUDA**: `torch.cuda.is_available()`
3. **CPU Fallback**: `torch.device("cpu")`

The active device is reported in `GET /health` and `GET /model-status`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns service health, version, and active device details |
| `GET` | `/model-status` | Reports architectural metadata and checkpoint status (trained vs untrained) |
| `POST` | `/analyze-description` | Extracts forensic facial attributes while enforcing uncertainty states |
| `POST` | `/validate-attributes` | Validates attributes and flags forensic contradictions |
| `POST` | `/evaluate` | Evaluates models; returns `model_not_trained` if checkpoints are missing |

---

## Setup & Running the ML Service

### 1. Create and Activate Virtual Environment
```bash
cd phantom-ai-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the FastAPI Service
```bash
# Run on localhost:8000
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Run Automated Tests
```bash
source .venv/bin/activate
pytest tests/ -v
```

