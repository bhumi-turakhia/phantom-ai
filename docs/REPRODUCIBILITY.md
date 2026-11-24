# Phantom AI 3.0 — Comprehensive Reproducibility Guide

This guide provides end-to-end instructions for reproducing the Phantom AI 3.0 runtime environment, verifying frozen neural network checkpoints, executing automated validation suites, and launching the full application stack.

---

## 1. System Requirements & Hardware Specifications

### 1.1 Hardware Specifications
- **Reference Platform:** Apple Silicon (M-series, e.g., M1/M2/M3/M4) or x86_64 Linux with NVIDIA GPU (CUDA 11.8+ / 12.x).
- **RAM:** Minimum 16 GB unified memory or system RAM (32 GB recommended for full dataset training).
- **Storage:** 
  - Application & Checkpoints: ~2 GB.
  - Full CelebA Dataset (Optional, for re-training): ~15 GB.
- **Compute Accelerators Supported:**
  - Apple Silicon `mps` (Metal Performance Shaders) — primary tested target.
  - NVIDIA `cuda` — automatically detected via PyTorch.
  - CPU fallback — verified operational across all inference pipelines.

### 1.2 Software Dependencies
- **Operating System:** macOS Sonoma / Sequoia or Ubuntu 22.04+ LTS.
- **Python:** Version 3.10 to 3.13 (Python 3.13.7 verified).
- **Node.js:** Version 20 LTS or higher (Node.js v25.9.0 verified).
- **Package Managers:** `npm` (v10+ / v11+) and `pip` (v24+).
- **Database:** MongoDB Community Edition 7.0+ or 8.x (running on port 27017).

---

## 2. Checkpoint Verification (SHA-256 Hashes)

Before launching any service, verify that the frozen model checkpoints match their canonical cryptographic digests.

```bash
cd phantom-ai-ml/checkpoints
shasum -a 256 *.pt
```

### Expected Checkpoint SHA-256 Digest Table
```
+------------------------------------+------------------------------------------------------------------+
| Checkpoint File                    | Expected SHA-256 Digest                                          |
+------------------------------------+------------------------------------------------------------------+
| cgan_generator_phase54.pt          | a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964 |
| forensic_evaluator_split_b.pt      | f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834 |
| nlp_char_bilstm_phase3c2.pt        | 2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3 |
| cgan_critic_phase54.pt             | b0e26457bdbbe07a2a2b037ba1b1c0223d1d4289da6104a060648aa47dfd63cb |
| latent_directions_phase57.pt       | e6c78a444d664ec8f49e73eba6eed32b4a376f0456f50376956ed6ef12ff766c |
+------------------------------------+------------------------------------------------------------------+
```

---

## 3. Environment Setup & Configuration

### 3.1 Database Setup (MongoDB)
Ensure MongoDB is running locally on port 27017:
```bash
# macOS (Homebrew)
brew services start mongodb/brew/mongodb-community

# Linux (systemd)
sudo systemctl start mongod
```

### 3.2 Machine Learning Service (`phantom-ai-ml`)
```bash
cd phantom-ai-ml

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify Python packages:
- `torch` (>= 2.15.0)
- `fastapi` (>= 0.115.0)
- `uvicorn` (>= 0.30.0)
- `pydantic` (>= 2.8.0)
- `numpy` (>= 2.0.0)
- `pillow` (>= 10.0.0)

### 3.3 Backend API Gateway (`phantom-ai-api`)
```bash
cd phantom-ai-api

# Install Node modules
npm install

# Build TypeScript
npm run build
```

Configure `phantom-ai-api/.env`:
```env
PORT=5001
NODE_ENV=development
MONGODB_URI=mongodb://127.0.0.1:27017/phantom_ai_dev
JWT_SECRET=super_secret_development_jwt_key_at_least_32_bytes_long
ML_SERVICE_URL=http://127.0.0.1:8000
RP_ID=localhost
RP_NAME=Phantom AI 3.0
CLIENT_ORIGIN=http://127.0.0.1:5173
```

### 3.4 Presentation Frontend (`phantom-ai-client`)
```bash
cd phantom-ai-client

# Install Node modules
npm install
```

Configure `phantom-ai-client/.env`:
```env
VITE_API_BASE_URL=http://127.0.0.1:5001/api
```

---

## 4. Running the Full Application Stack

To run the complete system, start the three services in separate terminal windows:

### Terminal 1: ML Inference Service
```bash
cd phantom-ai-ml
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*Health verification:* Visit `http://127.0.0.1:8000/api/health` — must return `status: ok` and confirm checkpoint digests.

### Terminal 2: Application API Gateway
```bash
cd phantom-ai-api
node dist/server.js
```
*Port:* Listens on `http://127.0.0.1:5001`.

### Terminal 3: React Presentation Client
```bash
cd phantom-ai-client
npm run dev -- --host 127.0.0.1 --port 5173
```
*Access:* Open `http://127.0.0.1:5173` in a modern WebAuthn-compliant browser (Chrome, Edge, Safari, Firefox).

---

## 5. Automated Test Suite Execution

The Phantom AI 3.0 verification suite separates unit, integration, browser acceptance, and negative security testing into distinct categories:

### 5.1 Application API & Gateway Tests (Jest)
```bash
cd phantom-ai-api
npm test
```
*Scope:* **84 Jest tests** covering JWT issuance, WebAuthn challenge/response verification, RBAC route protections, investigator approval enforcement, audit log persistence, and forensic report export packaging.

### 5.2 ML Microservice Tests (Python / PyTorch)
```bash
cd phantom-ai-ml
source .venv/bin/activate
pytest tests/ -v
```
*Scope:* **116 Python tests** validating GCIT mapping rules, explicit negative conditioning, tensor dimensions, mode-seeking diversity loss, evaluator multi-label outputs, and latent traversal bounds.

### 5.3 Live Browser Acceptance & Security Scenarios
- **8 Browser Workflow Acceptance Steps:** Full automated end-to-end user journey across the live stack (Passkey enrollment, login, case creation, transcript NLP analysis, investigator approval gate, composite generation, semantic editing, and report export).
- **18 Negative Security Scenarios:** Explicit adversarial and edge-case security checks (unauthenticated generation bypass, non-investigator approval attempts, invalid session token injection, malformed GCIT vectors, rate limiting, and unauthorized report tampering).

*(Note: These four test categories are independently structured and evaluated; their counts reflect distinct verification layers rather than overlapping totals).*

---

## 6. Reproducing Research Experiments & Benchmarks

All evaluation scripts operate directly on the frozen checkpoints without retraining.

### 6.1 Reproduce Independent NLP Benchmark & Calibration (Phase 3.7)
```bash
cd phantom-ai-ml
source .venv/bin/activate
python evaluate_phase37_independent.py
```
*Output:* Recomputes sequence precision, recall, strict span F1 ($0.5926$), and calibration error (ECE $0.4865$).

### 6.2 Reproduce Split C Held-Out Generative Benchmark (Phase 5.4)
```bash
cd phantom-ai-ml
source .venv/bin/activate
python eval_generator_phase54.py --split-c-dir /path/to/celeba/split_c
```
*Output:* Computes Custom Fréchet Distance ($203.85$), Forensic Attribute Verification F1 ($0.3562$), and MS-SSIM diversity ($0.3425$).

### 6.3 Reproduce Latent Traversal & Selectivity Audit (Phase 5.7)
```bash
cd phantom-ai-ml
source .venv/bin/activate
python validate_latent_traversal_phase57.py
```
*Output:* Validates the 5 accepted directions against the 15 rejected directions, regenerating `latent_traversal_validation_results.json`.

