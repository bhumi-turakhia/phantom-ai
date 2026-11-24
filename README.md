# Phantom AI 3.0: Neural-Forensic Facial Composite System

> **Research Prototype & Technical Specification**  
> *A transparent, explainable, and ethically constrained deep-learning framework for witness-guided forensic facial composite generation.*

---

## 1. System Overview & Core Capabilities

**Phantom AI 3.0** is an open-research, locally executable forensic facial composite generation system designed to replace rigid legacy feature-pasting tools (e.g., Identi-KIT, FACES 4.0) with a continuous, neural-cognitive workflow.

```
[ Witness Narrative ] 
          │
          ▼
┌─────────────────────────────────┐
│ Character-Aware BiLSTM-CRF      │  NLP Extraction & Spans
│ - Word + Char CNN Embeddings    │  Evidence States: PRESENT / ABSENT / UNKNOWN
│ - Epistemic Uncertainty & Regex │  Explicit Negation & Boundary Windows
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Mandatory Investigator Gate     │  Human-in-the-Loop Arbitration
│ - Verify & correct mappings     │  Cryptographic Audit Logging
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ GCIT (Condition Ingestion)      │  Continuous Translation
│ - c \in [-1, 1]^20, m \in {0,1} │  Explicit Negative Conditioning
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Conditional WGAN-GP (64x64)     │  Adversarial Synthesis
│ - Projection Conditioning Critic│  Spectral Normalization & Mode-Seeking Diversity
│ - Disjoint Identity Training    │  Evaluated on Held-Out Split C
└─────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ Dual Semantic Editing Studio    │  5 Approved Latent Directions (SVM)
│ - Continuous GCIT Traversal     │  Audit Trail & Version Lineage Tree
└─────────────────────────────────┘
          │
          ▼
[ Multi-Modal Forensic Package Export (SHA-256 Integrity Checksums) ]
```

### Key Technical Innovations
1. **Character-Aware BiLSTM-CRF:** Robust sequence labeling parsing witness descriptions into structured facial attribute entities, character spans, and a 4-state epistemic taxonomy (`PRESENT`, `ABSENT`, `UNKNOWN`, `CONTRADICTORY`).
2. **Generative Condition Ingestion Translator (GCIT):** A deterministic translator mapping extracted evidence states into continuous conditioning vectors $c \in [-1.0, 1.0]^{20}$ and binary masks $m \in \{0, 1\}^{20}$. Features explicit negative conditioning ($c_i = -1.0, m_i = 1$) to actively penalize negated attributes during generation.
3. **Conditional WGAN-GP with Projection Conditioning:** A 4.1M parameter generator synthesizing native $64 \times 64$ RGB composites via spectral normalization, Wasserstein gradient penalty ($\lambda_{GP} = 10.0$), and mode-seeking diversity regularization ($\lambda_{ms} = 1.0$).
4. **Dual Semantic Refinement:** 5 disentangled latent space normal directions ($> 1.15\times$ selectivity) paired with continuous GCIT slider modulation across 9 structural geometry dimensions.
5. **Independent Forensic Evaluator:** A multi-label ConvNet (`ForensicFeatureCNN`) trained on an identity-disjoint partition (Split B1/B2) to evaluate composite fidelity without generative reward gaming.
6. **Cryptographic Provenance & Security:** WebAuthn (FIDO2) passwordless authentication, role-based access control (RBAC), immutable audit logging, and SHA-256 tamper-evident checksums.
7. **Evidence Traceability & Explainable AI (Phase 7.0):** 15-point attribute-to-generation provenance ledger linking every 20-D GCIT conditioning dimension ($c_i$) and mask ($m_i$) to exact witness character offsets, surrounding sentence context, NLP extraction states, and non-destructive investigator override logs.
8. **Multi-Variation Comparison & Attribute Difference Analysis (Phase 7.1):** Side-by-side comparison workflow across intra-batch variations (differing strictly by stochastic latent seed $z \sim \mathcal{N}(0, I_{100})$), semantic edited vs. baseline variations, and cross-version composite lineages, with 20-D GCIT difference highlighting and non-destructive preferred variation selection.
9. **Investigation Timeline & Investigator Collaboration (Phase 7.7):** Chronological case activity timeline synthesized from persisted database records and `SecurityAudit` logs (with explicit provenance classification distinguishing audited events from historical derived state), 7-stage investigative workflow progress tracking (`Case Creation` $\to$ `Report Generation`), RBAC-governed case participant assignments (`CASE_PARTICIPANT_ADDED` / `CASE_PARTICIPANT_REMOVED`), and isolated investigator collaboration notes (`CaseComment`) with non-destructive edit history and soft-delete audit retention.

---

## 2. Frozen Checkpoints & Integrity Digest

All models in Phantom AI 3.0 are frozen and cryptographically verified. No third-party pretrained weights, proprietary APIs, or cloud LLMs are used.

| Model / Artifact | Checkpoint File | Architecture | Parameters | Expected SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| **Generator** | `phantom-ai-ml/checkpoints/cgan_generator_phase54.pt` | `ConditionalWGANGPGenerator` | $4,102,336$ trainable ($4,104,260$ total)* | `a1714389d0d1eddad10fae74a736ca41de5e230eada095e6868e5daeeb2a1964` |
| **Evaluator** | `phantom-ai-ml/checkpoints/forensic_evaluator_split_b.pt` | `ForensicFeatureCNN` | $4,824,020$ trainable ($4,828,381$ total)* | `f52ae5a8df4ae574432f353fd77fc605f6dcf758d67d492b447c139a79e78834` |
| **NLP Labeler** | `phantom-ai-ml/checkpoints/nlp_char_bilstm_phase3c2.pt` | `CharAwareBiLSTMAttributeLabeler` | $2,599,864$ trainable | `2d567c79d759e76940968fadb02d412b530d3405e377a2d6454c64e9f2e7b4a3` |
| **Critic** | `phantom-ai-ml/checkpoints/cgan_critic_phase54.pt` | `ProjectionCritic` | $4,316,289$ trainable | `b0e26457bdbbe07a2a2b037ba1b1c0223d1d4289da6104a060648aa47dfd63cb` |
| **Latent Dirs** | `phantom-ai-ml/checkpoints/latent_directions_phase57.pt` | Normal vectors in $\mathcal{Z} \subset \mathbb{R}^{100}$ | 5 vectors | `e6c78a444d664ec8f49e73eba6eed32b4a376f0456f50376956ed6ef12ff766c` |

*\* Note on parameter counts: In PyTorch, trainable gradient parameters represent weights and biases in `model.parameters()`. Checkpoint `state_dict` tensors additionally include persistent non-gradient BatchNorm running statistics (`running_mean`, `running_var`, `num_batches_tracked`: $+4,361$ elements for Evaluator, $+1,924$ elements for Generator), fully reconciling with the $4,828,381$ and $4,104,260$ counts cited in the Phase 5.8.1 audit.*

Verify on your machine:
```bash
shasum -a 256 phantom-ai-ml/checkpoints/*.pt
```

---

## 3. Documentation Index

Detailed research specifications, mathematical derivations, empirical tables, and operating guides are organized in the `docs/` directory:

| Document | Description |
| :--- | :--- |
| **[`docs/RESEARCH_METHODOLOGY.md`](docs/RESEARCH_METHODOLOGY.md)** | Full mathematical formulation: BiLSTM-CRF, GCIT mapping rules, Conditional WGAN-GP loss, projection critic, mode-seeking diversity, and SVM latent boundary derivations. |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | End-to-end multi-tier architecture, component responsibilities, Mermaid data-flow diagrams, WebAuthn integration, and security boundaries. |
| **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** | Step-by-step reproduction instructions, software prerequisites, environment variables, test execution commands, and service startup scripts. |
| **[`docs/EXPERIMENT_RESULTS.md`](docs/EXPERIMENT_RESULTS.md)** | Empirical results across all phases: CelebA identity splits, Evaluator F1/AUROC, Generator C-FD/MS-SSIM on Split C, latent selectivity ratios, and NLP calibration curves. |
| **[`docs/ETHICS_AND_LIMITATIONS.md`](docs/ETHICS_AND_LIMITATIONS.md)** | Comprehensive ethics statement: CelebA demographic and lighting biases, $64 \times 64$ resolution limits, verbal overshadowing risks, and prohibited use cases. |
| **[`docs/reproducibility_manifest.json`](docs/reproducibility_manifest.json)** | Machine-readable JSON manifest containing exact dependency versions, dataset partitions, hyperparameters, and evaluation metrics. |

---

## 4. Quick Start: Running the System

### Prerequisites
- macOS (Apple Silicon MPS) or Linux (NVIDIA GPU / CPU)
- Python 3.10–3.13
- Node.js 20+ (Node 25 supported)
- MongoDB 7.0+ or 8.x running locally on port `27017`

### Step 1: Start the ML Inference Microservice
```bash
cd phantom-ai-ml
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Step 2: Start the Application API Gateway
```bash
cd phantom-ai-api
node dist/server.js
```

### Step 3: Start the Presentation Frontend
```bash
cd phantom-ai-client
npm run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173` in a modern WebAuthn-compliant browser to begin an investigative session.

### Verification Suites & Acceptance Testing
Phantom AI 3.0 maintains distinct, non-overlapping verification layers (validated in Phase 7.8):
- **131 Jest Gateway & API Tests across 11 Suites** (84 foundational Phase 1–6.1 tests + 19 Phase 7.0 Evidence Traceability tests + 11 Phase 7.1 Multi-Variation Comparison tests + 12 Phase 7.7 Investigation Timeline & Collaboration tests + 5 Phase 7.8 Cross-Phase Final Integration & RBAC Isolation tests): Complete coverage of WebAuthn ceremonies, RBAC policies, investigator approval enforcement, 15-point XAI provenance, 20-D GCIT difference analysis, preferred variation selection, chronological timeline synthesis, participant assignments, isolated case comments, 7-stage progress tracking, and audit logging.
- **116 Python ML & PyTorch Tests:** Comprehensive unit and regression testing of GCIT translation, negative conditioning, WGAN-GP shapes, evaluator outputs, and latent traversal bounds.
- **29 Live API, Security & Data-Integrity Integration Checks (`verify_phase78_e2e.mjs`):** Live end-to-end validation across Express (`127.0.0.1:5001`), MongoDB (`127.0.0.1:27017`), and FastAPI PyTorch (`127.0.0.1:8000`) covering all 13 investigator workflow steps, 8 security/authorization boundaries, and 8 data-integrity checks.
- **10 Live Headless Chrome CDP Browser Workflow Milestones (`scripts/e2e_phase78_browser_workflow.py`):** End-to-end browser verification of the complete 13-step investigator journey across React, Express, MongoDB, and FastAPI (passkey enrollment, MFA login, case creation, NLP transcript analysis, Phase 7.0 Evidence Explainability Modal, investigator correction & approval, WGAN-GP composite generation & semantic editing, Phase 7.1 Multi-Variation Comparison Modal & preferred selection, Phase 7.7 7-Stage Case Progress & Activity Timeline, participant assignment & case comments, and forensic PDF report export) with `0` console/page exceptions and `0` network 4xx/5xx errors.
- **18 Negative Security Scenarios:** Adversarial checks verifying route tampering resistance, unauthorized generation blocking, token manipulation rejection, and input sanitization.

*(Note on Phase 6.0: The Phase 6.0 audit was a rigorous code-level security, dependency, and deployment-readiness assessment; it establishes security posture and readiness for deployment rather than claiming live production hosting).*

---

## 5. Summary of Key Research Findings

1. **Identity-Disjoint Training (CelebA):**
   - Partitioning identities into Split A ($8,126$ IDs, $161,780$ images), Split B ($980$ IDs, $19,313$ images), and Split C ($1,071$ IDs, $21,506$ images) prevented identity memorization and enabled uncompromised validation.
2. **Generative Fidelity on Unseen Identities (Split C):**
   - Custom Fréchet Distance: `203.8529`.
   - Forensic Attribute Verification (FAV) Macro F1: `0.3562` (Macro Accuracy: `73.60%`).
   - Mean pairwise MS-SSIM: `0.3425` (Seed Diversity: `0.6575`).
3. **Latent Space Entanglement:**
   - Out of 20 candidate attribute directions in $\mathcal{Z}$, only 5 achieved selectivity $\ge 1.15\times$: `hair_blond` ($1.34\times$), `hair_gray` ($1.41\times$), `hair_bangs` ($2.16\times$), `facial_goatee` ($1.23\times$), and `glasses` ($1.36\times$). The remaining 15 were rejected due to severe entanglement and reassigned to continuous GCIT modulation.
4. **NLP Extraction Miscalibration & Domain Drop:**
   - When evaluating the BiLSTM-CRF on natural conversational syntax outside its synthetic training templates, strict span F1 dropped by $28.6\%$ (from $0.8786$ to $0.5926$).
   - Expected Calibration Error (ECE) reached `0.4865`, demonstrating severe overconfidence. Consequently, heuristic NLP extraction confidence is suppressed and recorded as `null`, mandating human investigator verification.

---

## 6. Forensic, Ethical & Legal Disclaimers

> ### **CRITICAL FORENSIC USE RESTRICTIONS**
> 
> 1. **NON-BIOMETRIC APPROXIMATION:** Composites synthesized by Phantom AI 3.0 are **cognitive recall aids and investigative approximations ONLY**. They do not constitute biometric photographs or facial recognition probes.
> 2. **NO AUTOMATED SUSPECT MATCHING:** Composites produced by this software must **NEVER** be supplied as inputs to Automated Facial Recognition (AFR) search engines or automated biometric booking systems.
> 3. **NO STANDALONE JUDICIAL EVIDENCE:** Synthesized composites have **NO independent evidentiary admissibility** in legal proceedings as proof of guilt or identity.
> 4. **MANDATORY HUMAN-IN-THE-LOOP:** Autonomous generation without investigator approval is prohibited. All semantic extractions must be verified with the witness prior to synthesis.
> 5. **CHECKSUM PROVENANCE:** SHA-256 digests verify file-level data integrity and tamper-evidence; they do not represent asymmetric digital signatures or an independent institutional chain of custody.

