# Phantom AI 3.0 — Research Methodology & Technical Specification

## 1. Executive Summary & Research Motivation

Forensic facial composite generation has historically relied on manual sketch artistry or rigid feature-pasting software (e.g., Identi-KIT, FACES 4.0, E-FIT). These legacy approaches suffer from critical cognitive and operational bottlenecks:
1. **Verbal Overshadowing:** The cognitive phenomenon where attempting to describe an unfamiliar face in granular detail alters or degrades the witness's original visual memory trace.
2. **Feature Isolation Fallacy:** Human holistic face recognition processes faces gestalt-first rather than as separable assemblies of independent parts. Discrete component placement often produces uncanny, structurally disjoint composites.
3. **Uncertainty Suppression:** Legacy tools force witnesses to select discrete options, discarding nuanced uncertainty markers ("maybe", "not sure", "possibly") or explicit negative observations ("no facial hair", "definitely not blond").
4. **Lack of Provenance & Forensic Integrity:** Ad-hoc manual edits lack cryptographic auditability, creating vulnerabilities to investigator bias and evidentiary tampering.

**Phantom AI 3.0** introduces a fully integrated, transparent, neural-forensic cognitive workflow:
- **Neural Semantic Extraction:** A Character-Aware Bidirectional LSTM with a Linear-Chain Conditional Random Field (BiLSTM-CRF) that parses unstructured witness narratives into granular facial attributes, preserving character spans, negation polarity, and epistemic uncertainty.
- **Generative Condition Ingestion Translator (GCIT):** A formalized mapping that converts witness evidence states into continuous multi-label condition vectors and explicit conditioning masks.
- **Conditional WGAN-GP Synthesis:** A $64 \times 64$ conditional generative adversarial network featuring spectral normalization, projection conditioning, and mode-seeking diversity, trained on strictly identity-disjoint CelebA partitions.
- **Dual Semantic Editing:** A rigorous latent space linear traversal system ($z$-space hyperplanes) paired with continuous GCIT slider modulation, validated against an independent forensic evaluator to eliminate visual entanglement.
- **Cryptographic Provenance:** End-to-end SHA-256 audit trails, version lineage graphs, and WebAuthn FIDO2 authentication enforcing strict human-in-the-loop investigator gatekeeping.

---

## 2. Natural Language Processing (NLP) Extraction Pipeline

```
+-----------------------------------------------------------------------------+
|                         Witness Interview Transcript                        |
|   "He had messy dark brown hair, definitely no beard, and maybe blue eyes"   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                        Tokenization & Representation                        |
|                                                                             |
|   Token: "messy"           Token: "beard"             Token: "blue"         |
|   +--------------------+   +--------------------+     +-------------------+ |
|   | Word Emb (100-D)   |   | Word Emb (100-D)   | ... | Word Emb (100-D)  | |
|   | Char CNN (50-D)    |   | Char CNN (50-D)    |     | Char CNN (50-D)   | |
|   +--------------------+   +--------------------+     +-------------------+ |
|             |                        |                          |           |
|             v                        v                          v           |
|      Concat (150-D)           Concat (150-D)             Concat (150-D)     |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                 Bidirectional LSTM Encoder (2 Layers, 256-D)                |
|             Forward LSTM (128-D) --->   <--- Backward LSTM (128-D)          |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                        Linear Projection (21 BIO Tags)                      |
|                               Emission Scores P                             |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     Linear-Chain CRF Layer (Viterbi Search)                 |
|               Global Transition Matrix A \in \mathbb{R}^{21 \times 21}      |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|            Post-Processing: Uncertainty, Negation & Boundary Refinement     |
|   hair_type   -> PRESENT (messy)                                            |
|   hair_color  -> PRESENT (dark brown)                                       |
|   facial_hair -> ABSENT (no beard)                                          |
|   eye_color   -> UNKNOWN (maybe blue eyes - epistemic hedge)                |
+-----------------------------------------------------------------------------+
```

### 2.1 Character-Aware BiLSTM-CRF Architecture

To handle orthographic variability, misspellings, compound descriptors, and open-vocabulary witness terminology, Phantom AI employs a hybrid word- and character-level sequence labeling model (`CharAwareBiLSTMAttributeLabeler`).

#### 2.1.1 Character-Level CNN Encoder
For each word token $w_t$ composed of characters $[c_1, c_2, \dots, c_L]$:
1. Each character $c_k$ is embedded into a dense vector:
   $$e_{c_k} = \mathbf{E}_{char}(c_k) \in \mathbb{R}^{d_{char}}, \quad d_{char} = 30$$
2. A 1D convolutional filter of kernel size $K = 3$ with $N_f = 50$ filters scans the character sequence:
   $$h_{char, k} = \text{ReLU}(\mathbf{W}_c * e_{c_{k:k+K-1}} + b_c)$$
3. Max-over-time pooling extracts the most salient morphological feature map:
   $$v_{char}(w_t) = \max_{1 \le k \le L - K + 1} h_{char, k} \in \mathbb{R}^{50}$$

#### 2.1.2 Word-Level Embedding & Concatenation
Word tokens are embedded using a domain-tuned vocabulary lookup:
$$v_{word}(w_t) = \mathbf{E}_{word}(w_t) \in \mathbb{R}^{d_{word}}, \quad d_{word} = 100$$
The combined token representation $x_t$ merges syntactic semantic context with morphological robustness:
$$x_t = [v_{word}(w_t); v_{char}(w_t)] \in \mathbb{R}^{150}$$

#### 2.1.3 Bidirectional Recurrent Encoding
The sequence of concatenated vectors $[x_1, x_2, \dots, x_T]$ is processed by a 2-layer Bidirectional LSTM:
$$\overrightarrow{h}_t = \text{LSTM}_{fwd}(x_t, \overrightarrow{h}_{t-1}) \in \mathbb{R}^{128}$$
$$\overleftarrow{h}_t = \text{LSTM}_{bwd}(x_t, \overleftarrow{h}_{t+1}) \in \mathbb{R}^{128}$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t] \in \mathbb{R}^{256}$$

#### 2.1.4 Conditional Random Field (CRF) Layer
The concatenated hidden state $h_t$ is linearly projected to emission scores $P \in \mathbb{R}^{T \times K}$ over $K = 21$ BIO tagging states:
$$P_{t, j} = \mathbf{W}_e h_t + b_e$$

The tag set includes: `O`, `B-age`, `I-age`, `B-gender`, `I-gender`, `B-skin_tone`, `I-skin_tone`, `B-face_shape`, `I-face_shape`, `B-hair_color`, `I-hair_color`, `B-hair_type`, `I-hair_type`, `B-eye_color`, `I-eye_color`, `B-eyebrows`, `I-eyebrows`, `B-nose`, `I-nose`, `B-mouth_lips`, `I-mouth_lips`, `B-facial_hair`, `I-facial_hair`.

For a sequence of tags $y = (y_1, y_2, \dots, y_T)$, the total sequence score is:
$$S(X, y) = \sum_{t=1}^T P_{t, y_t} + \sum_{t=1}^{T-1} A_{y_t, y_{t+1}}$$
where $A \in \mathbb{R}^{K \times K}$ is the learned transition matrix.

During training, the negative log-likelihood loss is minimized using the forward-backward algorithm:
$$\mathcal{L}_{CRF} = -\log P(y|X) = -\left( S(X, y) - \log \sum_{\tilde{y} \in \mathcal{Y}} \exp(S(X, \tilde{y})) \right)$$

During inference, optimal sequence decoding is solved globally via the Viterbi dynamic programming algorithm:
$$y^* = \arg\max_{y \in \mathcal{Y}} S(X, y)$$

---

### 2.2 Evidence-State Taxonomy and Post-Processing

The raw sequence of BIO tags is post-processed into structured semantic slots governed by a four-state epistemic taxonomy:

```
                      +-------------------+
                      | Extracted Entity  |
                      |    (BIO Span)     |
                      +-------------------+
                                |
         +----------------------+----------------------+
         |                      |                      |
         v                      v                      v
+------------------+   +------------------+   +------------------+
| Negation Window  |   | Uncertainty Cue  |   | Conflicting Span |
| ("no", "not")    |   | ("maybe", "ish") |   | In Same Category |
+------------------+   +------------------+   +------------------+
         |                      |                      |
         v                      v                      v
   EvidenceState:         EvidenceState:         EvidenceState:
       ABSENT                UNKNOWN             CONTRADICTORY
         |                      |                      |
         +----------------------+----------------------+
                                |
                   (No Modifiers / Explicit Affirmation)
                                v
                          EvidenceState:
                             PRESENT
```

1. **`PRESENT`:** The witness affirms the visual characteristic (e.g., "thick black eyebrows", "square jaw").
2. **`ABSENT`:** The witness explicitly negates the characteristic (e.g., "clean-shaven", "no glasses", "definitely not blond"). The negation engine inspects a 3-token backward/forward window for polarity inverters (`no`, `not`, `without`, `clean`, `bare`, `never`).
3. **`UNKNOWN`:** The witness expresses low epistemic certainty or hedging (e.g., "maybe in his 30s", "possibly brown hair", "could not tell the eye color"). Uncertainty lexicons intercept these tokens, preventing low-confidence guesses from contaminating generation.
4. **`CONTRADICTORY`:** Mutually exclusive attributes occur within the same narrative (e.g., "he had long hair... wait, actually it was buzzed short"). Such conflicts are flagged for human investigator arbitration.

---

### 2.3 Generative Condition Ingestion Translator (GCIT)

The GCIT acts as the deterministic bridge between discrete NLP evidence states and the continuous mathematical conditioning required by the generative neural network.

The conditioning space is parameterized by a tuple $(c, m)$:
- **Condition Scalar Vector:** $c \in [-1.0, 1.0]^{20}$
- **Conditioning Binary Mask:** $m \in \{0, 1\}^{20}$

The 20 canonical dimensions correspond to:
`[Male, Young, Pale_Skin, Oval_Face, Pointy_Nose, High_Cheekbones, Chubby, Double_Chin, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Wavy_Hair, Straight_Hair, Receding_Hairline, Bangs, Eyeglasses, Bushy_Eyebrows, Arched_Eyebrows, Goatee]`

#### GCIT Formal Mapping Rules:
1. **Positive Conditioning (`PRESENT`):**
   $$c_i = +1.0, \quad m_i = 1$$
2. **Negative Conditioning (`ABSENT`):**
   $$c_i = -1.0, \quad m_i = 1$$
   *Crucial distinction:* Negative conditioning ($c_i = -1.0, m_i = 1$) actively penalizes the attribute during generation, unlike unconditioned generation ($m_i = 0$) which leaves the attribute unconstrained.
3. **Unconditioned / Hedged (`UNKNOWN` / Unmentioned):**
   $$c_i = 0.0, \quad m_i = 0$$
4. **Contradictions (`CONTRADICTORY`):**
   $$c_i = 0.0, \quad m_i = 0 \quad (\text{with UI investigator alert})$$

The masked condition vector ingested by the neural network is formed by concatenation:
$$c_{masked} = [c \odot m, m] \in \mathbb{R}^{40}$$

---

## 3. Generative Adversarial Network Architecture (Conditional WGAN-GP)

```
                         GENERATOR ARCHITECTURE (4,102,336 Trainable / 4,104,260 Total Tensors)

 Latent Noise: z ~ N(0, I_100)
 Masked GCIT:  c_masked \in R^40
               |
               v
       [ Concatenate: 140-D ]
               |
               v
       Linear Projection + Reshape: 512 x 4 x 4
               |
               v
       Transposed Conv 4x4 -> 8x8 (512 -> 256 ch, stride 2, pad 1) + BatchNorm + ReLU
               |
               v
       Transposed Conv 8x8 -> 16x16 (256 -> 128 ch, stride 2, pad 1) + BatchNorm + ReLU
               |
               v
       Transposed Conv 16x16 -> 32x32 (128 -> 64 ch, stride 2, pad 1) + BatchNorm + ReLU
               |
               v
       Transposed Conv 32x32 -> 64x64 (64 -> 32 ch, stride 2, pad 1) + BatchNorm + ReLU
               |
               v
       Conv2d 64x64 (32 -> 3 ch, kernel 3, pad 1) + Tanh
               |
               v
       Generated Composite Image: x_fake \in [-1, 1]^{3 x 64 x 64}
```

```
                          CRITIC ARCHITECTURE (PROJECTION CRITIC)

 Image: x \in R^{3 x 64 x 64}
               |
               v
       Conv2d 64x64 -> 32x32 (3 -> 64 ch, stride 2) + SpectralNorm + LayerNorm + LeakyReLU(0.2)
               |
               v
       Conv2d 32x32 -> 16x16 (64 -> 128 ch, stride 2) + SpectralNorm + LayerNorm + LeakyReLU(0.2)
               |
               v
       Conv2d 16x16 -> 8x8 (128 -> 256 ch, stride 2) + SpectralNorm + LayerNorm + LeakyReLU(0.2)
               |
               v
       Conv2d 8x8 -> 4x4 (256 -> 512 ch, stride 2) + SpectralNorm + LayerNorm + LeakyReLU(0.2)
               |
               v
       Global Sum Pooling -> phi(x) \in R^512
               |
       +-------+-------------------------+
       |                                 |
       v                                 v
   Linear(512 -> 1)              Linear Projection of c_masked:
   Unconditional Score: psi(x)   v(c_masked) \in R^512
       |                                 |
       +----------------< Inner Product >+
                                 |
                                 v
               Critic Output: D(x, c_masked) = psi(x) + <phi(x), v(c_masked)>
```

### 3.1 Loss Formulation

To ensure stable convergence without gradient explosion or mode collapse, Phantom AI 3.0 uses the Wasserstein distance with Gradient Penalty (Gulrajani et al., 2017) and Projection Conditioning (Miyato & Koyama, 2018).

#### 3.1.1 Critic Loss
$$\mathcal{L}_D = \mathbb{E}_{z \sim p_z, c \sim p_c} [D(G(z, c), c)] - \mathbb{E}_{x \sim p_{data}, c \sim p_c} [D(x, c)] + \lambda_{GP} \mathcal{L}_{GP}$$
The gradient penalty enforces the 1-Lipschitz continuity constraint:
$$\mathcal{L}_{GP} = \mathbb{E}_{\hat{x} \sim p_{\hat{x}}} \left[ \left( \|\nabla_{\hat{x}} D(\hat{x}, c)\|_2 - 1 \right)^2 \right]$$
where $\hat{x} = \epsilon x_{real} + (1 - \epsilon) x_{fake}$ with $\epsilon \sim \mathcal{U}(0, 1)$, and $\lambda_{GP} = 10.0$.

#### 3.1.2 Generator Loss with Mode-Seeking Regularization
To combat mode collapse where differing random noise vectors $z_1, z_2$ produce identical composites under strong conditioning masks, a mode-seeking diversity regularization term (Mao et al., 2019) is integrated:
$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z, c \sim p_c} [D(G(z, c), c)] - \lambda_{ms} \frac{\|G(z_1, c) - G(z_2, c)\|_1}{\|z_1 - z_2\|_1}$$
where $\lambda_{ms} = 1.0$.

---

## 4. Independent Forensic Evaluator

To rigorously assess generated composite alignment without reward gaming or synthetic evaluation leakage, Phantom AI 3.0 maintains a dedicated evaluation network (`ForensicFeatureCNN`).

- **Architecture:** 4-stage convolutional feature backbone with batch normalization, max-pooling, and individual sigmoid classification heads for each of the 20 facial attributes:
  - *Trainable Parameters:* $4,824,020$ floating-point parameters in `model.parameters()`.
  - *Checkpoint State Dict Elements:* $4,828,381$ total scalar tensors (including $4,361$ persistent BatchNorm running statistics: `running_mean`, `running_var`, and `num_batches_tracked` across the 9 BatchNorm2d layers). This explicitly reconciles the $4,828,381$ count documented in Phase 5.8.1 with the $4,824,020$ trainable gradient count.
- **Split Isolation:** Trained strictly on **Split B1** (15,521 images, 791 identities). The evaluator never observes Split A identities (Generator training) or Split C identities (Benchmark).
- **Validation Metrics (Split B2):** Macro F1: `0.5141`, Micro F1: `0.5877`, Accuracy: `77.84%`, AUROC: `0.8533`.

```
+-------------------------------------------------------------------------+
|                  DISJOINT DATASET PARTITIONS (CelebA)                   |
+-------------------------------------------------------------------------+
|  Split A (79.85%, 8,126 IDs, 161,780 images)                            |
|    - A1 Train (130,063 images): Conditional WGAN-GP Generator & Critic  |
|    - A2 Val   (31,717 images):  Adversarial Hyperparameter Selection    |
+-------------------------------------------------------------------------+
|  Split B (9.53%, 980 IDs, 19,313 images)                                |
|    - B1 Train (15,521 images):  Independent Forensic Evaluator CNN      |
|    - B2 Val   (3,792 images):   Evaluator Validation & Calibration      |
+-------------------------------------------------------------------------+
|  Split C (10.62%, 1,071 IDs, 21,506 images)                             |
|    - Pure Held-Out Benchmark:   Zero model exposed during training      |
+-------------------------------------------------------------------------+
```

---

## 5. Latent Space Semantic Editing & Entanglement Analysis

Post-generation composite refinement requires localized modification without regenerating the identity. Phantom AI provides a dual editing methodology:

### 5.1 Latent Space Linear Vector Traversal ($z$-space)

Following the boundary-finding approach (Shen et al., 2020), 20,000 random latent codes $z \sim \mathcal{N}(0, I_{100})$ are sampled, rendered by the generator, and scored by the independent forensic evaluator. For each attribute $a$:
1. Binary labels $y_a \in \{-1, +1\}$ are assigned using top/bottom evaluator decision percentiles.
2. A linear Support Vector Machine (SVM) finds the optimal separating hyperplane in $\mathcal{Z}$:
   $$w_a^T z + b_a = 0$$
3. The unit normal vector is calculated: $n_a = \frac{w_a}{\|w_a\|_2} \in \mathbb{R}^{100}$.
4. Traversal is achieved via scalar manipulation: $z' = z + \alpha n_a, \quad \alpha \in [-3.0, 3.0]$.

#### Selectivity and Entanglement Metric:
To quantify whether modifying attribute $a$ causes unintended drift in other facial attributes $j \neq a$:
$$\text{Selectivity}(a) = \frac{|\Delta \hat{y}_a|}{\frac{1}{K-1} \sum_{j \neq a} |\Delta \hat{y}_j|}$$

A selectivity threshold of $\ge 1.15\times$ was required for acceptance. Out of 20 candidate directions:
- **Accepted (5 directions):**
  1. `hair_blond` ($1.34\times$)
  2. `hair_gray` ($1.41\times$)
  3. `hair_bangs` ($2.16\times$)
  4. `facial_goatee` ($1.23\times$)
  5. `glasses` ($1.36\times$)
- **Rejected (15 directions):**
  Rejected due to severe visual entanglement (e.g., `Young` altered perceived gender and skin tone; `Pale_Skin` collapsed facial geometry; `Male` drastically warped hair structure).

### 5.2 Continuous GCIT Slider Traversal

For continuous features (e.g., age, face shape, nose width, hairline), Phantom AI allows direct continuous interpolation of the condition vector:
$$c_i \in [-1.0, 1.0], \quad m_i = 1$$
This bypasses latent space entanglement by leveraging the generator's conditional projection manifold directly.

