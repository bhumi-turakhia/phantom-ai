# Phantom AI 3.0 — Ethics, Bias & Operational Limitations Statement

## 1. Ethical Foundations & Intended Operating Scope

Forensic facial composite generation operates at the intersection of cognitive psychology, criminal justice, and generative artificial intelligence. The stakes of this application are exceptionally high: misidentifications can lead to wrongful arrests, traumatic investigative scrutiny for innocent individuals, and miscarriages of justice.

**Phantom AI 3.0 is designed strictly as an investigative lead aid and cognitive recall catalyst.**

### 1.1 Prohibited Use Cases (Mandatory Red Lines)
1. **No Automated Biometric Identification:** Phantom AI composites must **never** be supplied as query probes to automated facial recognition (AFR) systems or automated database search engines. The low native resolution ($64 \times 64$) and generative variance produce feature representations that trigger unpredictable false-match clusters in biometric algorithms.
2. **No Judicial Proof of Guilt:** Generated composites have **zero standalone evidentiary value** in legal proceedings. They represent subjective witness memory approximations, not photographic or biometric records. They must never be presented to juries as proof of identity.
3. **No Autonomous Investigation:** The software must never operate without a trained human investigator mediating all inputs, approvals, edits, and exports. Fully automated text-to-composite pipelines are strictly prohibited.

---

## 2. Dataset Provenance, Demographic Bias & Representation

Phantom AI's generative network and independent evaluator were trained on the Large-scale CelebFaces Attributes (CelebA) dataset. While CelebA provides thousands of identities with 40 binary attribute annotations, it introduces severe systemic biases that constrain the system's operational envelope:

```
+---------------------------------------------------------------------------------------------------+
|                                  CELEBA DATASET SYSTEMIC BIASES                                   |
+---------------------+----------------------------+------------------------------------------------+
| Dimension           | Observed Representation    | Real-World Forensic Impact                     |
+---------------------+----------------------------+------------------------------------------------+
| Age Distribution    | 77.9% annotated as "Young" | Severe underrepresentation of elderly, middle- |
|                     | 22.1% Non-Young            | aged, or heavily weathered faces.              |
+---------------------+----------------------------+------------------------------------------------+
| Gender Balance      | 58.1% Female               | Skewed toward female celebrity aesthetics;     |
|                     | 41.9% Male                 | male composites may inherit softened contours. |
+---------------------+----------------------------+------------------------------------------------+
| Racial / Skin Tone  | > 85% Light / Fair Skin    | Pronounced underrepresentation of dark skin    |
| Diversity           | < 15% Diverse Melanin Tones| tones; risks lower generative fidelity and     |
|                     |                            | attribute fidelity across non-Caucasian faces. |
+---------------------+----------------------------+------------------------------------------------+
| Photographic Style  | Professional studio light, | Real perpetrators are observed under poor      |
|                     | heavy makeup, red-carpet   | lighting, surveillance CCTV, or darkness;      |
|                     | posing, smiling (48.2%)    | model tends to synthesize glamorous aesthetics.|
+---------------------+----------------------------+------------------------------------------------+
```

### 2.1 Mitigation Measures Implemented
- **Identity-Disjoint Splits:** Partitioned identities across Split A, B, and C prevent memorization of specific celebrity faces.
- **Independent Forensic Evaluator:** An evaluator trained on a separate identity split prevents generative reward hacking.
- **Negative Conditioning (`ABSENT`):** Explicitly suppresses unwanted default features (e.g., smiling, makeup) when absent in the witness narrative.

---

## 3. Technical & Generative Resolution Limitations

### 3.1 Native $64 \times 64$ RGB Resolution
The generative model outputs native $64 \times 64$ pixel images. While sufficient for global facial structure, hair style, jawline geometry, and overall gestalt:
- It **cannot** render fine dermatological details: scars, freckles, moles, acne, fine wrinkles, skin texture, or tattoos.
- It **cannot** render fine ocular structures: iris color nuances, pupil abnormalities, or micro-asymmetries.
- Any upscaling (e.g., bicubic, bilinear) performed in the user interface is for visual display convenience only and adds no real structural information.

### 3.2 Discrete 20-Dimensional GCIT Parameterization
Human facial anatomy varies along continuous, high-dimensional biological axes. The 20 binary/ternary GCIT dimensions represent a discrete approximation. A witness's memory may contain characteristics that fall outside the supported 20 categories (e.g., cleft chins, specific ear shapes, facial piercings, distinct dental gaps).

### 3.3 Latent Space Entanglement
Generative latent spaces are inherently entangled. As demonstrated in Phase 5.7:
- Modifying age or perceived ethnicity in latent space often induces unwanted shifts in jawline, lighting, or hair length.
- Only 5 directions met the strict $1.15\times$ selectivity criterion. Attempting unapproved latent manipulations carries a high risk of distorting the composite away from the witness's memory.

---

## 4. NLP Extraction & Epistemic Uncertainty Limitations

### 4.1 Domain-Specific Generalization Drop
As established in Phase 3.7:
- The BiLSTM-CRF sequence labeler achieves high precision ($0.86$) on structured, formulaic synthetic benchmark sentences.
- When evaluated on natural, unconstrained conversational syntax, entity span F1 drops by **$28.6\%$** to **$0.5926$**.
- The model can struggle with convoluted sentence structures, non-standard slang, rapid clause switching, or ambiguous parenthetical remarks.

### 4.2 Severe Probability Miscalibration
- The sequence labeler exhibits an Expected Calibration Error (ECE) of **$0.4865$**.
- Softmax and CRF forward scores are overconfident, assigning $>90\%$ probability to predictions that fail strict span boundaries.
- **Operational Rule:** Numerical confidence scores are completely suppressed from the user interface and set to `null` to avoid conferring false mathematical authority to automated extractions.

---

## 5. Cognitive Psychology & Witness Memory Integrity

### 5.1 Verbal Overshadowing & Cognitive Contamination
Psychological research (Schooler & Engstler-Schooler, 1990; Wells et al., 2000) demonstrates that the process of generating a facial composite carries inherent risks to witness memory:
- **Verbal Overshadowing:** Articulating facial features can impair the witness's non-verbal perceptual recognition of the perpetrator.
- **Retroactive Interference:** Once a witness views a synthetic composite, their cognitive memory of the actual perpetrator risks being supplanted or contaminated by the synthetic image.

### 5.2 Best-Practice Investigative Protocols
To minimize cognitive contamination, investigators using Phantom AI 3.0 must adhere to the following protocol:
1. **Free-Narrative First:** Conduct a full cognitive interview using open-ended questions before transcribing into the NLP interface.
2. **Do Not Lead the Witness:** Never suggest specific attribute options to fill empty GCIT dimensions. If an attribute was not observed, leave it as `UNKNOWN` ($m_i = 0$).
3. **Investigator Approval Gate:** Always review extracted spans with the witness, correcting any negated or hedged attributes before initiating synthesis.
4. **Present Variations in Parallel:** Show the witness multiple seed variations simultaneously rather than iterating sequentially on a single face, reducing anchor bias.
5. **Clear Disclaimer on Export:** Every exported composite package must display the non-biometric warning and note that the composite is an artistic approximation of memory.

