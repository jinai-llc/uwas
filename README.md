# UWAS: Universal-Wide Association Study

**An AI-augmented multi-modal framework with knowledge regularization for comorbidity-stratified prediction of Alzheimer's disease**

[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)
[![Patent](https://img.shields.io/badge/Patent-US%2063%2F909%2C436-blue.svg)](#patent)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

---

## Overview

UWAS (Universal-Wide Association Study) is a novel analytical framework that integrates clinical, genetic, and neuroimaging data across heterogeneous modalities to identify comorbidity-specific predictive profiles of Alzheimer's disease (AD). Unlike conventional approaches (GWAS, TWAS, EWAS) that operate within a single data modality, UWAS evaluates predictor contributions across modalities simultaneously and detects how those contributions shift across pathologically distinct patient subgroups.

**Key innovation:** UWAS introduces *knowledge-constrained regression*, which leverages LLM-derived biological plausibility scores to regularize statistical associations — producing models whose feature weights reflect field-wide biological consensus rather than cohort-specific confounding.

### Central finding

Applying UWAS to the ADNI cohort (749 CN, 385 AD, 58 sites), we identified a diabetes-dependent reorganization of the multimodal predictive architecture of AD:

| Stratum | PHS weight | DTI AUC | Primary discriminators |
|---------|-----------|---------|----------------------|
| HTN−/DM− | **0.98** | 0.618 | Genetic (PHS, APOE) |
| HTN+/DM− | **0.96** | 0.777 | Genetic (preserved) |
| HTN−/DM+ | **−0.27** | 0.848 | Structural brain markers |
| HTN+/DM+ | **0.17** | 0.861 | Structural brain markers |

Diabetes reverses the direction of genetic risk contribution and shifts the predictive architecture from genetic dominance to neuroimaging dominance. Hypertension alone does not produce this effect.

---

## Pipeline

UWAS operates through a 9-step pipeline organized into four stages:

```
┌─────────────────────────────────────────────────────────────┐
│  A. Data Assembly                                           │
│  Step 1: ADNI Database Extraction (4,792 visits, 58 sites)  │
│  Step 2: Analytic Cohort Assembly (CN=749, AD=385)          │
│  Step 3: Multi-Modal Feature Assembly (14,229 variables)    │
├─────────────────────────────────────────────────────────────┤
│  B. AI-Augmented Feature Engineering                        │
│  Step 4: Longitudinal MRI → β₃ Transformation              │
│  Step 5: LLM-Guided Feature Annotation (GPT-4o-mini)       │
├─────────────────────────────────────────────────────────────┤
│  C. Feature Selection with Knowledge Regularization         │
│  Step 6: Modality-Separated Pre-Filtering (λk = 5.0)       │
│  Step 7: Backward Feature Elimination                       │
├─────────────────────────────────────────────────────────────┤
│  D. Validation & Stratification                             │
│  Step 8: Leave-One-Site-Out (LOSO) Cross-Validation         │
│  Step 9: HTN × DM Stratified Analysis                       │
└─────────────────────────────────────────────────────────────┘
```

### Knowledge-constrained regression

The standard L2 logistic regression loss is augmented with a knowledge penalty:

```
L = −log_likelihood + (1/2C) × Σ(βᵢ²) + λk × Σ((1 − rᵢ/10) × βᵢ²)
```

where `rᵢ` is the LLM-derived AD-relevance score (0–10) for feature *i* and `λk` is the knowledge penalty parameter. Features with high biological plausibility (rᵢ → 10) receive standard L2 regularization; features with low plausibility (rᵢ → 0, e.g., race, scan parameters) receive additional shrinkage proportional to `λk`.

### Modality-specific confound correction

| Modality | Confound | Correction | Pipeline step |
|----------|----------|------------|---------------|
| Clinical | Cohort demographics (race, sex, site) | Knowledge-regularization via LLM plausibility scores | Step 6 |
| Neuroimaging | Normal aging (age-related atrophy) | Age × Diagnosis interaction beta (β₃) | Step 4 |

---

## Requirements

```
Python >= 3.11
scikit-learn >= 1.3
numpy
pandas
scipy
matplotlib
openai          # for GPT-4o-mini feature annotation
```

## Installation

```bash
git clone https://github.com/jinai-llc/uwas.git
cd uwas
pip install -r requirements.txt
```

## Data access

This study uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). To obtain access:

1. Visit [adni.loni.usc.edu](https://adni.loni.usc.edu)
2. Apply for data access through the ADNI Data Use Agreement
3. Download the required tables (demographics, medical history, APOE genotype, PHS, FreeSurfer, DTI)

Due to ADNI data use policies, raw data cannot be redistributed in this repository.

---

## Results summary

### Unstratified modality-specific models

| Modality | Features | LOSO AUC |
|----------|----------|----------|
| Clinical | 8 | 0.776 ± 0.070 |
| T1 Structural MRI | 10 | 0.845 ± 0.126 |
| DTI Diffusion | 10 | 0.762 ± 0.132 |
| **Combined** | **10** | **0.950** |

### HTN × DM stratified mixed-modality models

| Stratum | n (CN/AD) | AUC | Features | Genetic present? |
|---------|-----------|-----|----------|-----------------|
| HTN−/DM− | 198/149 | 0.868 | 6 | Yes (PHS, Sex) |
| HTN+/DM− | 167/124 | 0.986 | 6 | Yes (APOE, PHS) |
| HTN−/DM+ | 237/51 | 0.975 | 6 | No |
| HTN+/DM+ | 147/61 | 0.939 | 6 | APOE only (weak) |

---

## Citation

If you use UWAS in your research, please cite:

```bibtex
@article{jin2026uwas,
  title={Differential associations of comorbid diabetes and hypertension with 
         multimodal predictive profiles in Alzheimer's disease: an AI-guided 
         analysis of ADNI cohorts},
  author={Jin, Guangxu and Du, Heng},
  journal={[submitted]},
  year={2026}
}
```

## Patent

U.S. Provisional Application No. 63/909,436.

## License

Copyright © 2026 JINAI L.L.C. All rights reserved.

This software is provided for academic research purposes only. Commercial use, redistribution, or incorporation into derivative works requires a written license from JINAI L.L.C. Contact: [email]

## Acknowledgments

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012).

---

*A collaborative effort between the Jin Lab ([JINAI L.L.C.](https://github.com/jinai-llc)) and the Du Lab*
