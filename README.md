# TCMNet: Concept-Guided Diagnosis & Prescription in Traditional Chinese Medicine

## Overview
TCMNet is a concept-guided multi-task neural network for predicting Traditional Chinese Medicine (TCM) syndromes and recommending herbal treatments from high-dimensional symptom data. The model integrates domain-specific medical concepts to improve interpretability while maintaining strong predictive performance.

---

## Problem
Traditional Chinese Medicine diagnosis relies on complex relationships between symptoms, underlying syndromes, and treatments. These relationships are difficult to model computationally due to:
- high-dimensional and sparse symptom data  
- overlapping symptom patterns across syndromes  
- reliance on qualitative domain knowledge  

This project aims to learn structured diagnostic relationships and treatment recommendations using machine learning.

---

## Approach

### Model Architecture
- Input: **1,861-dimensional binary symptom vector**
- Outputs:
  - **228-class syndrome prediction** (classification)
  - **14-dimensional concept vector** (continuous)

The model is trained as a **multi-task neural network**:
- Shared representation layers  
- Two prediction heads:
  - Syndrome classification (softmax)
  - Concept prediction (sigmoid)

### Loss Function
Joint objective combining:
- Cross-entropy loss (syndrome classification)  
- Mean squared error (concept prediction)  

This encourages the model to learn **medically meaningful intermediate representations**.

---

## Baseline: Tree-Augmented Naive Bayes (TAN)
We implement a probabilistic baseline using Tree-Augmented Naive Bayes:
- Models dependencies between features via a maximum spanning tree  
- Edges weighted by conditional mutual information  
- Maintains the class variable as a parent of all features  

---

## Data
- Source: SymMap knowledge graph  
- Synthetic dataset generation:
  - ~25 samples per syndrome  
  - ~5,700 total examples  

- Features:
  - 1,861 binary symptom features  
  - 14 concept features (Eight Principles + organ systems)  

---

## Results

| Model   | Accuracy | Macro-F1 | Top-5 Accuracy |
|--------|---------|----------|----------------|
| TAN    | 58.07%  | 56.55%   | 90.09%         |
| TCMNet | **88.25%** | **86.79%** | **99.21%**     |

**Concept prediction MSE:** 0.0016

### Key Findings
- Neural model significantly outperforms probabilistic baseline  
- Concept layer improves interpretability and generalization  
- Errors primarily occur between syndromes with overlapping symptoms  

---

## Herb Recommendation
We explore two approaches:
1. **Syndrome-based ranking** using syndrome–herb mappings  
2. **Concept-guided scoring**, combining:
   - cosine similarity between predicted concepts and herb profiles  
   - prior probabilities from training data  

---

## Tech Stack
- PyTorch  
- NumPy / Pandas  
- NetworkX (for TAN)  
- Matplotlib  

---

## Repository Structure

```bash
pipeline/
  data/              # Original, processed, patient, and legacy datasets
  data_processing/   # Feature extraction and SymMap processing scripts
  data_generation/   # Synthetic patient dataset generation
  training/          # Neural network and TAN training scripts
  artifacts/         # Frozen model and mappings for future inference APIs
  outputs/           # Generated metrics and predictions
backend/             # Inference APIs
frontend/            # User interaction layer
```

## Regenerating Inference Artifacts

Install dependencies, then run the exporter from the project root:

```bash
pip install -r requirements.txt
python3 pipeline/training/export_tcmnet_artifacts.py
python3 pipeline/training/export_metadata_artifacts.py
```

This writes `pipeline/artifacts/tcmnet.pt`, model config, symptom ordering, syndrome index mapping, concept label order, herb IDs, and recommendation matrices. For a quick smoke test, use `--epochs 1`.
The metadata export adds lean `id`/`label` records for symptoms, syndromes, herbs, and concepts.
