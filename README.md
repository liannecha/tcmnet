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
- Input: **1,869-dimensional binary symptom vector**
- Outputs:
  - **14-dimensional concept vector** (continuous)
  - **228-class syndrome prediction** (classification)
  - **596-dimensional neural herb ranking vector**

The model is trained as a **multi-task neural network**:
- Shared representation layers  
- Three prediction heads:
  - Concept prediction (sigmoid)
  - Syndrome classification (softmax)
  - Herb ranking (neural herb-head logits)

### Loss Function
Joint objective combining:
- Mean squared error (concept prediction)
- Cross-entropy loss (syndrome classification)
- Hybrid herb loss combining pairwise BPR ranking, multi-label BCE, and baseline-score distillation

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
  - 1,869 binary symptom features
  - 14 concept features (Eight Principles + organ systems)
  - 596 herb targets derived from syndrome-herb associations

---

## Results

| Model   | Syndrome Accuracy | Macro-F1 | Top-5 Syndrome Accuracy |
|--------|-------------------|----------|-------------------------|
| TAN    | 58.07%            | 56.55%   | 90.09%                  |
| TCMNet | **86.93%**        | **85.44%** | **98.16%**            |

**Concept prediction MSE:** 0.0016

### Herb Ranking

The active app recommender uses the neural herb head. The older matrix-based formula is preserved under `pipeline/baselines/` for comparison.

| Herb Ranker | Precision@5 | Recall@5 | Hit@5 | Precision@10 | Recall@10 | Hit@10 |
|-------------|-------------|----------|-------|--------------|-----------|--------|
| Matrix baseline | 53.94% | 68.43% | 90.09% | 39.64% | 76.47% | 90.94% |
| Neural herb head | **57.56%** | **75.44%** | **97.33%** | **42.63%** | **85.66%** | **99.05%** |

### Key Findings
- Neural model significantly outperforms probabilistic baseline  
- Concept layer improves interpretability and generalization  
- Errors primarily occur between syndromes with overlapping symptoms  
- The neural herb head outperforms the matrix baseline under inference conditions, where herb ranking uses model-predicted concepts and syndromes.

---

## Herb Recommendation
TCMNet now ranks herbs with a neural herb head conditioned on shared symptom features, predicted concepts, and syndrome logits. Herb scores are exported with the model and used by both backend inference and browser-side local inference.

The previous matrix ranker is retained as a baseline:

```text
score = alpha * concept_similarity + (1 - alpha) * syndrome_prior
```

This baseline is useful for ablations and sanity checks, but it is no longer the active app recommendation method.

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
  data/              # Original, processed, and patient datasets
  data_processing/   # Feature extraction and SymMap processing scripts
  data_generation/   # Synthetic patient dataset generation
  training/          # Neural network training and artifact export scripts
  baselines/         # TAN and matrix-based baseline models
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

This writes `pipeline/artifacts/tcmnet.pt`, model config, symptom ordering, syndrome index mapping, concept label order, herb IDs, herb-head weights, and comparison matrices for explanation/baseline evaluation. For a quick smoke test, use `--epochs 1`.
The metadata export adds lean `id`/`label` records for symptoms, syndromes, herbs, and concepts.

After retraining, refresh the browser-side artifact bundle:

```bash
python3 tools/export_frontend_artifacts.py
```
