# Efficient Fine-Tuning of Vision Transformer for Pneumonia Detection
## A Partial Freezing Approach for Medical Image Classification

## Overview

This project presents a complete deep learning pipeline for automated pneumonia detection from Chest X-Ray images using a Vision Transformer (ViT) architecture. The research focuses on improving the stability and efficiency of transfer learning for medical imaging tasks with limited labeled data.

The project was conducted in multiple phases:

- Reproducing a state-of-the-art Vision Transformer baseline for pneumonia detection
- Investigating training collapse during full fine-tuning
- Introducing a partial fine-tuning strategy using selective layer freezing
- Applying label smoothing regularization
- Fixing validation data leakage issues
- Extending the methodology to the BloodMNIST multi-class medical imaging dataset

The proposed approach significantly reduces trainable parameters while maintaining strong classification performance and improving training stability.

---

# Research Contributions

## Key Contributions

- Implemented a Vision Transformer-based pneumonia detection pipeline
- Reproduced baseline results from recent ViT medical imaging research
- Reduced trainable parameters from 86M to 14M using selective layer freezing
- Prevented training collapse caused by overfitting on small medical datasets
- Applied label smoothing for better model calibration and regularization
- Fixed validation transform leakage for reliable evaluation
- Extended the methodology to BloodMNIST for multi-class blood cell classification

---

# Architecture

This project uses:

- ViT-Base-Patch16-224
- HuggingFace Transformers
- PyTorch
- Transfer Learning
- Partial Fine-Tuning

## Fine-Tuning Strategy

### Frozen Components
- Patch Embedding Layer
- Encoder Blocks 0–9

### Trainable Components
- Final LayerNorm
- Last Encoder Blocks
- Classification Head

This reduced trainable parameters by approximately 84%.

---

# Datasets

## 1. Kermany Chest X-Ray Dataset

Binary classification:
- NORMAL
- PNEUMONIA

### Dataset Statistics
- Total Images: 5,863
- Training Images: 4,172
- Validation Images: 1,044
- Test Images: 624

---

## 2. BloodMNIST Dataset

Multi-class blood cell classification with 8 classes:

- Basophil
- Eosinophil
- Erythroblast
- Immature Granulocytes
- Lymphocyte
- Monocyte
- Neutrophil
- Platelet

---

# Experimental Setup

## Environment
- Python
- PyTorch
- HuggingFace Transformers
- Google Colab
- NVIDIA Tesla T4 GPU

## Techniques Used
- Transfer Learning
- Partial Fine-Tuning
- Label Smoothing
- Weighted Random Sampling
- Early Stopping
- Learning Rate Scheduling
- Data Augmentation

---

# Results

## Pneumonia Detection (Kermany Dataset)

| Metric | Phase 3 Result |
|--------|----------------|
| Accuracy | 89.90% |
| Recall (Sensitivity) | 94.87% |
| F1-Score | 92.15% |
| AUC-ROC | 0.9487 |

### Key Observation

The proposed method successfully prevented training collapse while achieving strong generalization performance on limited medical imaging data.

---

## BloodMNIST Results

| Metric | Result |
|--------|---------|
| Accuracy | 90.30% |
| Macro F1-Score | 89.25% |
| AUC-ROC | 0.9917 |

The methodology generalized effectively across different medical imaging domains.


# Future Improvements

- Attention map visualization
- Freeze-depth ablation studies
- MixUp and CutMix augmentation
- Diffusion-based synthetic augmentation
- Lower-resolution ViT variants
- Statistical significance testing



# Technologies Used

- Python
- PyTorch
- HuggingFace Transformers
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

---

# Authors

- Munaza Tariq  
- Amaim Anwar  
- Areeba Arshad  

Department of Data Science  
FAST National University of Computer and Emerging Sciences  
Islamabad, Pakistan


# Acknowledgments

This work builds upon:
- Vision Transformer (ViT)
- HuggingFace Transformers
- Kermany Chest X-Ray Dataset
- MedMNIST Benchmark
- Recent ViT-based medical imaging research
