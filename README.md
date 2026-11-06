# Metric Learning for Out-of-Distribution Detection

This repository contains a PyTorch project exploring whether metric-learning objectives improve **out-of-distribution (OOD) detection** compared with standard cross-entropy training.

The project trains a ResNet-18 model only on in-distribution data and then evaluates multiple OOD scoring functions on an external OOD test set.

---

## Project Overview

The goal of this project is to answer the following question:

> Does training a classifier with metric-learning losses produce embeddings that are better suited for detecting out-of-distribution samples than training with cross-entropy alone?

The workflow is:

1. Train a ResNet-18 classifier on a subset of CIFAR-10.
2. Learn an embedding space using one of several training objectives:
   - Cross-Entropy only
   - Cross-Entropy + Triplet loss
   - Cross-Entropy + SupCon loss
   - Cross-Entropy + Triplet loss + synthetic outlier training
3. Fit OOD scoring statistics on clean in-distribution training embeddings.
4. Evaluate OOD detection performance on an external OOD dataset.
5. Compare multiple OOD scoring methods.
6. Visualize embedding spaces using UMAP.

---

## Task Definition

### In-Distribution Data

The in-distribution data is CIFAR-10 using classes `0-4`:

```text
0: airplane
1: automobile
2: bird
3: cat
4: deer
```

The model is trained only on these five classes.

### Out-of-Distribution Data

The default OOD dataset is **SVHN** test data.

SVHN is external to CIFAR-10 and is not used during training.

Supported OOD datasets in the configuration include:

```python
ood_dataset = "svhn"
ood_dataset = "cifar100"
ood_dataset = "textures"
```

Note: `textures` requires a torchvision version that supports the DTD dataset. If it is unavailable, the script can fall back to SVHN.

---

## Key Final Results

The final reported results were obtained with:

```text
epochs = 20
ood_dataset = svhn
seed = 42
outlier_weight = 0.05
```

### Best OOD Detection Result

The strongest OOD detection result is:

```text
Triplet + Mahalanobis
```

| Experiment | OOD Score | ID Accuracy (%) | AUROC (%) ↑ | FPR95 (%) ↓ |
|---|---|---:|---:|---:|
| **Triplet** | **Mahalanobis** | **91.14** | **95.36** | **21.72** |

This configuration gives the best AUROC and the best FPR95 among all experiments.

---

### Best ID Accuracy Result

The highest in-distribution classification accuracy is achieved by SupCon:

```text
SupCon ID Accuracy: 92.16%
```

A strong SupCon-based OOD configuration is:

| Experiment | OOD Score | ID Accuracy (%) | AUROC (%) ↑ | FPR95 (%) ↓ |
|---|---|---:|---:|---:|
| SupCon | Energy | 92.16 | 85.56 | 36.68 |
| SupCon | kNN | 92.16 | 87.04 | 37.26 |

SupCon is a good choice if preserving or improving ID classification accuracy is important.

---

## Main Findings

### 1. Triplet loss significantly improves embedding-based OOD detection

Triplet training improves embedding-space OOD scores substantially.

For example:

| Method | CE AUROC (%) | Triplet AUROC (%) |
|---|---:|---:|
| Prototype | 83.46 | 92.04 |
| Mahalanobis | 86.39 | 95.36 |
| kNN | 85.78 | 92.61 |

The largest improvement is observed with **Mahalanobis distance**:

```text
CE Mahalanobis AUROC:        86.39%
Triplet Mahalanobis AUROC:   95.36%
```

This suggests that triplet loss produces tighter and better-separated class-conditional feature distributions.

---

### 2. SupCon improves ID accuracy and logit-based scores

SupCon achieves the highest ID accuracy:

```text
CE ID Accuracy:      90.72%
Triplet ID Accuracy: 91.14%
SupCon ID Accuracy:  92.16%
```

SupCon also improves Energy-based OOD detection compared with CE:

```text
CE Energy AUROC:      82.73%
SupCon Energy AUROC:  85.56%
```

However, SupCon does not match triplet training for Mahalanobis-based OOD detection:

```text
Triplet Mahalanobis AUROC: 95.36%
SupCon Mahalanobis AUROC:  88.31%
```

---

### 3. Synthetic outlier training did not improve over standard triplet training

The `triplet_outliers` experiment uses random synthetic outlier images and an energy-based outlier loss.

With `epochs = 20` and `outlier_weight = 0.05`, ID accuracy recovers compared with earlier, more aggressive settings:

```text
Triplet ID Accuracy:            91.14%
Triplet + Outliers ID Accuracy: 90.86%
```

However, OOD detection performance is generally worse than standard triplet training:

| Method | Triplet AUROC (%) | Triplet + Outliers AUROC (%) |
|---|---:|---:|
| MSP | 82.97 | 80.70 |
| Energy | 84.91 | 76.99 |
| Prototype | 92.04 | 81.79 |
| Mahalanobis | 95.36 | 86.74 |
| kNN | 92.61 | 84.16 |

The best `triplet_outliers` result is Mahalanobis:

```text
AUROC:  86.74%
FPR95:  44.68%
```

This is only slightly better than the CE Mahalanobis baseline and far below the standard triplet Mahalanobis result.

Therefore, the synthetic outlier experiment is included as an ablation/negative result. The current random-noise outlier strategy does not provide a clear benefit.

---

## Full Results Table

Higher is better for **ID Accuracy** and **AUROC**.  
Lower is better for **FPR95**.

| Experiment | OOD Score | ID Accuracy (%) | AUROC (%) ↑ | FPR95 (%) ↓ |
|---|---|---:|---:|---:|
| CE | MSP | 90.72 | 83.56 | 42.32 |
| CE | Energy | 90.72 | 82.73 | 39.56 |
| CE | Prototype | 90.72 | 83.46 | 50.50 |
| CE | Mahalanobis | 90.72 | 86.39 | 40.98 |
| CE | kNN | 90.72 | 85.78 | 37.12 |
| Triplet | MSP | 91.14 | 82.97 | 42.46 |
| Triplet | Energy | 91.14 | 84.91 | 36.58 |
| Triplet | Prototype | 91.14 | 92.04 | 30.60 |
| **Triplet** | **Mahalanobis** | **91.14** | **95.36** | **21.72** |
| Triplet | kNN | 91.14 | 92.61 | 25.04 |
| SupCon | MSP | 92.16 | 83.94 | 39.02 |
| SupCon | Energy | 92.16 | 85.56 | 36.68 |
| SupCon | Prototype | 92.16 | 83.13 | 42.62 |
| SupCon | Mahalanobis | 92.16 | 88.31 | 35.58 |
| SupCon | kNN | 92.16 | 87.04 | 37.26 |
| Triplet + Outliers | MSP | 90.86 | 80.70 | 47.68 |
| Triplet + Outliers | Energy | 90.86 | 76.99 | 49.86 |
| Triplet + Outliers | Prototype | 90.86 | 81.79 | 56.32 |
| Triplet + Outliers | Mahalanobis | 90.86 | 86.74 | 44.68 |
| Triplet + Outliers | kNN | 90.86 | 84.16 | 40.56 |

---

## Recommended Configurations

### Best OOD detection performance

Use:

```python
experiment = "triplet"
score_method = "Mahalanobis"
```

This gives the best overall OOD detection performance in the current experiments.

---

### Best balance between ID accuracy and OOD detection

Use:

```python
experiment = "supcon"
score_method = "Energy"
```

or:

```python
experiment = "supcon"
score_method = "kNN"
```

These configurations give the highest ID accuracy while still maintaining competitive OOD detection performance.

---

## Repository Structure

```text
.
├── main.ipynb           # Main training and evaluation notebook
├── README.md            # Project documentation
└── results/             # Umap plots
```

After running the script, the `results/` directory may contain:

```text
results/
├── ood_metrics_seed42.csv
├── logs_ce_seed42.csv
├── logs_triplet_seed42.csv
├── logs_supcon_seed42.csv
├── logs_triplet_outliers_seed42.csv
├── model_ce_seed42.pth
├── model_triplet_seed42.pth
├── model_supcon_seed42.pth
├── model_triplet_outliers_seed42.pth
├── umap_ce_svhn_seed42.png
└── umap_triplet_outliers_svhn_seed42.png
```

---

## Setup

### Requirements

Create a `requirements.txt` file containing:

```text
torch
torchvision
numpy
pandas
scikit-learn
pytorch-metric-learning
matplotlib
umap-learn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch build for your CUDA version.

---

## Running the Experiments

Run the main script:

```bash
python main.py
```

The script will:

1. Download CIFAR-10 and the selected OOD dataset if needed.
2. Train each experiment defined in `Config.experiments`.
3. Evaluate all selected OOD scoring methods.
4. Save metrics, logs, checkpoints, and UMAP visualizations to `results/`.

---

## Configuration

Most experiment settings are controlled through the `Config` class.

Example:

```python
class Config:
    seed = 42
    data_root = "./data"
    output_dir = "results"

    id_classes = [0, 1, 2, 3, 4]

    ood_dataset = "svhn"

    experiments = [
        "ce",
        "triplet",
        "supcon",
        "triplet_outliers",
    ]

    epochs = 20
    batch_size = 128
    lr = 0.05

    outlier_weight = 0.05
    outlier_batch_size = 32

    score_methods = [
        "MSP",
        "Energy",
        "Prototype",
        "Mahalanobis",
        "kNN",
    ]

    run_umap = True
```

---

## Experiment Names

The script supports the following experiment names:

```python
"ce"
"triplet"
"ms"
"supcon"

"ce_outliers"
"triplet_outliers"
"ms_outliers"
"supcon_outliers"
```

For example:

```python
experiments = [
    "ce",
    "triplet",
    "supcon",
    "triplet_outliers",
]
```

For limited compute resources, you can run fewer experiments:

```python
experiments = [
    "triplet",
]
```

or:

```python
experiments = [
    "ce",
    "triplet",
]
```

---

## OOD Scoring Methods

Five OOD scoring methods are evaluated.

### MSP

Maximum Softmax Probability.

```text
score = 1 - max softmax probability
```

Higher score means more likely OOD.

---

### Energy

Energy-based score:

```text
score = -logsumexp(logits / T)
```

Higher energy means more likely OOD.

---

### Prototype

Cosine distance to the nearest class prototype:

```text
score = 1 - max cosine similarity to ID class prototypes
```

Prototypes are computed from clean ID training embeddings.

---

### Mahalanobis

Minimum Mahalanobis distance to ID class means using a tied covariance estimate:

```text
score = min_c (x - mean_c)^T Sigma^{-1} (x - mean_c)
```

This is the strongest scoring method in the current experiments.

---

### kNN

Distance-based kNN score in normalized embedding space:

```text
score = 1 - mean top-k cosine similarity to ID training embeddings
```

---

## Model Details

The model is a CIFAR-friendly ResNet-18.

Modifications from the standard ImageNet ResNet:

```text
- First convolution changed from 7x7 stride 2 to 3x3 stride 1
- Max pooling removed
- Final classifier replaced with a custom linear head
```

The model outputs:

```text
- 5-class logits
- 512-dimensional L2-normalized embeddings
```

---

## Training Objectives

### Cross-Entropy

Standard supervised classification on the five in-distribution classes.

---

### Triplet

Cross-Entropy plus triplet loss.

The triplet loss uses:

```text
- L2-normalized embeddings
- BatchHardMiner
- TripletMarginLoss
```

This objective encourages tighter and more separated ID class clusters.

---

### SupCon

Cross-Entropy plus supervised contrastive loss.

SupCon achieves the highest ID classification accuracy in the final experiments.

---

### Triplet + Synthetic Outliers

Cross-Entropy plus triplet loss plus an energy-based synthetic outlier loss.

The synthetic outliers are random RGB noise images.

The outlier loss encourages:

```text
- low energy for in-distribution samples
- high energy for synthetic outlier samples
```

In the final experiments, this method recovers ID accuracy but does not outperform standard triplet training for OOD detection.

---

## Data Augmentation

Training uses standard CIFAR-style augmentation:

```text
- Resize to 32x32
- Random crop with padding
- Random horizontal flip
- CIFAR-10 normalization
```

Evaluation and prototype fitting use clean test-style transforms.

---

## UMAP Visualization

The script can generate UMAP plots comparing in-distribution and OOD embeddings.

UMAP settings are controlled by:

```python
run_umap = True

umap_experiments = [
    "ce",
    "triplet_outliers",
]

umap_max_points = 2500
```

UMAP plots are saved to the `results/` directory.

If `umap-learn` is not installed, the script will still run and will skip UMAP generation.

---

## Evaluation Metrics

### ID Accuracy

Classification accuracy on the in-distribution CIFAR-10 test classes.

Higher is better.

---

### AUROC

Area Under the Receiver Operating Characteristic Curve.

Higher is better.

The positive class is OOD.

---

### FPR95

False Positive Rate at 95% True Positive Rate.

Lower is better.

This metric is useful for deployment-oriented OOD detection because it measures how many ID samples are incorrectly flagged as OOD when the detector is operating at high OOD recall.

---

## Interpretation of the Final Results

The final results support the following conclusions:

1. **Metric learning improves OOD detection.**  
   Triplet training produces a large improvement in embedding-based OOD scores, especially Mahalanobis distance.

2. **Mahalanobis distance is the strongest OOD score.**  
   It benefits strongly from the improved embedding structure produced by triplet training.

3. **SupCon is useful when ID accuracy matters.**  
   SupCon gives the highest ID accuracy and produces strong Energy and kNN OOD scores.

4. **Random synthetic outliers are not sufficient.**  
   The current outlier experiment does not improve OOD detection beyond standard triplet training. More realistic auxiliary OOD data would likely be needed for outlier exposure to help.

5. **The best current configuration is Triplet + Mahalanobis.**

---

## Limitations

This project has several limitations:

- Results are from a single seed.
- The OOD dataset is SVHN, which is external but still relatively simple.
- CIFAR-100 or Textures/DTD would provide additional OOD evaluation settings.
- Synthetic outliers are simple random-noise images.
- No extensive hyperparameter search was performed.
- The project uses a relatively small ResNet-18 model trained from scratch.
- Threshold selection for deployment is not fully developed.

---

## Future Work

Possible next steps include:

- Running multiple seeds and reporting mean/std.
- Evaluating on harder OOD datasets such as Textures/DTD or CIFAR-100.
- Replacing random-noise outliers with real auxiliary OOD images.
- Exploring score fusion, for example:
  - Mahalanobis + kNN
  - Mahalanobis + Energy
  - Prototype + Energy
- Tuning OOD thresholds on a validation set.
- Testing larger backbones or pretrained feature extractors.
- Exploring OOD-specific training objectives.
- Adding temperature scaling or calibration analysis.

---

## Computational Notes

The experiments are designed to be adjustable for limited compute.

To reduce runtime:

```python
epochs = 10

experiments = [
    "ce",
    "triplet",
]

score_methods = [
    "Mahalanobis",
    "kNN",
]

run_umap = False
```

To reproduce the final reported setting:

```python
epochs = 20

experiments = [
    "ce",
    "triplet",
    "supcon",
    "triplet_outliers",
]

ood_dataset = "svhn"

outlier_weight = 0.05

score_methods = [
    "MSP",
    "Energy",
    "Prototype",
    "Mahalanobis",
    "kNN",
]
```

---

## Summary

This project demonstrates that metric learning can substantially improve OOD detection when paired with embedding-space scoring methods.

The strongest result is:

```text
Triplet-trained embeddings + Mahalanobis distance
```

which achieves:

```text
ID Accuracy: 91.14%
AUROC:       95.36%
FPR95:       21.72%
```

SupCon provides the highest ID accuracy and remains a strong option when classification performance is the primary concern.

The current synthetic outlier strategy is included as a negative result and would require more realistic outlier data or a different outlier objective to be competitive.
