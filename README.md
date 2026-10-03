# OOD Detection with Metric Learning

This project explores whether metric learning improves out-of-distribution detection.

The model is trained only on 5 CIFAR-10 classes, treated as in-distribution. OOD detection is evaluated on an external dataset. SVHN is used by default, while CIFAR-100 and Textures/DTD can be selected through configuration.

## Experiments

- CrossEntropy baseline
- CrossEntropy + Triplet loss
- CrossEntropy + SupCon loss
- CrossEntropy + Triplet loss + synthetic outlier exposure

Synthetic outlier training uses random noise images and an energy-based loss to encourage the model to assign high energy / low confidence to outlier inputs.

## OOD Scoring Methods

Each model is evaluated with the following OOD scores:

- MSP
- Energy
- Prototype distance
- Mahalanobis distance
- kNN distance

Reported metrics:

- ID classification accuracy
- OOD AUROC
- FPR95 at 95% TPR

## Visualization

The project can generate UMAP plots of ID and OOD embeddings.

## Setup

Install dependencies:

    pip install torch torchvision numpy pandas scikit-learn matplotlib umap-learn pytorch-metric-learning

## Run

Run the main script:

    python main.py

## Configuration

Most settings are in the Config class.

Example:

    ood_dataset = "svhn"
    epochs = 15
    batch_size = 128
    experiments = ["ce", "triplet", "supcon", "triplet_outliers"]

Supported OOD datasets:

    ood_dataset = "svhn"
    ood_dataset = "cifar100"
    ood_dataset = "textures"

For limited compute, run fewer experiments:

    experiments = ["ce", "triplet_outliers"]

## Outputs

Results are saved in:

    results/

Typical outputs include:

- training logs
- OOD metrics CSV
- model checkpoints
- UMAP plots

## Notes

- Current experiments use a single seed.
- Results are preliminary until full runs complete.
- SVHN is easier; CIFAR-100 or Textures/DTD are harder OOD options.
