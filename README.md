# Regularization-on-CIFAR-10

Comparing regularization techniques on the CIFAR-10 dataset.

This repository is a work in progress. A full PyTorch experiment framework for comparing weight-based, stochastic, data augmentation, label-based, and training-procedure regularization techniques will be scaffolded shortly.

## Baseline

The baseline uses **ResNet-18** (adapted for 32×32 CIFAR-10 images) trained with **SGD** (lr=0.1, momentum=0.9, weight-decay=5e-4) and cosine-annealing learning-rate schedule for 200 epochs.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the baseline
python train_baseline.py

# Train with custom options
python train_baseline.py --epochs 100 --lr 0.05 --lr-schedule step
```

### Project Structure

```
├── models/
│   └── resnet.py          # ResNet-18 adapted for CIFAR-10
├── train_baseline.py      # Baseline training script (ResNet-18 + SGD)
├── utils.py               # Data loading and evaluation helpers
├── requirements.txt       # Python dependencies
└── README.md
```

### Command-Line Options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 0.1 | Initial learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 5e-4 | L2 weight decay |
| `--batch-size` | 128 | Mini-batch size |
| `--lr-schedule` | cosine | `cosine` or `step` |
| `--lr-milestones` | 100 150 | Epochs to decay LR (step schedule) |
| `--lr-gamma` | 0.1 | LR decay factor (step schedule) |
| `--seed` | 42 | Random seed |
| `--data-dir` | ./data | CIFAR-10 download directory |
| `--checkpoint-dir` | ./checkpoints | Saved model directory |