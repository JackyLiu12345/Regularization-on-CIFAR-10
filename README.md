# Regularization on CIFAR-10

A comparative study of regularization techniques for deep learning, using **ResNet-18** on the **CIFAR-10** dataset.

## Overview

This repository provides a modular framework for training a CIFAR-10 adapted ResNet-18 with various regularization methods. Each method can be used independently via a dedicated training script, or combined freely through the unified `train.py` entry point.

### Regularization Methods Implemented

| Method | Flag(s) | Script | Reference |
|---|---|---|---|
| **Baseline** (no regularization) | — | `train_baseline.py` | — |
| **L2 Weight Decay** | `--weight-decay` | `train.py` | Krogh & Hertz, 1991 |
| **Dropout** | `--dropout` | `train_dropout.py` | [Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html) |
| **Label Smoothing** | `--label-smoothing` | `train_label_smoothing.py` | [Szegedy et al., 2016](https://arxiv.org/abs/1512.00567) |
| **Mixup** | `--mixup-alpha` | `train_mixup.py` | [Zhang et al., 2018](https://arxiv.org/abs/1710.09412) |
| **Cutout** | `--cutout-length` | `train_cutout.py` | [DeVries & Taylor, 2017](https://arxiv.org/abs/1708.04552) |

All methods can also be combined via the unified script `train.py`.

## Setup

```bash
# Clone the repository
git clone https://github.com/JackyLiu12345/Regularization-on-CIFAR-10.git
cd Regularization-on-CIFAR-10

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision ≥ 0.13
- matplotlib ≥ 3.5

## Quick Start

### Baseline (no extra regularization)

```bash
python train_baseline.py --plot
```

### Individual Regularization Scripts

```bash
# Dropout (default rate=0.3)
python train_dropout.py --plot

# Label Smoothing (default factor=0.1)
python train_label_smoothing.py --plot

# Mixup (default alpha=0.2)
python train_mixup.py --plot

# Cutout (default mask=16×16)
python train_cutout.py --plot
```

### Unified Script — Mix & Match

```bash
# Combine dropout + label smoothing + Cutout
python train.py --dropout 0.3 --label-smoothing 0.1 --cutout-length 16 --plot

# Mixup + L2 weight decay
python train.py --mixup-alpha 0.2 --weight-decay 1e-3 --plot

# All together
python train.py --dropout 0.3 --label-smoothing 0.1 --mixup-alpha 0.2 --cutout-length 16 --plot
```

## Visualization

All training scripts support the `--plot` flag, which generates and saves:

- `loss_curve.png` — Train loss and val loss vs. epoch
- `accuracy_curve.png` — Train accuracy and val accuracy vs. epoch
- `training_curves.png` — Combined side-by-side view

Plots are saved to `./plots/` by default (configurable with `--plot-dir`).

## Project Structure

```
Regularization-on-CIFAR-10/
├── models/
│   ├── __init__.py
│   └── resnet.py              # ResNet-18 adapted for CIFAR-10 (32×32), optional dropout
├── train.py                   # Unified script with all regularization flags
├── train_baseline.py          # Baseline: ResNet-18 + SGD, no extra regularization
├── train_dropout.py           # Dropout regularization
├── train_label_smoothing.py   # Label smoothing regularization
├── train_mixup.py             # Mixup data augmentation
├── train_cutout.py            # Cutout data augmentation
├── utils.py                   # Data loaders, Cutout transform, Mixup helpers, visualization
├── requirements.txt
├── .gitignore
└── README.md
```

## Command-Line Options

All scripts share a common set of arguments:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 0.1 | Initial learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--weight-decay` | 5e-4 | L2 weight decay |
| `--batch-size` | 128 | Mini-batch size |
| `--lr-schedule` | cosine | `cosine` or `step` |
| `--lr-milestones` | 100 150 | Epochs to decay LR (step schedule only) |
| `--lr-gamma` | 0.1 | LR decay factor (step schedule only) |
| `--seed` | 42 | Random seed |
| `--data-dir` | ./data | CIFAR-10 download directory |
| `--checkpoint-dir` | ./checkpoints | Saved model directory |
| `--plot` | off | Save training curve plots |
| `--plot-dir` | ./plots | Directory for saved plots |

### Regularization-specific options (unified `train.py`)

| Flag | Default | Description |
|---|---|---|
| `--dropout` | 0.0 | Dropout probability before the final pooling layer |
| `--label-smoothing` | 0.0 | Label smoothing factor for CrossEntropyLoss |
| `--mixup-alpha` | 0.0 | Mixup Beta distribution parameter (0 = off) |
| `--cutout-length` | 0 | Cutout mask side length in pixels (0 = off) |

## Model

**ResNet-18 (CIFAR-10 variant)**

- Initial 7×7 conv replaced with 3×3 conv (stride 1, padding 1) to preserve spatial resolution on 32×32 inputs
- No initial max-pooling
- Standard BasicBlock residual blocks: [2, 2, 2, 2]
- Kaiming (He) weight initialization
- ~11.2M parameters
- Optional dropout before the final average pooling layer

## Regularization Method Details

### L2 Weight Decay
Adds an L2 penalty on model weights to the loss function, discouraging large weights. Controlled via `--weight-decay` (default 5e-4).

### Dropout
Randomly zeroes elements of the feature map before the final pooling layer during training. Controlled via `--dropout` (typical range: 0.1–0.5).

### Label Smoothing
Replaces hard 0/1 targets with soft targets (e.g. 0.9 for the correct class, 0.01 for others when smoothing=0.1). Prevents overconfident predictions. Controlled via `--label-smoothing` (typical range: 0.05–0.2).

### Mixup
Creates virtual training examples by linearly interpolating pairs of inputs and their labels. Encourages linear behavior between training examples and improves generalization. Controlled via `--mixup-alpha` (typical range: 0.1–1.0).

### Cutout
Randomly masks a square region of the input image during training, forcing the model to attend to less discriminative parts. Controlled via `--cutout-length` (typical value: 16 for CIFAR-10).

## License

This project is for educational and research purposes.