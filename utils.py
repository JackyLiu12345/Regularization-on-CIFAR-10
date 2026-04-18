"""Shared utilities for data loading, training helpers, and visualization."""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ── Dataset statistics ───────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DATASET_INFO = {
    "cifar10": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD, "num_classes": 10,
                "cls": torchvision.datasets.CIFAR10},
    "cifar100": {"mean": CIFAR100_MEAN, "std": CIFAR100_STD, "num_classes": 100,
                 "cls": torchvision.datasets.CIFAR100},
}


# ── Data loading ─────────────────────────────────────────────────────────────

class Cutout:
    """Randomly mask a square region of the input image (tensor).

    Reference: DeVries & Taylor, "Improved Regularization of Convolutional
    Neural Networks with Cutout", 2017.  https://arxiv.org/abs/1708.04552

    Parameters
    ----------
    length : int
        Side length of the square mask.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        """Apply cutout to a CHW tensor."""
        _, h, w = img.size()
        mask = torch.ones_like(img)

        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)

        y1 = max(0, cy - self.length // 2)
        y2 = min(h, cy + self.length // 2)
        x1 = max(0, cx - self.length // 2)
        x2 = min(w, cx + self.length // 2)

        mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=2,
                        cutout_length=0):
    """Return CIFAR-10 train and test data loaders with standard preprocessing.

    Training set uses random cropping and horizontal flipping.
    Both sets are normalized to per-channel mean/std of CIFAR-10.

    Parameters
    ----------
    cutout_length : int
        If > 0, apply Cutout with the given mask side length (e.g. 16).
    """
    return get_data_loaders("cifar10", data_dir=data_dir, batch_size=batch_size,
                            num_workers=num_workers, cutout_length=cutout_length)


def get_data_loaders(dataset="cifar10", data_dir="./data", batch_size=128,
                     num_workers=2, cutout_length=0):
    """Return train and test data loaders for the specified dataset.

    Supported datasets: ``cifar10``, ``cifar100``.

    Parameters
    ----------
    dataset : str
        Dataset name (``"cifar10"`` or ``"cifar100"``).
    cutout_length : int
        If > 0, apply Cutout with the given mask side length.
    """
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_INFO)}")

    info = DATASET_INFO[dataset]
    mean, std = info["mean"], info["std"]
    dataset_cls = info["cls"]

    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if cutout_length > 0:
        train_transforms.append(Cutout(cutout_length))

    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = dataset_cls(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_cls(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def num_classes_for(dataset="cifar10"):
    """Return the number of classes for the given dataset name."""
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_INFO)}")
    return DATASET_INFO[dataset]["num_classes"]


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_accuracy(model, data_loader, device):
    """Compute top-1 accuracy of *model* on the given *data_loader*."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def evaluate_loss_and_accuracy(model, data_loader, criterion, device):
    """Compute average loss and top-1 accuracy on a data loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=1.0):
    """Apply Mixup to a batch.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2018.
    https://arxiv.org/abs/1710.09412

    Returns mixed inputs, pairs of targets, and the mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss as a weighted combination."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Visualization ────────────────────────────────────────────────────────────

def plot_training_curves(history, save_dir="./plots", title_prefix=""):
    """Plot and save training curves from a history dictionary.

    Parameters
    ----------
    history : dict
        Must contain keys: ``train_loss``, ``val_loss``, ``train_acc``,
        ``val_acc``, each mapping to a list of per-epoch values.
    save_dir : str
        Directory in which to save the PNG files.
    title_prefix : str
        Optional prefix for plot titles (e.g. ``"Baseline"``).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    prefix = f"{title_prefix} – " if title_prefix else ""

    # ── Loss curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{prefix}Loss vs. Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # ── Accuracy curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{prefix}Accuracy vs. Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)

    # ── Combined (2×1 subplot) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{prefix}Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{prefix}Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {save_dir}/")
