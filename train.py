"""Unified training script: ResNet-18 on CIFAR-10 with selectable regularization.

This script lets you mix and match regularization techniques through CLI flags.
It subsumes ``train_baseline.py`` (which uses no regularization flags) and adds:

  - **L2 weight decay** (``--weight-decay``, default 5e-4)
  - **Dropout** (``--dropout``)
  - **Label smoothing** (``--label-smoothing``)
  - **Mixup** (``--mixup-alpha``)
  - **Cutout** (``--cutout-length``)

Usage
-----
    # Baseline (no extra regularization)
    python train.py

    # Dropout + label smoothing
    python train.py --dropout 0.3 --label-smoothing 0.1

    # Mixup + Cutout
    python train.py --mixup-alpha 0.2 --cutout-length 16

    # Everything together
    python train.py --dropout 0.3 --label-smoothing 0.1 --mixup-alpha 0.2 --cutout-length 16 --plot
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from models.resnet import resnet18
from utils import (
    get_cifar10_loaders,
    evaluate_loss_and_accuracy,
    mixup_data,
    mixup_criterion,
    plot_training_curves,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="ResNet-18 on CIFAR-10 with selectable regularization"
    )

    # ── Data ────────────────────────────────────────────────────────────
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)

    # ── Optimizer ───────────────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4, help="L2 weight decay")

    # ── LR schedule ────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr-schedule", choices=["cosine", "step"], default="cosine")
    p.add_argument("--lr-milestones", nargs="+", type=int, default=[100, 150])
    p.add_argument("--lr-gamma", type=float, default=0.1)

    # ── Regularization ─────────────────────────────────────────────────
    p.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout probability (0 = off). Applied before the final pooling layer.",
    )
    p.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (0 = off).",
    )
    p.add_argument(
        "--mixup-alpha", type=float, default=0.0,
        help="Mixup interpolation strength (0 = off). Typical value: 0.2–1.0.",
    )
    p.add_argument(
        "--cutout-length", type=int, default=0,
        help="Cutout mask side length in pixels (0 = off). Typical value for CIFAR-10: 16.",
    )

    # ── Misc ───────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")

    # ── Visualization ──────────────────────────────────────────────────
    p.add_argument("--plot", action="store_true", help="Save training curve plots")
    p.add_argument("--plot-dir", type=str, default="./plots")

    return p.parse_args()


# ── Training loops ───────────────────────────────────────────────────────────

def train_one_epoch_standard(model, loader, criterion, optimizer, device):
    """Standard training loop (no Mixup)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def train_one_epoch_mixup(model, loader, criterion, optimizer, device, alpha):
    """Training loop with Mixup data augmentation.

    Note: training accuracy is approximate because mixed examples do not have
    a single true label.  Validation accuracy (computed on clean data) is the
    reliable metric.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(y_a).float()
                     + (1 - lam) * predicted.eq(y_b).float()).sum().item()

    return running_loss / total, 100.0 * correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Summarize config
    reg_flags = []
    if args.weight_decay > 0:
        reg_flags.append(f"L2={args.weight_decay}")
    if args.dropout > 0:
        reg_flags.append(f"Dropout={args.dropout}")
    if args.label_smoothing > 0:
        reg_flags.append(f"LabelSmooth={args.label_smoothing}")
    if args.mixup_alpha > 0:
        reg_flags.append(f"Mixup(α={args.mixup_alpha})")
    if args.cutout_length > 0:
        reg_flags.append(f"Cutout({args.cutout_length})")
    reg_str = ", ".join(reg_flags) if reg_flags else "None (baseline)"
    print(f"Device: {device}  |  Regularization: {reg_str}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cutout_length=args.cutout_length,
    )

    # Model (with optional dropout)
    model = resnet18(num_classes=10, dropout_rate=args.dropout).to(device)

    # Loss (with optional label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # LR schedule
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    use_mixup = args.mixup_alpha > 0
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        if use_mixup:
            train_loss, train_acc = train_one_epoch_mixup(
                model, train_loader, criterion, optimizer, device, args.mixup_alpha
            )
        else:
            train_loss, train_acc = train_one_epoch_standard(
                model, train_loader, criterion, optimizer, device
            )

        val_loss, val_acc = evaluate_loss_and_accuracy(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.2f}%  {val_loss:8.4f}  {val_acc:6.2f}%  {current_lr:.1e}  {elapsed:5.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint_dir, "best.pt"),
            )

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")

    if args.plot:
        plot_training_curves(history, save_dir=args.plot_dir, title_prefix=reg_str)


if __name__ == "__main__":
    main()
