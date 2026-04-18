"""Baseline training script: ResNet-18 + SGD on CIFAR-10.

Usage
-----
    python train_baseline.py [OPTIONS]

Key defaults (standard CIFAR-10 baseline):
    --epochs 200  --lr 0.1  --momentum 0.9  --weight-decay 5e-4
    --batch-size 128  --lr-schedule cosine

The script prints per-epoch train loss, train accuracy, and test accuracy,
and saves the best model checkpoint to ``checkpoints/``.
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from models.resnet import resnet18
from utils import get_cifar10_loaders, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline: ResNet-18 + SGD on CIFAR-10")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    # Optimiser
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="L2 weight decay")

    # Schedule
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "step"],
        help="Learning-rate schedule (cosine annealing or step decay)",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        type=int,
        default=[100, 150],
        help="Epochs at which to decay LR (only used with --lr-schedule step)",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="LR decay factor for step schedule",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Directory for saved models"
    )

    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return (average loss, accuracy %)."""
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

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = resnet18(num_classes=10).to(device)

    # Loss & optimiser
    criterion = nn.CrossEntropyLoss()
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

    best_test_acc = 0.0
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Test Acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = compute_accuracy(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.2f}%  {test_acc:7.2f}%  {current_lr:.1e}  {elapsed:5.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                },
                os.path.join(args.checkpoint_dir, "resnet18_baseline_best.pt"),
            )

    print(f"\nBest test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
