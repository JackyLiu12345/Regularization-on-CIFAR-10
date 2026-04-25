"""Training with Label Smoothing regularization on CIFAR-10/100.

Label smoothing replaces hard 0/1 targets with soft targets, preventing the
model from becoming overconfident.  It was introduced by Szegedy et al. in
"Rethinking the Inception Architecture" (2016).

Usage
-----
    python train_label_smoothing.py                           # smoothing=0.1
    python train_label_smoothing.py --label-smoothing 0.2
    python train_label_smoothing.py --dataset cifar100 --label-smoothing 0.1 --plot
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from models.resnet import resnet18
from utils import (
    get_data_loaders,
    num_classes_for,
    evaluate_loss_and_accuracy,
    plot_training_curves,
)


def parse_args():
    p = argparse.ArgumentParser(description="Label smoothing on CIFAR-10/100")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                   help="Dataset to train on (default: cifar10)")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr-schedule", choices=["cosine", "step"], default="cosine")
    p.add_argument("--lr-milestones", nargs="+", type=int, default=[100, 150])
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label smoothing factor (default 0.1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-dir", type=str, default="./plots")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Label smoothing: {args.label_smoothing}")

    n_classes = num_classes_for(args.dataset)

    train_loader, test_loader = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers
    )

    model = resnet18(num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}  {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loss_and_accuracy(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:8.2f}%  {val_loss:8.4f}  {val_acc:6.2f}%  {current_lr:.1e}  {elapsed:5.1f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "val_acc": val_acc},
                os.path.join(args.checkpoint_dir, "resnet18_label_smoothing_best.pt"),
            )

    print(f"\nBest val accuracy: {best_val_acc:.2f}%")
    if args.plot:
        plot_training_curves(history, save_dir=args.plot_dir,
                             title_prefix=f"Label Smoothing({args.label_smoothing})")


if __name__ == "__main__":
    main()
