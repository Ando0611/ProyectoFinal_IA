"""Loop de entrenamiento U-Net multi-tarea sobre CelebAMask-HQ.

Uso:
    python -m src.train
    python -m src.train --epochs 30 --batch-size 4 --lambda-attr 0.5
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import (
    ATTRS,
    BATCH_SIZE,
    IMAGE_SIZE,
    LAMBDA_ATTR,
    LOGS_DIR,
    LR,
    MODELS_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    SEED,
    SEG_CLASSES,
    TRAIN_RATIO,
    WEIGHT_DECAY,
)
from src.dataset import CelebAMaskHQDataset, split_ids
from src.losses import MultiTaskLoss
from src.metrics import AttrMetric, SegMetric
from src.model import UNetMultiTask, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena U-Net multi-tarea")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-attr", type=float, default=LAMBDA_ATTR)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--output", type=str, default=str(MODELS_DIR / "unet_multitask_best.pt"),
    )
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)

    seg_metric = SegMetric()
    attr_metric = AttrMetric()

    total_loss = 0.0
    total_seg = 0.0
    total_attr = 0.0
    n_batches = 0

    desc = "train" if train else "valid"
    bar = tqdm(loader, desc=desc, leave=False)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for images, masks, attrs in bar:
            images = images.to(device)
            masks = masks.to(device)
            attrs = attrs.to(device)

            seg_logits, attr_logits = model(images)
            loss, l_seg, l_attr = criterion(seg_logits, attr_logits, masks, attrs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_seg += l_seg.item()
            total_attr += l_attr.item()
            n_batches += 1

            pred_mask = seg_logits.argmax(dim=1)
            seg_metric.update(pred_mask.detach(), masks.detach())
            attr_metric.update(attr_logits.detach(), attrs.detach())

            bar.set_postfix(
                loss=f"{loss.item():.3f}",
                seg=f"{l_seg.item():.3f}",
                attr=f"{l_attr.item():.3f}",
            )

    per_class_iou, mean_iou = seg_metric.compute_iou()
    attr_report = attr_metric.compute()

    return {
        "loss": total_loss / max(n_batches, 1),
        "seg_loss": total_seg / max(n_batches, 1),
        "attr_loss": total_attr / max(n_batches, 1),
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "attr_report": attr_report,
    }


def format_attr_report(report):
    lines = []
    for name in ATTRS:
        r = report[name]
        lines.append(f"    {name:11s} acc={r['acc']:.3f} f1={r['f1']:.3f}")
    mean = report["_mean"]
    lines.append(f"    {'_mean':11s} acc={mean['acc']:.3f} f1={mean['f1']:.3f}")
    return "\n".join(lines)


def main():
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    # 1) Datasets y split
    full = CelebAMaskHQDataset(image_size=args.image_size, augment=False)
    train_ids, valid_ids = split_ids(full.ids, args.train_ratio, args.seed)

    train_ds = CelebAMaskHQDataset(
        image_size=args.image_size, ids=train_ids, augment=True,
    )
    valid_ds = CelebAMaskHQDataset(
        image_size=args.image_size, ids=valid_ids, augment=False,
    )

    print(f"Train: {len(train_ds)} | Valid: {len(valid_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    # 2) Modelo, loss, optimizador
    model = UNetMultiTask(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params/1e6:.2f}M")

    criterion = MultiTaskLoss(lambda_attr=args.lambda_attr)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # 3) Loop
    history = []
    best_iou = -1.0
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        train_stats = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
        )
        valid_stats = run_epoch(
            model, valid_loader, criterion, optimizer, device, train=False,
        )

        print(f"TRAIN  loss={train_stats['loss']:.4f}  "
              f"seg={train_stats['seg_loss']:.4f}  "
              f"attr={train_stats['attr_loss']:.4f}  "
              f"mIoU={train_stats['mean_iou']:.4f}")
        print(f"VALID  loss={valid_stats['loss']:.4f}  "
              f"seg={valid_stats['seg_loss']:.4f}  "
              f"attr={valid_stats['attr_loss']:.4f}  "
              f"mIoU={valid_stats['mean_iou']:.4f}")
        print("  Atributos (valid):")
        print(format_attr_report(valid_stats["attr_report"]))

        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in train_stats.items() if k != "per_class_iou"},
            "valid": {k: v for k, v in valid_stats.items() if k != "per_class_iou"},
        })

        if valid_stats["mean_iou"] > best_iou:
            best_iou = valid_stats["mean_iou"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "mean_iou": valid_stats["mean_iou"],
                "attr_report": valid_stats["attr_report"],
                "config": {
                    "image_size": args.image_size,
                    "base_channels": args.base_channels,
                    "seg_classes": SEG_CLASSES,
                    "attrs": ATTRS,
                },
            }, output_path)
            print(f"  Checkpoint guardado: {output_path} (mIoU={best_iou:.4f})")

    history_path = LOGS_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistorial: {history_path}")
    print(f"Mejor checkpoint: {output_path}")


if __name__ == "__main__":
    main()
