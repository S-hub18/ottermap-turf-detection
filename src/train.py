"""
Trains a U-Net (segmentation_models_pytorch) with a pretrained encoder for
binary turf segmentation.

Given only 3 source images (~240 tiles after tiling), the encoder is kept
mostly frozen for the first warmup epochs and trained at a much lower LR
than the decoder afterward — full end-to-end fine-tuning at a uniform LR on
this little data would overfit the encoder's general-purpose features to
this dataset's specific look, which is exactly what hurts transfer to the
evaluation imagery.

Loss: Dice + BCE. Dice handles the class imbalance from background-heavy
tiles; BCE keeps per-pixel gradients well-behaved early in training.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp

from dataset import TurfDataset, get_train_transform, get_val_transform


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def iou_score(pred, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = ((pred + target) > 0).float().sum(dim=(2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def build_optimizer(model, args, phase):
    if phase == "warmup":
        for p in model.encoder.parameters():
            p.requires_grad = False
        # Include segmentation_head alongside decoder — it starts from random
        # init and needs gradient signal during warmup too.
        return torch.optim.Adam(
            list(model.decoder.parameters()) + list(model.segmentation_head.parameters()),
            lr=args.lr,
        )

    for p in model.encoder.parameters():
        p.requires_grad = True
    return torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
        {"params": model.decoder.parameters(), "lr": args.lr},
        {"params": model.segmentation_head.parameters(), "lr": args.lr},
    ])


def build_scheduler(optimizer, phase_epochs):
    return CosineAnnealingLR(optimizer, T_max=max(phase_epochs, 1), eta_min=0.0)


def get_lr(optimizer):
    return max(group["lr"] for group in optimizer.param_groups)


def evaluate(model, loader, device, thresh=0.5):
    model.eval()
    total_inter = 0.0
    total_pred_sum = 0.0
    total_target_sum = 0.0
    tp = fp = fn = 0.0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            pred_bin = probs > thresh
            target_bin = masks > 0.5

            tp += (pred_bin & target_bin).sum().item()
            fp += (pred_bin & ~target_bin).sum().item()
            fn += (~pred_bin & target_bin).sum().item()

            total_inter += (probs * masks).sum().item()
            total_pred_sum += probs.sum().item()
            total_target_sum += masks.sum().item()

    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * total_inter + eps) / (total_pred_sum + total_target_sum + eps)
    return {
        "val_iou": iou,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_dice": dice,
    }


def main(args):
    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M-series) GPU
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    train_ds = TurfDataset(args.manifest, split="train", transform=get_train_transform(args.img_size))
    val_ds = TurfDataset(args.manifest, split="val", transform=get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(f"Train tiles: {len(train_ds)}  Val tiles: {len(val_ds)}")

    bce = nn.BCEWithLogitsLoss()

    def loss_fn(pred, target):
        return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)

    best_mIoU = 0.0
    history = []
    os.makedirs(os.path.dirname(args.out_weights), exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_json), exist_ok=True)

    current_phase = "warmup" if args.warmup_epochs > 0 else "finetune"
    current_phase_epochs = args.warmup_epochs if current_phase == "warmup" else args.epochs
    optimizer = build_optimizer(model, args, current_phase)
    scheduler = build_scheduler(optimizer, current_phase_epochs)

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs < args.epochs:
            current_phase = "finetune"
            current_phase_epochs = args.epochs - args.warmup_epochs
            optimizer = build_optimizer(model, args, current_phase)
            scheduler = build_scheduler(optimizer, current_phase_epochs)

        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        metrics = evaluate(model, val_loader, device)
        current_lr = get_lr(optimizer)
        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_mIoU={metrics['val_iou']:.4f}  "
            f"[Per-Class (Turf)] precision={metrics['val_precision']:.4f}  "
            f"recall={metrics['val_recall']:.4f}  "
            f"f1={metrics['val_f1']:.4f}  "
            f"val_dice={metrics['val_dice']:.4f}  "
            f"lr={current_lr:.3e}"
        )

        if metrics["val_iou"] >= best_mIoU:
            best_mIoU = metrics["val_iou"]
            torch.save({
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "epoch": epoch,
                "phase": current_phase,
                "train_loss": train_loss,
                "val_mIoU": metrics["val_iou"],
                "val_precision": metrics["val_precision"],
                "val_recall": metrics["val_recall"],
                "val_f1": metrics["val_f1"],
                "val_dice": metrics["val_dice"],
                "lr": current_lr,
            }, args.out_weights)
            print(f"  -> saved new best checkpoint (val_mIoU={metrics['val_iou']:.4f})")

        history.append({
            "epoch": epoch + 1,
            "phase": current_phase,
            "train_loss": train_loss,
            "val_mIoU": metrics["val_iou"],
            "val_precision": metrics["val_precision"],
            "val_recall": metrics["val_recall"],
            "val_f1": metrics["val_f1"],
            "val_dice": metrics["val_dice"],
            "lr": current_lr,
        })

        scheduler.step()

    with open(args.metrics_json, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val IoU: {best_iou:.4f}. Weights: {args.out_weights}")
    print(f"Metrics JSON: {args.metrics_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="../data/tiles/manifest.json")
    ap.add_argument("--out_weights", default="../weights/turf_unet_resnet34.pt")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--metrics_json", default="../results/train_metrics.json")
    ap.add_argument("--resume", type=str, default="", help="Path to existing weights to resume from")
    args = ap.parse_args()
    main(args)
