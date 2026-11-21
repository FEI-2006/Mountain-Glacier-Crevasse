"""Train DDV-DeepLabV3+ for crevasse segmentation.

Dataset layout (512x512 8-bit RGB PNG tiles, masks with the same file names):

    <data>/train/images/*.png   <data>/train/masks/*.png
    <data>/val/images/*.png     <data>/val/masks/*.png

Mask pixels: 0 = background, >0 = crevasse.

Example:
    python train.py --data /path/to/dataset --output runs
"""
import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ddv import build_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- data
class CrevasseTiles(Dataset):
    """Paired RGB tiles and binary masks; ImageNet normalisation, no augmentation."""

    def __init__(self, root, split):
        self.images = sorted((root / split / "images").glob("*.png"))
        self.masks = [root / split / "masks" / p.name for p in self.images]
        if not self.images or any(not p.is_file() for p in self.masks):
            raise ValueError(f"Missing or unpaired PNG tiles in {root / split}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as src:
            image = np.array(src.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(self.masks[index]) as src:
            mask = np.array(src)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if image.shape != (512, 512, 3) or mask.shape != (512, 512):
            raise ValueError(f"Expected 512x512 tiles: {self.images[index]}")
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        image = (image - image.new_tensor(MEAN)[:, None, None]) / image.new_tensor(STD)[:, None, None]
        return image, torch.from_numpy((mask > 0).astype(np.int64))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------- loss
class WeightedCEDiceLoss(nn.Module):
    """0.5 * class-weighted cross-entropy (background 0.25, crevasse 1.0) + 0.5 * soft Dice loss."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.tensor([0.25, 1.0]))

    def forward(self, logits, target):
        logits = logits.float()  # loss is computed in FP32
        ce = F.cross_entropy(logits, target, weight=self.weights)
        fg = logits.softmax(1)[:, 1]
        truth = (target == 1).float()
        dice = (2 * (fg * truth).sum((1, 2)) + 1) / (fg.sum((1, 2)) + truth.sum((1, 2)) + 1)
        return 0.5 * ce + 0.5 * (1 - dice.mean())


# ----------------------------------------------------------------------------- schedule / metrics
def lr_factor(step, total_steps, warmup_steps):
    """Linear warm-up from 0.1x, then polynomial decay with power 0.9."""
    if warmup_steps and step < warmup_steps:
        return 0.1 + 0.9 * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress) ** 0.9


def scores(cm):
    tn, fp, fn, tp = (int(v) for v in cm.flatten())
    ratio = lambda a, b: a / b if b else 0.0  # noqa: E731
    iou = ratio(tp, tp + fp + fn)
    bg_iou = ratio(tn, tn + fp + fn)
    return dict(crevasse_iou=iou, background_iou=bg_iou, mean_iou=(iou + bg_iou) / 2,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                dice=ratio(2 * tp, 2 * tp + fp + fn))


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    cm = torch.zeros((2, 2), dtype=torch.int64, device=device)
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            preds = model(images).argmax(1)
        cm += torch.bincount((2 * masks + preds).flatten(), minlength=4).reshape(2, 2)
    return scores(cm.cpu().numpy())


def save(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="dataset root with train/ and val/")
    p.add_argument("--output", type=Path, required=True, help="folder for run directories")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-lr", type=float, default=0.01)
    p.add_argument("--backbone-lr", type=float, default=0.001)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--output-stride", type=int, default=16, choices=(8, 16))
    p.add_argument("--precision", default="bf16", choices=("bf16", "fp32"),
                   help="bf16 mixed precision (falls back to fp32 if unsupported); fp16 overflows")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", type=Path, default=None, help="latest.pth of an interrupted run")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16 if args.precision == "bf16" and device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else None)

    train_set = CrevasseTiles(args.data, "train")
    val_set = CrevasseTiles(args.data, "val")
    generator = torch.Generator().manual_seed(args.seed)
    loader_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              generator=generator, **loader_kw)
    val_loader = DataLoader(val_set, batch_size=min(8, args.batch_size), shuffle=False, **loader_kw)

    model = build_model(num_classes=2, output_stride=args.output_stride, pretrained_backbone=True).to(device)
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.01
    optimizer = torch.optim.SGD([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr}],
        momentum=0.9, weight_decay=1e-4)
    criterion = WeightedCEDiceLoss().to(device)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    start_epoch, step, best_iou, best_epoch = 1, 0, -1.0, 0

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch, step = state["epoch"] + 1, state["cur_itrs"]
        best_iou, best_epoch = state["best_score"], state["best_epoch"]
        if "train_generator_state" in state:
            random.setstate(state["python_random_state"])
            np.random.set_state(state["numpy_random_state"])
            torch.set_rng_state(state["torch_rng_state"])
            generator.set_state(state["train_generator_state"])
        run_dir = args.resume.parent
    else:
        run_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    print(f"Run: {run_dir} | train {len(train_set)} / val {len(val_set)} tiles | "
          f"{'bf16' if amp_dtype else 'fp32'} on {device}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, masks in train_loader:
            factor = lr_factor(step, total_steps, warmup_steps)
            for group, base_lr in zip(optimizer.param_groups, (args.backbone_lr, args.head_lr)):
                group["lr"] = base_lr * factor
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(images)
            loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradients at epoch {epoch}, step {step}")
            optimizer.step()
            step += 1
            loss_sum += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch}/{args.epochs}  step {step}/{total_steps}  loss {loss.item():.4f}", flush=True)

        metrics = evaluate(model, val_loader, device, amp_dtype)
        improved = metrics["crevasse_iou"] > best_iou
        if improved:
            best_iou, best_epoch = metrics["crevasse_iou"], epoch
        row = dict(epoch=epoch, step=step, train_loss=loss_sum / steps_per_epoch, **metrics)
        log = run_dir / "metrics.csv"
        new_file = not log.exists()
        with log.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        state = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                     epoch=epoch, cur_itrs=step, best_score=best_iou, best_epoch=best_epoch,
                     metrics=metrics, python_random_state=random.getstate(),
                     numpy_random_state=np.random.get_state(), torch_rng_state=torch.get_rng_state(),
                     train_generator_state=generator.get_state())
        save(run_dir / "latest.pth", state)
        if improved:
            save(run_dir / "best.pth", state)
        print(f"epoch {epoch}: val crevasse IoU {metrics['crevasse_iou']:.4f} "
              f"(best {best_iou:.4f} @ epoch {best_epoch})", flush=True)

    print(f"Done. Best model: {run_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
