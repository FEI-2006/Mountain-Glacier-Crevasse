"""Predict crevasse masks with a trained DDV-DeepLabV3+ checkpoint.

Every 8-bit RGB image in --input (PNG/TIF/JPG/BMP/WebP) is predicted at its native size.
Output: <output>/masks/*.png (0 = background, 255 = crevasse) and
        <output>/overlays/*.png (crevasses drawn in red on the image).

Example:
    python predict.py --input tiles/ --checkpoint runs/<run>/best.pth --output predictions/
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ddv import build_model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_rgb(path):
    with Image.open(path) as image:
        bits = getattr(image, "tag_v2", {}).get(258, ())
        bits = (bits,) if isinstance(bits, int) else bits
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P"} or any(b != 8 for b in bits):
            raise ValueError(f"{path}: expected an 8-bit RGB image (got mode {image.mode}, bits {bits})")
        return np.asarray(image.convert("RGB")).copy()


def to_tensor(images, device):
    x = (np.stack(images).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(device)


def red_overlay(rgb, mask):
    out = rgb.copy()
    out[mask] = (0.18 * rgb[mask] + 0.82 * np.array([255, 20, 20])).astype(np.uint8)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="folder with RGB image tiles")
    p.add_argument("--checkpoint", type=Path, required=True, help="trained checkpoint (best.pth)")
    p.add_argument("--output", type=Path, required=True, help="output folder")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-stride", type=int, default=16, choices=(8, 16))
    args = p.parse_args()

    paths = sorted(q for q in args.input.iterdir() if q.suffix.lower() in EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.input}")
    stems = [q.stem for q in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Input images must have unique file names (ignoring extension)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    model = build_model(num_classes=2, output_stride=args.output_stride, pretrained_backbone=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state", state), strict=True)
    model.to(device).eval()

    (args.output / "masks").mkdir(parents=True, exist_ok=True)
    (args.output / "overlays").mkdir(parents=True, exist_ok=True)

    by_size = defaultdict(list)  # images in one batch must share a size
    for q in paths:
        with Image.open(q) as image:
            by_size[image.size].append(q)

    done = 0
    for group in by_size.values():
        for start in range(0, len(group), args.batch_size):
            batch = group[start:start + args.batch_size]
            images = [read_rgb(q) for q in batch]
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                                                        enabled=amp_dtype is not None):
                masks = model(to_tensor(images, device)).argmax(1).cpu().numpy().astype(bool)
            for q, rgb, mask in zip(batch, images, masks):
                Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / "masks" / f"{q.stem}.png")
                Image.fromarray(red_overlay(rgb, mask)).save(args.output / "overlays" / f"{q.stem}.png")
            done += len(batch)
            print(f"{done}/{len(paths)}", flush=True)
    print(f"Saved masks and overlays to {args.output}")


if __name__ == "__main__":
    main()
