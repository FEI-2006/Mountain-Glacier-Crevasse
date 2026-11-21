"""Build DDV-DeepLabV3+ and run one forward/backward pass on random input (CPU or GPU)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddv import build_model  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, output_stride=16, pretrained_backbone=False).to(device)
    x = torch.randn(2, 3, 256, 256, device=device)
    y = model(x)
    assert y.shape == (2, 2, 256, 256), y.shape
    y.mean().backward()
    blocks = [type(m).__name__ for m in model.classifier.context]
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"context after ASPP: {' -> '.join(blocks)}")
    print(f"output {tuple(y.shape)}, {params:.2f} M parameters, device {device}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
