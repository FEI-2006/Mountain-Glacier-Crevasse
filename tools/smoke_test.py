"""Build DDV-DeepLabV3+ and run one forward/backward pass on random input (CPU or GPU)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddv import build_model  # noqa: E402
from ddv.dsconv import DSC  # noqa: E402


def check_dsconv_batch_indexing(device):
    """Each image in a batch must be sampled from its own feature map.

    Two constant feature maps (1 and 7) are sampled on the integer grid with zero offset,
    so the output must reproduce each input exactly.
    """
    channels, width, height = 3, 6, 5
    feat = torch.stack([
        torch.full((channels, width, height), 1.0),
        torch.full((channels, width, height), 7.0),
    ]).to(device)
    yy, xx = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="ij")
    y = yy.expand(2, width, height).float().to(device)
    x = xx.expand(2, width, height).float().to(device)
    dsc = DSC(feat.shape, kernel_size=1, extend_scope=1, morph=0, device=device)
    out = dsc._bilinear_interpolate_3D(feat, y, x)
    assert torch.allclose(out[0], feat[0]), "sample 0 did not read its own features"
    assert torch.allclose(out[1], feat[1]), "sample 1 read features from another sample"
    print("DSConv batch indexing: each sample reads its own features.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_dsconv_batch_indexing(device)
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
