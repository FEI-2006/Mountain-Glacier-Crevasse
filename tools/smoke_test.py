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


def check_dsconv_offset_chain(device):
    """Offsets accumulate from the kernel centre out to both ends, and every
    non-centre offset channel receives a gradient.

    With all offsets set to 0.5 and kernel size 5, the five sampling points must be
    shifted by 1.0, 0.5, 0, 0.5 and 1.0 (centre fixed, then accumulated outwards).
    """
    k, width, height = 5, 4, 3
    expected = torch.tensor([1.0, 0.5, 0.0, 0.5, 1.0])
    others = torch.arange(k) != k // 2
    for morph in (0, 1):
        offset = torch.full((2, 2 * k, width, height), 0.5, device=device, requires_grad=True)
        dsc = DSC(offset.shape, kernel_size=k, extend_scope=1, morph=morph, device=device)
        y, x = dsc._coordinate_map_3D(offset, True)
        if morph == 0:  # vertical swing of a horizontal kernel
            moved = y
            shift = y.reshape(2, width, k, height) - torch.arange(width, device=device).view(1, -1, 1, 1)
            got = shift[0, 0, :, 0]
        else:  # horizontal swing of a vertical kernel
            moved = x
            shift = x.reshape(2, width, height, k) - torch.arange(height, device=device).view(1, 1, -1, 1)
            got = shift[0, 0, 0, :]
        assert torch.allclose(got.detach().cpu(), expected), f"morph {morph}: offsets {got.tolist()}"
        moved.sum().backward()
        grad = offset.grad[:, :k] if morph == 0 else offset.grad[:, k:]
        grad = grad.abs().sum(dim=(0, 2, 3)).cpu()
        assert (grad[others] > 0).all(), f"morph {morph}: offset channels without gradient {grad.tolist()}"
    print("DSConv offsets: accumulated to both kernel ends, all offset channels receive gradients.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_dsconv_batch_indexing(device)
    check_dsconv_offset_chain(device)
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
