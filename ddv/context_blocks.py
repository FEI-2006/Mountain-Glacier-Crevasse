"""Shape-preserving context blocks of DDV-DeepLabV3+.

DSConv  - Dynamic Snake Convolution (Qi et al., ICCV 2023)
D-LKA   - Deformable Large Kernel Attention (Azad et al., WACV 2024)
VMC     - VMamba-style Context: a pure-PyTorch directional state-space block
          inspired by VMamba (Liu et al., 2024); it is NOT the official VMamba.

All blocks map a (B, C, H, W) tensor to a tensor of the same shape and are applied
after ASPP in the DeepLabV3+ head.
"""

from __future__ import annotations

import torch
from torch import nn

from .dsconv import DSConv  # upstream ICCV 2023 implementation (MIT)


class DynamicSnakeContext(nn.Module):
    """Shape-preserving two-view Dynamic Snake Convolution context block.

    The horizontal and vertical branches follow the ICCV 2023 DSConv design.
    Their outputs are fused and added to the input so this block can replace a
    DeepLabV3+ context module without changing downstream tensor dimensions.
    """

    def __init__(self, channels: int, kernel_size: int = 5, extend_scope: float = 1.0):
        super().__init__()
        if channels < 4 or channels % 4 != 0:
            raise ValueError("channels must be divisible by 4 (DSConv uses GroupNorm(channels // 4))")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizontal = DSConv(channels, channels, kernel_size, extend_scope, 0, True, device)
        self.vertical = DSConv(channels, channels, kernel_size, extend_scope, 1, True, device)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The upstream DSConv stores a device attribute; keep it in sync with the input.
        self.horizontal.device = x.device
        self.vertical.device = x.device
        y = self.fuse(torch.cat([self.horizontal(x), self.vertical(x)], dim=1))
        return x + y


class _OffsetDeformConv2d(nn.Module):
    """Depthwise deformable convolution used by the WACV 2024 D-LKA block."""

    def __init__(self, channels: int, kernel_size: int, padding: int, dilation: int = 1,
                 offset_kernel_size: int = 3):
        super().__init__()
        offset_padding = offset_kernel_size // 2
        self.offset = nn.Conv2d(
            channels,
            2 * kernel_size * kernel_size,
            kernel_size=offset_kernel_size,
            padding=offset_padding,
            bias=True,
        )
        try:
            from torchvision.ops import DeformConv2d
        except Exception as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("D-LKA requires torchvision.ops.DeformConv2d") from exc
        self.deform = DeformConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deform(x, self.offset(x))


class DeformableLKAContext(nn.Module):
    """Shape-preserving Deformable Large Kernel Attention context block.

    Follows the 2-D deformable LKA block of Azad et al. (WACV 2024), with the
    deformable depthwise convolutions implemented by ``torchvision.ops.DeformConv2d``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.conv0 = _OffsetDeformConv2d(channels, kernel_size=5, padding=2)
        self.conv_spatial = _OffsetDeformConv2d(
            channels, kernel_size=7, padding=9, dilation=3
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.proj_1(x)
        y = self.activation(y)
        attention = self.conv1(self.conv_spatial(self.conv0(y)))
        y = self.proj_2(y * attention)
        return y + shortcut


class VMambaStyleContext(nn.Module):
    """Pure-PyTorch directional state-space context inspired by VMamba.

    The official VMamba needs ``mamba_ssm``/Triton CUDA kernels. This lightweight
    block keeps four horizontal/vertical scan directions and a learned diagonal
    state update in plain PyTorch, so it runs on any platform (including Windows).
    """

    def __init__(self, channels: int, hidden_channels: int = 64):
        super().__init__()
        hidden_channels = max(16, min(hidden_channels, channels))
        self.in_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.delta_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.a_log = nn.Parameter(torch.full((hidden_channels,), -2.0))
        self.skip = nn.Parameter(torch.ones(hidden_channels))
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def _scan(self, sequence: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply a stable diagonal state-space recurrence along the last axis."""
        a = -torch.exp(self.a_log).view(1, -1, 1)
        log_decay = (delta * a).clamp(min=-20.0, max=0.0)
        log_product = torch.cumsum(log_decay, dim=-1)
        bounded_product = torch.exp(log_product.clamp(min=-20.0, max=0.0))
        weighted_input = delta * sequence / bounded_product
        state = bounded_product * torch.cumsum(weighted_input, dim=-1)
        return state + self.skip.view(1, -1, 1) * sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        y = self.norm(self.in_proj(x))
        delta = torch.nn.functional.softplus(self.delta_proj(y)) + 1e-3

        row = y.flatten(2)
        row_delta = delta.flatten(2)
        row_forward = self._scan(row, row_delta)
        row_reverse = torch.flip(
            self._scan(torch.flip(row, dims=[-1]), torch.flip(row_delta, dims=[-1])),
            dims=[-1],
        )

        column = y.transpose(2, 3).flatten(2)
        column_delta = delta.transpose(2, 3).flatten(2)
        column_forward = self._scan(column, column_delta)
        column_reverse = torch.flip(
            self._scan(torch.flip(column, dims=[-1]), torch.flip(column_delta, dims=[-1])),
            dims=[-1],
        )

        b, _, h, w = y.shape
        row_context = (row_forward + row_reverse).view(b, -1, h, w)
        column_context = (column_forward + column_reverse).view(b, -1, w, h).transpose(2, 3)
        context = 0.25 * (row_context + column_context)
        return shortcut + self.out_proj(context)


__all__ = ["DynamicSnakeContext", "DeformableLKAContext", "VMambaStyleContext"]
