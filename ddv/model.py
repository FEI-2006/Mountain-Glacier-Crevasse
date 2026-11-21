"""DDV-DeepLabV3+: DeepLabV3+ (ResNet-50) with DSConv, D-LKA and VMC after ASPP.

    image ─► ResNet-50 ─► ASPP ─► DSConv ─► D-LKA ─► VMC ─┐
                │                                         ├─► decoder ─► logits
                └──────────── low-level features ─────────┘

The DeepLabV3+ parts follow VainF/DeepLabV3Plus-Pytorch (MIT); parameter names are
kept so that checkpoints trained with that code base load with ``strict=True``.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

from .context_blocks import DeformableLKAContext, DynamicSnakeContext, VMambaStyleContext

# ImageNet ResNet-50 weights used for training.
RESNET50_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


class ResNet50Features(nn.ModuleDict):
    """ResNet-50 up to layer4; returns ``low_level`` (layer1) and ``out`` (layer4) features."""

    def __init__(self, resnet):
        super().__init__(OrderedDict(
            (name, getattr(resnet, name))
            for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")))

    def forward(self, x):
        features = {}
        for name, module in self.items():
            x = module(x)
            if name == "layer1":
                features["low_level"] = x
        features["out"] = x
        return features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(super().forward(x), size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        r1, r2, r3 = atrous_rates
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            ASPPConv(in_channels, out_channels, r1),
            ASPPConv(in_channels, out_channels, r2),
            ASPPConv(in_channels, out_channels, r3),
            ASPPPooling(in_channels, out_channels),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))


class DDVHead(nn.Module):
    """DeepLabV3+ decoder head with the DDV context chain (DSConv -> D-LKA -> VMC) after ASPP."""

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.context = nn.Sequential(
            DynamicSnakeContext(256),
            DeformableLKAContext(256),
            VMambaStyleContext(256))
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, features):
        low_level = self.project(features["low_level"])
        x = self.context(self.aspp(features["out"]))
        x = F.interpolate(x, size=low_level.shape[2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([low_level, x], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DDVDeepLabV3Plus(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(self.backbone(x))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(num_classes=2, output_stride=16, pretrained_backbone=True):
    """Build DDV-DeepLabV3+ with a ResNet-50 backbone.

    Args:
        num_classes: 2 for background / crevasse.
        output_stride: 16 (used in this work) or 8.
        pretrained_backbone: load ImageNet ResNet-50 weights (for training).
    """
    if output_stride == 16:
        dilation, aspp_dilate = [False, False, True], [6, 12, 18]
    elif output_stride == 8:
        dilation, aspp_dilate = [False, True, True], [12, 24, 36]
    else:
        raise ValueError("output_stride must be 8 or 16")
    resnet = resnet50(weights=None, replace_stride_with_dilation=dilation)
    if pretrained_backbone:
        resnet.load_state_dict(load_state_dict_from_url(RESNET50_URL, progress=True))
    head = DDVHead(2048, 256, num_classes, aspp_dilate)
    return DDVDeepLabV3Plus(ResNet50Features(resnet), head)
