# DDV-DeepLabV3+ for Mountain Glacier Crevasse Segmentation

PyTorch code of **DDV-DeepLabV3+** for mapping crevasses on mountain glaciers from
high-resolution optical imagery (e.g. aerial orthophotos).

DDV-DeepLabV3+ extends DeepLabV3+ (ResNet-50) with three context modules placed in sequence
after ASPP: **D**SConv, **D**-LKA and **V**MC.

## Model

```
image ─► ResNet-50 ─► ASPP ─► DSConv ─► D-LKA ─► VMC ─┐
            │                                         ├─► decoder ─► crevasse mask
            └──────────── low-level features ─────────┘
```

| Module | Role |
|---|---|
| **DSConv** – Dynamic Snake Convolution (Qi et al., ICCV 2023) | follows thin, curved crevasse lines; horizontal and vertical snake branches, fused and added back to the input |
| **D-LKA** – Deformable Large Kernel Attention (Azad et al., WACV 2024) | large, deformable receptive field for long and irregular crevasses |
| **VMC** – VMamba-style Context | four-direction (left/right/up/down) selective state-space scan for long-range context |

Each module keeps the 256-channel feature shape and has a residual connection.

`ddv/dsconv.py` fixes two errors in the upstream DSConv code:

1. **Offset chain.** `DSC._coordinate_map_3D` accumulated the snake offsets with
   `range(1, center)`, one step short, so the two outermost kernel points kept their raw,
   detached offsets: they were not chained to their neighbours and received no gradient.
   The loop now runs to both ends of the kernel.
2. **Batch indexing.** In `DSC._bilinear_interpolate_3D` the gather indices subtracted the
   per-sample offset, so every image in a batch read its sampled neighbours from the first
   image of the batch. The indices now keep that offset, and each image samples only its
   own features.
VMC is our own pure-PyTorch block inspired by VMamba (Liu et al., 2024); it is not the
official VMamba code and needs no `mamba_ssm`/Triton kernels, so it also runs on Windows.

## Files

```
ddv/model.py           DDV-DeepLabV3+ (ResNet-50 backbone, ASPP, DDV context chain, decoder)
ddv/context_blocks.py  DSConv, D-LKA and VMC blocks
ddv/dsconv.py          Dynamic Snake Convolution (upstream implementation, two fixes)
train.py               training
predict.py             prediction on image tiles
tools/smoke_test.py    checks the two DSConv fixes, builds the model and runs one forward/backward pass
```

## Installation

```bash
git clone https://github.com/FEI-2006/Mountain-Glacier-Crevasse.git
cd Mountain-Glacier-Crevasse
pip install -r requirements.txt   # install the CUDA build of PyTorch for your GPU first
python tools/smoke_test.py
```

Developed with Python 3.9, PyTorch 2.8 and CUDA 12.8.

## Data

```
dataset/
  train/images/*.png   512×512 8-bit RGB tiles
  train/masks/*.png    same file names; 0 = background, >0 = crevasse
  val/images/*.png
  val/masks/*.png
```

The dataset is not included in this repository.

## Training

```bash
python train.py --data /path/to/dataset --output runs
```

Default settings: ImageNet-pretrained ResNet-50, output stride 16, 100 epochs, batch size 16,
SGD (momentum 0.9, weight decay 1e-4), learning rate 0.01 (head) / 0.001 (backbone),
5 warm-up epochs then polynomial decay (power 0.9), gradient clipping 5.0, BF16 mixed precision,
loss = 0.5 × weighted cross-entropy (background 0.25, crevasse 1.0) + 0.5 × soft Dice.
Inputs are normalised with ImageNet mean/std; no augmentation is used.
The checkpoint with the highest validation crevasse IoU is saved as `best.pth`.
Run `python train.py -h` for all options; `--resume runs/<run>/latest.pth` continues an interrupted run.

## Prediction

```bash
python predict.py --input /path/to/tiles --checkpoint runs/<run>/best.pth --output predictions
```

Writes `masks/` (0/255 PNG) and `overlays/` (crevasses in red). Images are predicted at their
native size; georeferencing of GeoTIFF inputs is not copied to the PNG outputs.

## Acknowledgements

[VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) ·
[YaoleiQi/DSCNet](https://github.com/YaoleiQi/DSCNet) ·
[xmindflow/deformableLKA](https://github.com/xmindflow/deformableLKA) ·
[MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba). See `THIRD_PARTY_NOTICES.md`.

## License

MIT (see `LICENSE`). Third-party code keeps its original license.
