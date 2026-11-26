# Third-party notices

## DeepLabV3Plus-Pytorch (ASPP and DeepLabV3+ head in ddv/model.py)
https://github.com/VainF/DeepLabV3Plus-Pytorch

```
MIT License

Copyright (c) 2020 Gongfan Fang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DSCNet - Dynamic Snake Convolution (ddv/dsconv.py, with two fixes: offset accumulation in `DSC._coordinate_map_3D` and batch indexing in `DSC._bilinear_interpolate_3D`)
https://github.com/YaoleiQi/DSCNet — Qi et al., "Dynamic Snake Convolution based on Topological Geometric Constraints for Tubular Structure Segmentation", ICCV 2023.

```
MIT License

Copyright (c) 2025 Yaolei Qi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Deformable Large Kernel Attention (DeformableLKAContext in ddv/context_blocks.py)
Re-implementation following https://github.com/xmindflow/deformableLKA — Azad et al., "Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation", WACV 2024. See the upstream repository for its license.

## VMamba (idea only for VMambaStyleContext in ddv/context_blocks.py)
No VMamba source code is included; the block is an independent pure-PyTorch implementation inspired by https://github.com/MzeroMiko/VMamba — Liu et al., "VMamba: Visual State Space Model", 2024 (MIT License).
