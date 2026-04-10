# Two-Stage Unsupervised Low-Light Image Enhancement

> PyTorch implementation of **"A Two-stage Unsupervised Approach for Low Light Image Enhancement"**  
> Hu et al., 2020 — [arXiv:2010.09316](https://arxiv.org/abs/2010.09316)

---

## Overview

This project implements a two-stage unsupervised pipeline for enhancing low-light images without requiring paired training data. The approach decomposes the problem into:

1. **Stage 1 – Pre-enhancement**: Adaptive tone mapping based on Retinex theory (Equations 2–4 from the paper)
2. **Stage 2 – Post-refinement**: A U-Net style refinement network trained with adversarial loss (Equations 5–9)

The method addresses three key challenges in low-light enhancement: the need for paired training images, poor performance on extremely dark inputs, and noise amplification.

---

## Architecture

```
Low-light image
      │
      ▼
┌─────────────────────┐
│  Stage 1            │   Retinex-based Adaptive Tone Mapping
│  (Pre-enhancement)  │   Eq. (2–4): Lg = log(Lw/L̄w + 1) / log(Lwmax/L̄w + 1)
└─────────────────────┘
      │
      ▼ Pre-enhanced image
      │
      ▼
┌─────────────────────┐
│  Stage 2            │   U-Net Encoder-Decoder (Table I, paper-exact)
│  (Refinement Net)   │   Loss: L = l_rec + λ·l_per + µ·l_tv + β·l_adv
└─────────────────────┘
      │
      ▼ Refined image
```

### Refinement Network (Table I — Exact)

| Layer    | Input → Output     | Spatial Size | In Ch | Out Ch |
|----------|--------------------|--------------|-------|--------|
| conv1    | Y′ → x1            | 128×128      | 3     | 32     |
| conv2    | x1 → x2            | 128×128      | 32    | 32     |
| down1    | x2 → x3            | 64×64        | 32    | 32     |
| down2    | x3 → x4            | 32×32        | 32    | 64     |
| down3    | x4 → x5            | 16×16        | 64    | 128    |
| down4    | x5 → x6            | 8×8          | 128   | 256    |
| conv3    | x6 → x7            | 8×8          | 256   | 512    |
| conv4    | x7 → x8            | 8×8          | 512   | 512    |
| up1      | x8 → x9            | 16×16        | 512   | 256    |
| fusion1  | [x9, x5] → x10     | 16×16        | 384   | 256    |
| up2      | x10 → x11          | 32×32        | 256   | 128    |
| fusion2  | [x11, x4] → x12    | 32×32        | 192   | 128    |
| up3      | x12 → x13          | 64×64        | 128   | 64     |
| fusion3  | [x13, x3] → x14    | 64×64        | 96    | 64     |
| up4      | x14 → x15          | 128×128      | 64    | 32     |
| fusion4  | [x15, x2] → x16    | 128×128      | 64    | 32     |
| conv5    | x16 → Y            | 128×128      | 32    | 3      |

### Loss Function (Eq. 9)

```
L = l_rec + λ·l_per + µ·l_tv + β·l_adv
```

| Term   | Description                          | Weight |
|--------|--------------------------------------|--------|
| l_rec  | L1 reconstruction loss (Eq. 5)       | 1.0    |
| l_per  | VGG16 perceptual loss (Eq. 6)        | λ = 1.0|
| l_tv   | Total variation loss (Eq. 7)         | µ = 0.1|
| l_adv  | Relativistic adversarial loss (Eq. 8)| β = 1.0|

---

## Dataset

This implementation uses the **LOL (Low-Light) dataset**:

- **Training set (our485):** 485 paired low/high-light images
- **Test set (eval15):** 15 paired low/high-light images

The dataset is downloaded automatically from Google Drive during setup. It follows the unpaired training paradigm — low-light images (trainA) and normal-light images (trainB) are used without explicit correspondence during GAN training.

---

## Requirements

```
Python >= 3.8
torch >= 2.0
torchvision
Pillow
numpy
matplotlib
scikit-image
gdown
tqdm
```

Install all dependencies:

```bash
pip install torch torchvision pillow numpy matplotlib scikit-image gdown tqdm
```

---

## Usage

### On Kaggle / Colab (GPU recommended)

Open and run `notebook402bb36b62.ipynb` end-to-end. The notebook:
1. Installs dependencies
2. Downloads and organizes the LOL dataset
3. Defines all models (AdaptiveToneMapping, RefinementNet, Discriminator, losses)
4. Trains for 1000 epochs
5. Evaluates and displays visual results

### Training Configuration

| Hyperparameter | Value          |
|----------------|----------------|
| Optimizer      | Adam           |
| Learning rate  | 0.0001         |
| β1, β2         | 0.9, 0.999     |
| Weight decay   | 0.0001         |
| Batch size     | 64             |
| Patch size     | 128 × 128      |
| Epochs         | 1000           |
| GPU used       | NVIDIA Tesla T4|

---

## Results

### Quantitative Results on LOL (Unpaired Dataset)

| Method              | PSNR ↑ | SSIM ↑ | NIQE ↓ |
|---------------------|--------|--------|--------|
| Input (low-light)   | 10.370 | 0.275  | 5.299  |
| EnlightenGAN        | 17.314 | 0.711  | 4.591  |
| Pre-enhancement only| 17.337 | 0.698  | 7.012  |
| **Ours (Two-stage)**| **18.064** | **0.720** | **4.474** |

### Benchmark Datasets (NIQE ↓, lower is better)

| Method       | MEF   | LIME  | NPE   |
|--------------|-------|-------|-------|
| RetinexNet   | 4.149 | 4.420 | 4.485 |
| LIME         | 3.720 | 4.155 | 4.268 |
| EnlightenGAN | 3.232 | 3.719 | 4.113 |
| KinD         | 3.343 | 3.724 | 3.883 |
| **Ours**     | **3.027** | **3.599** | **3.014** |

---

## Graphical Results & Visualization

Run `visualize_results.ipynb` (included in this repo) to reproduce all figures from the paper:

| Figure | Description |
|--------|-------------|
| Fig. 1 style | Side-by-side: Input → Pre-enhanced → Refined → Ground truth |
| Fig. 3 style | Noise suppression comparison across stages |
| Training curves | Generator and Discriminator loss over epochs |
| Metrics bar chart | PSNR / SSIM / NIQE comparison across methods |
| NIQE benchmark radar | Multi-dataset comparison radar chart |

See `visualize_results.ipynb` for detailed plotting code.

---

## Project Structure

```
.
├── implementation.ipynb    # Main training notebook (Kaggle)
├── visualize_results.ipynb     # Visualization & evaluation notebook
├── README.md
└── data/                       # Auto-created by notebook
    ├── our485/
    │   ├── low/                # 485 low-light training images
    │   └── high/               # 485 normal-light training images
    ├── eval15/
    │   ├── low/                # 15 low-light test images
    │   └── high/               # 15 normal-light test images
    └── dataset/
        ├── trainA/             # Symlinked low-light for GAN training
        ├── trainB/             # Symlinked normal-light for GAN training
        └── test/               # Test images
```

---

## Implementation Notes

- The **Discriminator** uses a relativistic structure (RaGAN) with InstanceNorm, matching Eq. (8)
- The **DownBlock** consists of two conv layers + MaxPool2d, matching the paper's description
- **Skip connections** (fusion layers) concatenate encoder and decoder features at 4 scales
- Training uses the **unpaired paradigm**: low-light images are pre-enhanced in Stage 1, then the GAN refines them using unpaired normal-light images as the real distribution
- The Stage 1 tone mapping is parameter-free and runs at test time with no training required

---

## Citation

```bibtex
@article{hu2020twostage,
  title={A Two-stage Unsupervised Approach for Low light Image Enhancement},
  author={Hu, Junjie and Guo, Xiyue and Chen, Junfeng and Liang, Guanqi and Deng, Fuqin and Lam, Tin Lun},
  journal={arXiv preprint arXiv:2010.09316},
  year={2020}
}
```

---

## Acknowledgements

- [LOL Dataset](https://daooshee.github.io/BMVC2018website/) — Wei et al., BMVC 2018
- [EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) — used as baseline
- [U-Net](https://arxiv.org/abs/1505.04597) — Ronneberger et al., MICCAI 2015
- VGG16 perceptual features from [torchvision](https://pytorch.org/vision/stable/models.html)
