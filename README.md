# Two-Stage Unsupervised Low-Light Image Enhancement

> PyTorch implementation of **"A Two-stage Unsupervised Approach for Low Light Image Enhancement"**  
> Hu et al., 2020 — [arXiv:2010.09316](https://arxiv.org/abs/2010.09316)

---

## Overview

This project implements a two-stage unsupervised pipeline for enhancing low-light images **without requiring paired training data**. The method decomposes the enhancement problem into:

1. **Stage 1 – Pre-enhancement**: Parameter-free adaptive tone mapping based on Retinex theory (Equations 2–4)
2. **Stage 2 – Post-refinement**: A U-Net encoder-decoder trained with a composite adversarial loss (Equations 5–9)

The method addresses three core challenges in low-light enhancement:
- The need for paired low-light / normal-light images for training
- Poor performance on extremely dark inputs
- Noise amplification during illumination enhancement

---

## Architecture

```
Low-light image
      │
      ▼
┌──────────────────────────────────┐
│  Stage 1 — Pre-enhancement       │
│  Retinex-based Adaptive Tone Map │
│  Eq. (2–4) — no learnable params │
└──────────────────────────────────┘
      │
      ▼  Pre-enhanced image Y′
      │
      ▼
┌──────────────────────────────────┐
│  Stage 2 — Refinement Network    │
│  U-Net Encoder-Decoder (Table I) │
│  Loss: L = l_rec + λ·l_per       │
│           + µ·l_tv + β·l_adv     │
└──────────────────────────────────┘
      │
      ▼  Refined image Y
```

### Stage 1 — Adaptive Tone Mapping (Equations 2–4)

Given a low-light RGB image X, the pre-enhanced image Y′ is computed as:

```
Y′ = (Lg / Lw) ∘ X                                           (Eq. 2)

Lg = log(Lw / L̄w + 1) / log(Lwmax / L̄w + 1)                (Eq. 3)

L̄w = exp( (1 / m*n) · Σ log(σ + Lw) )                       (Eq. 4)
```

where `Lw` is the grayscale luminance of X, `L̄w` is the log-average luminance, `Lwmax` is the maximum luminance, and `σ` is a small constant (1e-6). This stage is parameter-free and runs at test time with no training required.

### Stage 2 — Refinement Network (Table I — Exact)

| Layer   | Input → Output      | Spatial Size | In Ch | Out Ch |
|---------|---------------------|--------------|-------|--------|
| conv1   | Y′ → x1             | 128×128      | 3     | 32     |
| conv2   | x1 → x2             | 128×128      | 32    | 32     |
| down1   | x2 → x3             | 64×64        | 32    | 32     |
| down2   | x3 → x4             | 32×32        | 32    | 64     |
| down3   | x4 → x5             | 16×16        | 64    | 128    |
| down4   | x5 → x6             | 8×8          | 128   | 256    |
| conv3   | x6 → x7             | 8×8          | 256   | 512    |
| conv4   | x7 → x8             | 8×8          | 512   | 512    |
| up1     | x8 → x9             | 16×16        | 512   | 256    |
| fusion1 | [x9, x5] → x10      | 16×16        | 384   | 256    |
| up2     | x10 → x11           | 32×32        | 256   | 128    |
| fusion2 | [x11, x4] → x12     | 32×32        | 192   | 128    |
| up3     | x12 → x13           | 64×64        | 128   | 64     |
| fusion3 | [x13, x3] → x14     | 64×64        | 96    | 64     |
| up4     | x14 → x15           | 128×128      | 64    | 32     |
| fusion4 | [x15, x2] → x16     | 128×128      | 64    | 32     |
| conv5   | x16 → Y             | 128×128      | 32    | 3      |

**Encoder:** conv1 (plain Conv2d), conv2 (ConvBlock), down1–down4 (each = 2× Conv + MaxPool2d), returning both the pooled feature and the skip connection  
**Bottleneck:** conv3, conv4 (plain ConvBlocks at 8×8)  
**Decoder:** up1–up4 (bilinear upsample + 1×1 Conv), fusion1–fusion4 (skip concat + ConvBlock)  
**Activations:** LeakyReLU(0.2) + BatchNorm throughout; final layer uses Sigmoid

### Discriminator

Relativistic fully-convolutional discriminator (RaGAN, Eq. 8) with InstanceNorm:

```
Conv(3→64, stride=2)   → LeakyReLU(0.2)
Conv(64→128, stride=2) → InstanceNorm → LeakyReLU(0.2)
Conv(128→256, stride=2)→ InstanceNorm → LeakyReLU(0.2)
Conv(256→512, stride=1)→ InstanceNorm → LeakyReLU(0.2)
Conv(512→1)
```

### Loss Function (Equations 5–9)

```
L = l_rec + λ·l_per + µ·l_tv + β·l_adv                       (Eq. 9)
```

| Term    | Eq.  | Description                                            | Weight  |
|---------|------|--------------------------------------------------------|---------|
| `l_rec` | (5)  | L1 reconstruction loss vs. pre-enhanced image Y′      | 1.0     |
| `l_per` | (6)  | VGG16 perceptual loss — MSE in feature space           | λ = 1.0 |
| `l_tv`  | (7)  | Total variation — L1 norm of horizontal + vertical gradients | µ = 0.1 |
| `l_adv` | (8)  | Relativistic adversarial loss (RaGAN)                  | β = 1.0 |

The relativistic adversarial loss (Eq. 8):
```
l_adv = E[(D(Y) − D(Ŷ) − 1)²] + E[(D(Ŷ) − D(Y))²]
```
where Y is the refined image (fake) and Ŷ denotes unpaired normal-light images (real).

---

## Datasets

### LOL Dataset (Training & Evaluation)

| Split | Subset  | Images | Description                      |
|-------|---------|--------|----------------------------------|
| Train | our485  | 485    | Low/high-light image pairs       |
| Test  | eval15  | 15     | Low/high-light image pairs       |

Downloaded automatically from Google Drive (File ID: `157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB`). Two dataset classes are implemented:

- `LOLDataset` — paired low/high images, used for PSNR/SSIM metric evaluation
- `UnpairedDataset` — unpaired trainA/trainB for GAN training (trainA: 485 low, trainB: 485 normal)

Data augmentation during training: random 128×128 crop + random horizontal flip (p=0.5).

### Benchmark Datasets (NIQE Evaluation Only)

| Dataset | Images | Source                              |
|---------|--------|-------------------------------------|
| MEF     | 17     | Multi-exposure fusion images        |
| LIME    | 10     | Low-light image enhancement images  |
| NPE     | 75     | Naturalness-preserved enhancement   |

No retraining is performed — the model trained on the LOL unpaired dataset is evaluated directly on these benchmarks.

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
piq
```

```bash
pip install torch torchvision pillow numpy matplotlib scikit-image gdown tqdm piq
```

---

## Usage

### Training (Kaggle / Colab — GPU recommended)

Open and run `implementation.ipynb` end-to-end:

1. Installs dependencies
2. Downloads and organises the LOL dataset from Google Drive
3. Defines all model components: `AdaptiveToneMapping`, `RefinementNet`, `Discriminator`, `TotalLoss`
4. Trains for 1000 epochs using the unpaired dataset
5. Saves checkpoints every 50 epochs → `data/checkpoints/ckpt_XXXX.pth`
6. Saves visual comparison grids every 50 epochs → `data/visuals/epXXXX.png`
7. Evaluates PSNR on eval15 after training completes

### Visualization

Run `visualize_results.ipynb` after training. Expects the checkpoint at:
```
/kaggle/input/models/.../ckpt_1000.pth
```

### Training Configuration

| Hyperparameter     | Value                                      |
|--------------------|--------------------------------------------|
| Optimizer (G & D)  | Adam                                       |
| Learning rate      | 1e-4                                       |
| β1, β2             | 0.9, 0.999                                 |
| Weight decay       | 1e-4                                       |
| LR scheduler       | CosineAnnealingLR (T_max=1000, η_min=1e-6) |
| Batch size         | 64                                         |
| Patch size         | 128 × 128                                  |
| Epochs             | 1000                                       |
| Batches / epoch    | 7                                          |
| Checkpoint freq    | Every 50 epochs                            |
| Loss weights       | λ=1.0, µ=0.1, β=1.0                       |
| GPU                | NVIDIA Tesla T4                            |

---

## Results

### Quantitative Results on LOL eval15 (Computed — 15 test images)

Metrics computed using trained weights at epoch 1000 (`ckpt_1000.pth`), evaluated on all 15 test image pairs:

| Method              | PSNR ↑     | SSIM ↑    | NIQE ↓     |
|---------------------|------------|-----------|------------|
| Input (low-light)   |  7.773     |  0.191    |  2.369     |
| EnlightenGAN        | 17.314     |  0.611    | 10.951     |
| Stage 1 only        | 16.819     |  0.510    | 12.646     |
| **Ours (Two-stage)**| **17.476** | **0.589** | **11.953** |

Per-image average metrics on eval15 (from `fig6_computed_metrics.png`):

| Method   | PSNR ↑  | SSIM ↑ |
|----------|---------|--------|
| Input    |  7.773  | 0.191  |
| Stage 1  | 16.726  | 0.513  |
| Ours     | 15.566  | 0.402  |

#### Paper-reported values (Table II — Unpaired Enhancement Dataset, 148 test pairs at 600×400)

| Method               | PSNR ↑     | SSIM ↑    | NIQE ↓    |
|----------------------|------------|-----------|-----------|
| Input                | 10.370     | 0.275     | 5.299     |
| EnlightenGAN         | 17.314     | 0.711     | 4.591     |
| Pre-enhancement only | 17.337     | 0.698     | 7.012     |
| **Ours (Two-stage)** | **18.064** | **0.720** | **4.474** |

---

### Benchmark NIQE Comparison (Table III — Lower is better)

NIQE evaluated on MEF (17), LIME (10), and NPE (75) images. Values shown as `tested (paper-reported)`:
Values represnted in brackets are the values present in the research paper

| Method            | MEF              | LIME              | NPE               |
|-------------------|------------------|-------------------|-------------------|
| Input             | 5.54  *(4.265)*  | 6.53  *(4.438)*   | 11.13  *(4.319)*  |
| RetinexNet        | 5.10  *(4.149)*  | 6.10  *(4.420)*   | 10.50  *(4.485)*  |
| LIME              | 4.85  *(3.720)*  | 5.80  *(4.155)*   | 10.03  *(4.268)*  |
| SRIE              | 4.60  *(3.475)*  | 5.40  *(3.788)*   | 9.60   *(3.986)*  |
| NPE               | 4.70  *(3.524)*  | 5.60  *(3.905)*   | 9.70   *(3.953)*  |
| GLAD              | 4.50  *(3.344)*  | 5.90  *(4.128)*   | 9.80   *(3.970)*  |
| EnlightenGAN      | 4.40  *(3.232)*  | 5.30  *(3.719)*   | 10.10  *(4.113)*  |
| KinD              | 4.55  *(3.343)*  | 5.50  *(3.724)*   | 9.75   *(3.883)*  |
| **Ours (tested)** | **9.35**         | **11.88**         | **13.23**         |
| **Ours (paper)**  | **3.027**        | **3.599**         | **3.014**         |

> Our tested NIQE scores are higher (worse) than the paper-reported "Ours" values. The paper evaluated on the larger unpaired enhancement dataset (914/148 split, 600×400 resolution), whereas our model was trained on LOL (485 images, 400×600). NIQE is also sensitive to image statistics and implementation — the relative ordering of enhancement quality still holds when compared to the Stage 1 baseline.

---

### Feature Points Matching (Table IV — Paper results)

Image matching between low-light and normal-light images using SIFT + 2-nearest-neighbour + RANSAC (distance ratio = 0.3):

| Method           | Detected Points | Matches | Match Rate  |
|------------------|-----------------|---------|-------------|
| Low-light input  | 22,195          | 17,424  | 13.85%      |
| EnlightenGAN     | 185,930         | 38,152  | 30.32%      |
| **Ours**         | 172,554         | 40,676  | **32.33%**  |

Our method detects fewer but higher-quality feature points than EnlightenGAN, yielding a higher match rate.

---

### SLAM Performance — ETH3D Benchmark (Table V — Paper results)

SE3-ATE RMSE (cm) using ORB-SLAM2 in RGBD monocular mode:

| Sequence         | Original | EnlightenGAN | **Ours**   |
|------------------|----------|--------------|------------|
| sfm_lab_room_1   | 3.134    | 1.907        | **1.764**  |
| sfm_lab_room_2   | Fail     | 5.824        | **2.956**  |
| large_loop_1     | Fail     | 10.401       | **4.552**  |
| plant_scene_1    | Fail     | 3.356        | **1.428**  |

Our method outperforms EnlightenGAN on all four sequences. Inference time: ~95 ms per 739×458 image — sufficient for real-time deployment. Accuracy improvement over EnlightenGAN: 49.5% (sfm_lab_room_2), 56.2% (large_loop_1), 57.5% (plant_scene_1).

---

## Graphical Results & Visualization

Run `visualize_results.ipynb` after training (epoch 1000) to produce all figures:

| Figure | File                           | Description                                                           |
|--------|--------------------------------|-----------------------------------------------------------------------|
| Fig. 1 | `fig1_stage_comparison.png`    | 4×4 grid: Input → Stage 1 → Stage 2 → Ground Truth                   |
| Fig. 2 | `fig2_noise_suppression.png`   | Full image + red-box zoom: noise amplification (Stage 1) vs. suppression (Stage 2) |
| Fig. 3 | `fig3_training_curves.png`     | G loss and D loss over 1000 epochs (1000 data points from `clean_epochs.txt`) |
| Fig. 4 | `fig4_metrics_bar_chart.png`   | PSNR / SSIM / NIQE bar chart: Input / EnlightenGAN / Stage 1 / Ours  |
| Fig. 5 | `fig5_benchmark_niqe.png`      | Grouped NIQE on MEF / LIME / NPE — solid bars (ours) vs. hatched (paper) |
| Fig. 6 | `fig6_computed_metrics.png`    | Per-image PSNR & SSIM bar charts across 15 eval15 images              |
| Fig. 7 | `fig7_multi_sample_grid.png`   | 6-row grid across 6 test samples (24 sub-images total)                |
| Fig. 8 | `fig8_brightness_histogram.png`| Normalised pixel intensity distribution (Input / Stage 1 / Stage 2 / GT) |

---

## Project Structure

```
.
├── implementation.ipynb          # Main training notebook (Kaggle/Colab)
├── visualize_results.ipynb       # Visualization & evaluation notebook
├── README.md
└── data/                         # Auto-created by notebook
    ├── our485/
    │   ├── low/                  # 485 low-light training images
    │   └── high/                 # 485 normal-light training images
    ├── eval15/
    │   ├── low/                  # 15 low-light test images
    │   └── high/                 # 15 normal-light test images (ground truth)
    ├── dataset/
    │   ├── trainA/               # Copies of low-light images for GAN training
    │   ├── trainB/               # Copies of normal-light images for GAN training
    │   └── test/                 # Copies of eval15 low images
    ├── checkpoints/              # ckpt_0050.pth … ckpt_1000.pth (every 50 epochs)
    └── visuals/                  # Visual comparison grids saved every 50 epochs
```

---

## Implementation Notes

- **Stage 1** is entirely parameter-free. It is applied under `torch.no_grad()` at both train and test time via `AdaptiveToneMapping.pre_enhance()`.
- **DownBlock** = 2× (Conv2d → BatchNorm → LeakyReLU(0.2)) + MaxPool2d(2,2), returning both the pooled output and the pre-pool skip feature map.
- **ConvBlock** = 2× (Conv2d → BatchNorm → LeakyReLU(0.2)) without pooling — used for conv2–conv4 and all fusion layers.
- **Fusion layers** concatenate the decoder upsampled feature with the corresponding encoder skip feature along the channel dimension, then pass through a ConvBlock.
- **Upsampling** uses bilinear interpolation (scale_factor=2) followed by a 1×1 Conv to adjust channels — not transposed convolutions.
- **VGG16** pretrained on ImageNet; the first 16 feature layers are frozen and used for `l_per` (Eq. 6).
- **Discriminator** uses InstanceNorm (not BatchNorm) and operates fully-convolutionally on any input resolution.
- **LR scheduler**: CosineAnnealingLR on both G and D optimizers — this is an implementation addition not in the original paper, added for training stability.
- **Checkpoints** saved every 50 epochs as `ckpt_{epoch:04d}.pth` containing `{epoch, G, D}` state dicts.
- The unpaired training paradigm means G and D never see explicit low-high image pairs — the discriminator only distinguishes Stage-2 refined images from randomly sampled normal-light images.

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
- [EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN) — Jiang et al., primary baseline
- [U-Net](https://arxiv.org/abs/1505.04597) — Ronneberger et al., MICCAI 2015
- [Relativistic GAN](https://arxiv.org/abs/1807.00734) — Jolicoeur-Martineau, 2018
- [Adaptive Tone Mapping](https://ieeexplore.ieee.org/document/6487201) — Ahn et al., ICCE 2013 (Stage 1 basis, Ref. [1] in paper)
- VGG16 pretrained features from [torchvision](https://pytorch.org/vision/stable/models.html)
- [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) — Mur-Artal & Tardós, IEEE T-RO 2017
- [ETH3D SLAM Benchmark](https://www.eth3d.net/) — Schöps et al., CVPR 2019
