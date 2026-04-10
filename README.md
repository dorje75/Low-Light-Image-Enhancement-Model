Two-Stage Unsupervised Low-Light Image Enhancement

PyTorch implementation of "A Two-stage Unsupervised Approach for Low Light Image Enhancement"
Hu et al., 2020 — arXiv:2010.09316


Overview
This project implements a two-stage unsupervised pipeline for enhancing low-light images without requiring paired training data. The approach decomposes the problem into:

Stage 1 – Pre-enhancement: Adaptive tone mapping based on Retinex theory (Equations 2–4 from the paper)
Stage 2 – Post-refinement: A U-Net style refinement network trained with adversarial loss (Equations 5–9)

The method addresses three key challenges in low-light enhancement: the need for paired training images, poor performance on extremely dark inputs, and noise amplification.

Architecture
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
Refinement Network (Table I — Exact)
LayerInput → OutputSpatial SizeIn ChOut Chconv1Y′ → x1128×128332conv2x1 → x2128×1283232down1x2 → x364×643232down2x3 → x432×323264down3x4 → x516×1664128down4x5 → x68×8128256conv3x6 → x78×8256512conv4x7 → x88×8512512up1x8 → x916×16512256fusion1[x9, x5] → x1016×16384256up2x10 → x1132×32256128fusion2[x11, x4] → x1232×32192128up3x12 → x1364×6412864fusion3[x13, x3] → x1464×649664up4x14 → x15128×1286432fusion4[x15, x2] → x16128×1286432conv5x16 → Y128×128323
Loss Function (Eq. 9)
L = l_rec + λ·l_per + µ·l_tv + β·l_adv
TermDescriptionWeightl_recL1 reconstruction loss (Eq. 5)1.0l_perVGG16 perceptual loss (Eq. 6)λ = 1.0l_tvTotal variation loss (Eq. 7)µ = 0.1l_advRelativistic adversarial loss (Eq. 8)β = 1.0

Dataset
This implementation uses the LOL (Low-Light) dataset:

Training set (our485): 485 paired low/high-light images
Test set (eval15): 15 paired low/high-light images

The dataset is downloaded automatically from Google Drive during setup. It follows the unpaired training paradigm — low-light images (trainA) and normal-light images (trainB) are used without explicit correspondence during GAN training.

Requirements
Python >= 3.8
torch >= 2.0
torchvision
Pillow
numpy
matplotlib
scikit-image
gdown
tqdm
Install all dependencies:
bashpip install torch torchvision pillow numpy matplotlib scikit-image gdown tqdm

Usage
On Kaggle / Colab (GPU recommended)
Open and run notebook402bb36b62.ipynb end-to-end. The notebook:

Installs dependencies
Downloads and organizes the LOL dataset
Defines all models (AdaptiveToneMapping, RefinementNet, Discriminator, losses)
Trains for 1000 epochs
Evaluates and displays visual results

Training Configuration
HyperparameterValueOptimizerAdamLearning rate0.0001β1, β20.9, 0.999Weight decay0.0001Batch size64Patch size128 × 128Epochs1000GPU usedNVIDIA Tesla T4

Results
Quantitative Results on LOL (Unpaired Dataset)
MethodPSNR ↑SSIM ↑NIQE ↓Input (low-light)10.3700.2755.299EnlightenGAN17.3140.7114.591Pre-enhancement only17.3370.6987.012Ours (Two-stage)18.0640.7204.474
Benchmark Datasets (NIQE ↓, lower is better)
MethodMEFLIMENPERetinexNet4.1494.4204.485LIME3.7204.1554.268EnlightenGAN3.2323.7194.113KinD3.3433.7243.883Ours3.0273.5993.014

Graphical Results & Visualization
Run visualize_results.ipynb (included in this repo) to reproduce all figures from the paper:
FigureDescriptionFig. 1 styleSide-by-side: Input → Pre-enhanced → Refined → Ground truthFig. 3 styleNoise suppression comparison across stagesTraining curvesGenerator and Discriminator loss over epochsMetrics bar chartPSNR / SSIM / NIQE comparison across methodsNIQE benchmark radarMulti-dataset comparison radar chart
See visualize_results.ipynb for detailed plotting code.

Project Structure
.
├── notebook402bb36b62.ipynb    # Main training notebook (Kaggle)
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

Implementation Notes

The Discriminator uses a relativistic structure (RaGAN) with InstanceNorm, matching Eq. (8)
The DownBlock consists of two conv layers + MaxPool2d, matching the paper's description
Skip connections (fusion layers) concatenate encoder and decoder features at 4 scales
Training uses the unpaired paradigm: low-light images are pre-enhanced in Stage 1, then the GAN refines them using unpaired normal-light images as the real distribution
The Stage 1 tone mapping is parameter-free and runs at test time with no training required


Citation
bibtex@article{hu2020twostage,
  title={A Two-stage Unsupervised Approach for Low light Image Enhancement},
  author={Hu, Junjie and Guo, Xiyue and Chen, Junfeng and Liang, Guanqi and Deng, Fuqin and Lam, Tin Lun},
  journal={arXiv preprint arXiv:2010.09316},
  year={2020}
}

Acknowledgements

LOL Dataset — Wei et al., BMVC 2018
EnlightenGAN — used as baseline
U-Net — Ronneberger et al., MICCAI 2015
VGG16 perceptual features from torchvision
