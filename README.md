# WFT-UNet

A Wavelet-Frequency-Transformer based U-Net for medical image segmentation.

## Introduction

WFT-UNet is a novel network architecture that integrates wavelet transform and frequency domain analysis with transformer-based attention mechanisms for 3D medical image segmentation. The model leverages dual-domain learning to capture both spatial and frequency features effectively.

## Features

- **Wavelet Transform Integration**: Utilizes 3D Discrete Wavelet Transform (DWT) for multi-scale feature extraction
- **Frequency Domain Learning**: Incorporates frequency branch for enhanced feature representation
- **Cross-Band Attention**: Novel attention mechanism for inter-band feature fusion
- **Dual-Domain Architecture**: Combines spatial and frequency domain processing

## Requirements

- Python 3.8+
- CUDA 11.8+
- GPU: VGPU-32GB (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

### BTCV Dataset

The Beyond the Cranial Vault (BTCV) dataset contains 30 abdominal CT scans with multi-organ segmentation labels.

### AMOS2022 Dataset

The AMOS (AMOS Challenge 2022) dataset provides large-scale multi-organ CT and MRI images with segmentation annotations.

## Training

### BTCV Dataset

```bash
python train_btcv.py
```

### AMOS2022 Dataset

```bash
python train_amos.py
```

## Model Architecture

The model architecture is implemented in the `networks/` directory:

```
networks/
├── model.py              # Main model definition (WFT-UNet)
├── DWT_IDWT_layer.py     # 3D Discrete Wavelet Transform layers
├── DWT_IDWT_Functions.py # Wavelet transform functions
```

### Key Components

- **DualDomainBlock**: Combines wavelet and spatial processing branches
- **WaveletCrossBandBlock**: Cross-band attention for wavelet coefficients
- **BandGraphFusion**: Graph-based fusion of detail bands
- **FrequencyBranch**: Frequency domain feature extraction
- **CAFM**: Cross-Attention Fusion Module

## Project Structure

```
WFT-UNet/
├── networks/
│   ├── model.py
│   ├── DWT_IDWT_layer.py
│   ├── DWT_IDWT_Functions.py
├── train_btcv.py
├── train_amos.py
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [MONAI](https://monai.io/) - Medical Open Network for AI
- [Swin-UNETR](https://github.com/Project-MONAI/MONAI) - Swin Transformer for Medical Image Segmentation
