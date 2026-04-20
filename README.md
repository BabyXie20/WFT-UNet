# WFT-UNet

Official implementation of **WFT-UNet: A Wavelet-Fourier Transform UNet for 3D Abdominal Multi-organ Segmentation**.

This repository provides the PyTorch implementation of WFT-UNet, a lightweight spatial-frequency fusion network for 3D abdominal CT multi-organ segmentation. WFT-UNet integrates spatial-domain feature learning, wavelet-domain structural detail enhancement, Fourier-domain dynamic filtering, and cross-attentive spatial-frequency fusion.

## Manuscript

This repository is directly associated with the manuscript submitted to *The Visual Computer*:

**WFT-UNet: A Wavelet-Fourier Transform UNet for 3D Abdominal Multi-organ Segmentation**

If you use this code, model, or experimental setting in your research, please cite the associated manuscript.

## Permanent Archive

The current GitHub repository is publicly available at:

https://github.com/BabyXie20/WFT-UNet.git

A permanent Zenodo archive and DOI will be created for the release version associated with the manuscript. The DOI will be added here once the release is finalized.

## Introduction

WFT-UNet is designed for 3D abdominal CT multi-organ segmentation. It addresses the challenge of jointly preserving local structural details and modeling global anatomical context by combining spatial-domain learning with wavelet- and Fourier-domain representations.

The framework follows a compact U-shaped encoder-decoder architecture and introduces spatial-frequency fusion blocks into the encoder. These blocks combine local spatial modeling, cross-band wavelet enhancement, frequency dynamic filtering, and cross-attentive spatial-frequency fusion.

## Key Features

- **Lightweight 3D volumetric segmentation framework**
- **Cross-band Wavelet Enhancement Module (CWEM)** for structural and boundary detail modeling
- **Frequency Dynamic Filtering Module (FDFM)** for adaptive Fourier-domain spectral modulation
- **Cross-Attentive Fusion Module (CAFM)** for spatial-frequency feature integration
- **Evaluation on BTCV and AMOS2022 abdominal CT benchmarks**

## Requirements

The experiments were conducted with the following environment:

- Python 3.10.8
- PyTorch 2.1.2
- CUDA 11.8
- MONAI
- NumPy
- SimpleITK
- nibabel
- scikit-image
- tqdm
- einops
- PyWavelets
- GPU: NVIDIA Tesla V100 32GB or equivalent

## Installation

```bash
git clone https://github.com/BabyXie20/WFT-UNet.git
cd WFT-UNet
pip install -r requirements.txt
```

## Datasets

### BTCV Dataset

The BTCV abdominal CT dataset contains 30 labeled CT scans with annotations for 13 abdominal organs. Following the commonly used benchmark protocol, 18 scans are used for training and 12 scans are used for testing.

Preprocessing settings:

- Resampling spacing: 1.5 × 1.5 × 2.0 mm³
- Intensity clipping: [-175, 250] HU
- Intensity normalization: linearly rescaled to [0, 1]
- Training patch size: 96 × 96 × 96

### AMOS2022 Dataset

The CT subset of AMOS2022 is used in this work. Following the experimental setting in the manuscript, 200 CT scans are selected, including 160 scans for training, 20 scans for validation, and 20 scans for testing.

Preprocessing settings:

- Resampling spacing: 1.5 × 1.5 × 2.0 mm³
- Intensity clipping: [-125, 275] HU
- Intensity normalization: linearly rescaled to [0, 1]
- Training patch size: 96 × 96 × 96

Please organize the datasets as follows:

```text
data/
├── BTCV/
│   ├── imagesTr/
│   ├── labelsTr/
│   ├── imagesTs/
│   └── labelsTs/
└── AMOS2022/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesVa/
    ├── labelsVa/
    ├── imagesTs/
    └── labelsTs/
```

Please note that the datasets are not distributed in this repository. Users should download them from their official sources and follow the corresponding data usage agreements.

## Training and Evaluation

The training and evaluation procedures are integrated in the same scripts. During training, the model is periodically evaluated according to the validation or testing workflow implemented in the corresponding script.

### BTCV Dataset

```bash
python train_btcv.py
```

### AMOS2022 Dataset

```bash
python train_amos.py
```

Default training settings:

- Optimizer: SGD
- Initial learning rate: 0.01
- Momentum: 0.9
- Batch size: 2
- Patch size: 96 × 96 × 96
- Data augmentation: random affine transformation, random intensity shifting, random intensity scaling, and random Gaussian noise
- Inference strategy: sliding-window prediction with an overlap ratio of 0.5

The evaluation metrics include:

- Dice Similarity Coefficient
- 95th percentile Hausdorff Distance
- Intersection over Union
- Number of trainable parameters
- GFLOPs

## Expected Results

The main results reported in the manuscript are:

| Dataset | Dice (%) | HD95 (mm) | IoU (%) | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|
| BTCV | 82.09 | 8.55 | 71.87 | 11.51M | 163.3 |
| AMOS2022 | 88.14 | 5.689 | 79.98 | 11.51M | 163.3 |

Due to differences in hardware, CUDA versions, random seeds, and preprocessing details, slight numerical variations may occur.

## Model Architecture

The model architecture is implemented in the `networks/` directory:

```text
networks/
├── model.py              # Main WFT-UNet architecture
├── DWT_IDWT_layer.py     # 3D Discrete Wavelet Transform and inverse transform layers
├── DWT_IDWT_Functions.py # Low-level wavelet transform functions
```

### Key Components

- **CWEM: Cross-band Wavelet Enhancement Module**  
  CWEM models dependencies among wavelet subbands and enhances structural and boundary-related features through cross-band wavelet interaction.

- **FDFM: Frequency Dynamic Filtering Module**  
  FDFM performs adaptive modulation of low-, middle-, and high-frequency Fourier responses to strengthen global spectral modeling.

- **CAFM: Cross-Attentive Fusion Module**  
  CAFM integrates spatial and frequency-domain features through cross-channel interaction and spatial attention.

## Project Structure

```text
WFT-UNet/
├── networks/
│   ├── model.py
│   ├── DWT_IDWT_layer.py
│   └── DWT_IDWT_Functions.py
├── train_btcv.py
├── train_amos.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility Checklist

This repository provides:

- [x] Source code for WFT-UNet
- [x] Model architecture implementation
- [x] Training scripts for BTCV and AMOS2022
- [x] Integrated evaluation workflow
- [x] Dataset organization and preprocessing settings
- [x] Environment requirements
- [x] Main experimental settings
- [x] Expected benchmark results
- [x] Manuscript association statement
- [ ] Pretrained checkpoints
- [ ] Zenodo DOI

## Citation

The complete citation information will be updated after publication.

Before publication, please cite the associated manuscript if you use this repository:

**WFT-UNet: A Wavelet-Fourier Transform UNet for 3D Abdominal Multi-organ Segmentation**  
Manuscript submitted to *The Visual Computer*.

## License

This project is licensed under the MIT License.

## Acknowledgments

We thank the developers of MONAI and related open-source medical image segmentation frameworks.
