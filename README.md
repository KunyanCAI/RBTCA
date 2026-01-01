# Region-Based Text-Consistent Augmentation for Multimodal Medical Segmentation (RBTCA)

**Authors:** Kunyan Cai, Chenggang Yan, Min He, Liangqiong Qu, Shuai Wang, and Tao Tan

**Paper:** MICCAI 2025 Paper [Link](https://link.springer.com/chapter/10.1007/978-3-032-04947-6_51)

## Introduction

This repository contains the official PyTorch implementation of the **RBTCA** framework.

**Region-Based Text-Consistent Augmentation (RBTCA)** is a novel plug-and-play framework designed to harmonize multimodal augmentations while preserving semantic consistency between medical images and their textual reports.

**Key Features:**
*   **Modality-Aware Representation (MAR):** Integrates a image, a spatial text prompt ($C_T$), and a text-guided ROI ($R_I$) into a unified **representation**.
*   **Segmentor Compatibility:** The fused MAR allows the use of standard segmentation backbones (e.g., U-Net).
*   **Text-Consistent Augmentation (TCA):** Applies synchronized geometric and mixing augmentations to the MAR, ensuring the text cues remain aligned with the augmented image.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KunyanCAI/RBTCA
    cd RBTCA/RBTCA_Refactored
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yaml -n rbtca
    conda activate rbtca
    ```

## Dataset Preparation

This code is optimized for the **QaTa-Covid19** dataset. 

*   **QaTa-COV19 Source**: The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset).
*   **Text Descriptions**: The textual reports for QaTa-COV19 are sourced from the [LViT project](https://github.com/HUANGLIZI/LViT).

**Note on Lung Tumor CT Dataset:** Our paper also evaluates an internal **Lung Tumor CT** dataset. However, as that dataset is 3D, uses a different loading format, and is not open-source, it is not included in this public repository.

### QaTa-Covid19 Format
*   **Images & Masks**: Single-channel grayscale images stored as `.png` files.
*   **Text Descriptions**: Stored in an Excel file (`Train_Val_text.xlsx`) with `Image` and `Description` columns.

Please organize your data as follows:

```
datasets/
└── QaTa-Covid19/
    ├──Train_Folder
    |   ├── img/                # PNG images 
    |   ├── labelcol/           # PNG masks 
    |   └── Train_text.xlsx     # Text descriptions
    ├──Val_Folder
    |   ├── img/          
    |   ├── labelcol/      
    |   └── Val_text.xlsx 
    └──Test_Folder
        ├── img/          
        ├── labelcol/     
        └── Test_text.xlsx 

```

## Supported Segmentors

This repository provides three built-in segmentors:
*   **UNet** (Default)
*   **AttnUNet**
*   **ResUNet**

Other segmentors mentioned in the paper, such as **SwinUnet**, **ConvNext**, **ASDA**, **LViT** and **ReMamber**, are based on their respective official implementations.

## Usage

### Configuration

You can modify parameters in `configs/config.py`.
*   **`gpu_ids`**: Set the GPU IDs to use (e.g., `"0"`).
*   **`batch_size`**, **`learning_rate`**, **`epochs`**: Adjust hyperparameters.

### Training

```bash
python -m train.py
```

### Testing

```bash
python -m test.py
```

## License

MIT License.
