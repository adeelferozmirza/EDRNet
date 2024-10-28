# MRI Image Analysis Model

## Overview
This repository contains a deep learning EDRNet model for analyzing MRI (Magnetic Resonance Imaging) images. The model has been trained to perform image segmentation and analysis on medical imaging data.

## Pre-trained Model
You can download the pre-trained model weights (.pt file) from here:
[Download Model Weights](https://drive.google.com/file/d/1sTXUWm-b3GWFHCp_FwMmgPS_hBkGMvkA/view?usp=drive_link)

## Requirements
```
pytorch >= 1.7.0
numpy
scikit-learn
pillow
matplotlib
albumentations
opencv-python
```

## Installation
```bash

```

## Usage
1. Download the pre-trained model weights using the link above
2. Place the .pt file in the `models/` directory
3. Run the evaluation script:
```python
python evaluate.py --model_path models/model.pt --data_path path/to/your/mri/data
```

