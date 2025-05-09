# AI Model Training for Chest Radiograph Interpretation

## Overview

This repository contains the implementation of a training pipeline for chest radiograph interpretation using the MIMIC-CXR dataset. The goal of this project is to identify systematic patterns of AI misclassification in chest radiograph interpretation across different demographic groups, including age, gender, and race. The training pipeline supports multiple model architectures and diagnostic tasks, with an emphasis on analyzing systematic AI errors and demographic disparities.

## Dataset

* **MIMIC-CXR**: The training pipeline is designed to utilize the MIMIC-CXR dataset, consisting of 81,432 chest radiographs from 35,143 unique patients.
* The dataset is preprocessed to include metadata such as patient demographics and diagnostic labels.

## Tasks

The training pipeline supports the following six binary classification tasks:

* Cardiomegaly
* Pleural Effusion
* Pulmonary Edema
* Fracture Detection
* Consolidation/Pneumonia
* No Findings (Normal)

## Model Architectures

The following model architectures are implemented and can be trained using the provided training script:

* ConvNeXt Large (`convnext_large.fb_in22k_ft_in1k`)
* ConvNeXt Base (`convnext_base.fb_in22k_ft_in1k` and `convnext_base_384_in22ft1k`)
* EfficientNet B0 (`efficientnet_b0.ra_in1k`)
* ResMLP (`resmlp_36_distilled_224`)
* BEiT Base (`beit_base_patch16_224.in22k_ft_in22k`)
* LeViT (`levit_256.fb_dist_in1k` and `levit_384`)
* RegNet (`regnety_008.pycls_in1k`)
* DenseNet (`densenet121`)
* ConvNeXt Tiny (`convnext_tiny.fb_in1k`)
* ViT Base (`vit_base_patch16_224`)
* MobileViT (`mobilevit_xs.cvnets_in1k`)
* DeiT3 Small (`deit3_small_patch16_224.fb_in22k_ft_in1k`)
* EfficientFormer (`efficientformer_l1`)
* MLP-Mixer (`mixer_b16_224`)
* ResNet50 (`resnet50.tv_in1k`)
* BEiT Large (`beit_large_patch16_224`)
* Swin Transformer Base (`swin_base_patch4_window7_224`)
* DenseNet-121 (`densenet121`)

Each model is trained separately for each classification task, resulting in 120 unique models.

## Training Strategy

* Data split: 60% Training, 20% Validation, 20% Testing
* Stratified by patient to prevent data leakage
* Class balancing: The majority class is downsampled to match the minority class size during training
* Training duration: 500 epochs per model architecture

## Implementation Details

* Framework: PyTorch Lightning
* Model Selection: Models are selected based on balanced accuracy (combining sensitivity and specificity)
* Error Analysis: Error consistency analysis is conducted to identify systematic biases in model predictions.

## Usage

### Installation

```
pip install -r requirements.txt
```

### Training

To train a model, run:

```
python model_training.py <label_index> <model_path>
```

* `label_index`: Index of the label to be trained (e.g., 0, 1, 2, ...)
* `model_path`: Path to the pre-trained model architecture

Example:

```
python model_training.py 0 "resnet50"
```

### Output

* Model checkpoints are saved under the `./weights/` directory.
* Training logs and metrics are recorded using Weights & Biases.

### Predict

To predict, run:

```
python model_predict.py <label_index> <model_path>
```

* `label_index`: Index of the label to be trained (e.g., 0, 1, 2, ...)
* `model_path`: Path to the pre-trained model architecture

Example:

```
python model_predict.py 0 "resnet50"
```
### Output

* Predictions are saved under the `./all_pred/` directory.

## Acknowledgements

This project leverages the MIMIC-CXR dataset and is based on research focused on identifying AI misclassification patterns in chest radiograph interpretation.




