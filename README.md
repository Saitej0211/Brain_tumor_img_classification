# Brain Tumor Image Semantic Segmentation

This project focuses on semantic segmentation of brain tumor images using the [Brain Tumor Image Dataset](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation). The objective is to develop a machine learning pipeline that accurately identifies and segments tumor regions in brain MRI scans.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributions](#contributions)
- [Acknowledgements](#acknowledgements)

## Introduction

Brain tumors are critical medical conditions that require precise diagnosis and treatment planning. Semantic segmentation in medical imaging provides a way to accurately delineate tumor boundaries, aiding in diagnosis and surgical planning. This project utilizes deep learning techniques to perform pixel-wise segmentation of brain tumors from MRI scans.

## Dataset

The dataset is sourced from Kaggle and consists of:
- **MRI images**: These are grayscale images of brain MRI scans.
- **Ground truth masks**: Corresponding masks highlighting tumor regions for semantic segmentation tasks.

### Dataset Details:
- **Total Images**: 3,064
- **Classes**: Binary segmentation (Tumor vs. Non-Tumor regions)
- **File Format**: `.png` images for both input scans and segmentation masks.

You can access the dataset [here](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation).

## Project Workflow

1. **Data Preprocessing**:
   - Resizing and normalizing images.
   - Splitting the dataset into training, validation, and testing subsets.
   - Data augmentation to enhance model generalization.

2. **Model Development**:
   - Selection of a semantic segmentation model architecture (e.g., U-Net, DeepLabV3+).
   - Implementation and fine-tuning of the model.

3. **Training**:
   - Training the model using cross-entropy loss for binary segmentation.
   - Use of optimizers such as Adam and techniques like learning rate scheduling.

4. **Evaluation**:
   - Metrics: Dice Similarity Coefficient (DSC), Intersection over Union (IoU), Precision, Recall.
   - Visualization of segmentation results on test data.

5. **Deployment**:
   - Save and export the trained model for deployment.
   - Create a simple pipeline for inference on new MRI scans.

## Model Architecture

The project uses a U-Net architecture, a popular model for medical image segmentation due to its encoder-decoder structure. Modifications include:
- Pretrained encoders (e.g., ResNet or VGG).
- Dropout layers to prevent overfitting.
- Custom loss functions combining Dice Loss and Binary Cross-Entropy.

## Requirements

The project dependencies are managed using Python. Below is the list of major libraries used:
- Python 3.8+
- TensorFlow/Keras or PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- Albumentations (for data augmentation)

Install the dependencies using:
```bash
pip install -r requirements.txt
