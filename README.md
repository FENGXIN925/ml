# Machine Learning and Deep Learning Models for Pneumonia Detection

This repository contains implementations and results of various machine learning and deep learning models for detecting pneumonia using chest X-ray images. The goal is to compare the performance of different approaches and provide insights into the best-performing techniques.

## Project Structure

- **results/**: Directory containing detailed result files for various models and experiments.
- **CNN1.py**: Implementation of a Convolutional Neural Network (CNN) for pneumonia detection.
- **GCN.py**: Implementation of a Graph Convolutional Network (GCN).
- **Hog Svm.py**: Code for feature extraction using Histogram of Oriented Gradients (HOG) and classification using Support Vector Machine (SVM).
- **Pneumonia_Chest.ipynb**: Jupyter Notebook for exploratory data analysis (EDA) and a CNN demonstration.
- **Transformer.py**: Implementation of a Transformer-based model for pneumonia detection.
- **cnn_results.json**: Results of the CNN model.
- **gcn_results.json**: Results of the GCN model.
- **hog_svm_results.json**: Results of the HOG-SVM model.
- **transfer_results.json**: Results of a transfer learning-based approach.
- **model_comparison.csv**: CSV file summarizing the performance metrics of all models.
- **requirements.txt**: List of dependencies required to run the project.

## Usage

### 1. Environment Setup
Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
### 2. get data set from kaggle
```python
from google.colab import files

# 上传 `kaggle.json`
files.upload()
```

```python
import os

# 创建 `.kaggle` 目录
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
```python
!pip install kaggle
```
```python
### import kagglehub

import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)
```

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/drive/My Drive/kaggle'
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q -o chest-xray-pneumonia.zip
```
