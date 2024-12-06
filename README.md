# Image Enhancement Projects
## Overview-
This repository contains two Jupyter notebooks that demonstrate image enhancement techniques using convolutional neural networks (CNNs) with different activation functions. These notebooks are part of a study to evaluate the impact of activation functions on the quality of image enhancement, particularly focusing on enhancing low-light images from the LOL dataset.

## Notebooks-
1. Image Enhancement using ReLU
File: image-enhancement-using-relu.ipynb

This notebook explores the use of the ReLU (Rectified Linear Unit) activation function in a deep learning model designed to enhance underexposed images. The model architecture consists of multiple convolutional layers, with ReLU activation promoting non-linear learning.

Features:
Utilizes ReLU activation layers.
Sequential model architecture with upsampling and downsampling layers.
Evaluation using PSNR (Peak Signal-to-Noise Ratio) as the metric to assess enhancement quality.

2. Image Enhancement using LeakyReLU
File: image-enhancement-using-leakyrelu.ipynb

Similar in structure to the first notebook, this one implements the LeakyReLU activation function, which allows a small gradient when the unit is not active and has shown potential benefits in maintaining gradient flow in deep networks.

Features:
Incorporates LeakyReLU activation to prevent dying neurons problem common with ReLU.
Similar architectural framework as the ReLU notebook for direct comparison.
Uses PSNR for evaluating image quality after processing.

## Usage-
To use these notebooks:

Clone the repository to your local machine or open them in an environment that supports Jupyter (e.g., JupyterHub, Google Colab).
Ensure that all dependencies are installed, including TensorFlow, Keras, NumPy, OpenCV, and Matplotlib. You can install dependencies via pip:
bash
Copy code
pip install tensorflow keras numpy opencv-python-headless matplotlib
Run the notebooks cell by cell, examining the output of each to understand the workflow and modifications made to the images.
Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy
OpenCV
Matplotlib

## Dataset-
The LOL dataset used in these notebooks is publicly available in Kaggle (https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) and consists of low-light images paired with their enhanced versions. This dataset is commonly used to evaluate image enhancement techniques.

## Final Score-
The average PSNR for the ReLU model: 15.70
The average PSNR for the LeakyReLU model: 19.97

## Conclusion-
These notebooks are designed for educational purposes to illustrate how different activation functions in neural networks can affect the performance of tasks such as image enhancement. They provide a basis for further exploration and tweaking of model parameters for improved performance.