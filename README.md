# MNIST Handwritten Digit Classification using MLP and CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oNBCiY9seWZBn2klBAfwMjHt8MIxLviD?usp=sharing)
[![Python](https://img.shields.io/badge/Python_3.9_+-3776AB?logo=python&logoColor=FF6F00)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow)](https://tensorflow.org)

<!--
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
-->


> **Brief Description:** This project implements and compares two deep learning models, a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN), for classifying handwritten digits from the MNIST dataset. 

> The goal of this project is to demonstrate the application of both MLP and CNN architectures for image classification on the well-known MNIST dataset. The notebook covers data loading, preprocessing, model definition, training, evaluation, and a comparative analysis of the two models' performance.


## 📊 Dataset

### Dataset Information

- **Name**: MNIST
- **Source**: Dataset available as default in Colab sample data resources.
- **Size**:  It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a 28x28 grayscale image.
- **Format**: Images
  

## 🧠 Model Architecture

### Model Overview

### 1. Multi-Layer Perceptron (MLP)

A basic feedforward neural network with dense layers.

**Architecture:**

- Flatten layer to convert 28x28 images into a 784-dimensional vector.
- Two Dense layers with ReLU activation.
- Dropout layer for regularization.
- Output Dense layer with Softmax activation for classification.

**Configuration:**

- Optimizer: Adam with a learning rate of 0.001
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 15
- Batch Size: 64
- Validation Split: 0.1

### 2. Convolutional Neural Network (CNN)

A more complex network designed to handle spatial data, effective for image recognition.

**Architecture:**

- Two Conv2D layers with ReLU activation for feature extraction.
- Two MaxPooling2D layers for spatial downsampling.
- Flatten layer to prepare for dense layers.
- Dense layer with ReLU activation.
- Dropout layer for regularization.
- Output Dense layer with Softmax activation for classification.

**Configuration:**

- Optimizer: Adam with a learning rate of 0.001
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 15
- Batch Size: 64
- Validation Split: 0.1


## 📈 Results

The notebook provides a comparative analysis of the MLP and CNN models based on their test accuracy and loss. It also includes a visualization of sample predictions and a comparative error analysis.

<table>

<tr>
<td align="center">

#### MODEL PERFORMANCE SAMPLE

</td>

</tr>

<tr>
<td>

![Performance Comparison](https://raw.githubusercontent.com/ritanjit/MNIST_Digit_Classification_MLP_CNN/main/model_predictions.png) 

</td>
</tr>
</table>


## ⚙️ Configuration

### How to Run the Code

1.  **Open in Google Colab:** Click the "Open in Colab" badge.
2.  **Run the cells:** Execute the code cells sequentially from top to bottom.
3.  **Explore the results:** The notebook includes visualizations of sample images, model architectures, training history plots, and a comparative analysis of the MLP and CNN models.

### Files

*   `.ipynb`: The main Colab notebook containing all the code.
*   `model_predictions.png`: A figure visualizing the predicted classes using the model.

## 🚀 Future Work

*   Experiment with different hyperparameters for each model.
*   Implement additional model architectures (e.g., DenseNet, VGG).
*   Explore data augmentation techniques to further improve performance.
*   Analyze misclassifications in more detail to understand model weaknesses.

---

<div align="center">

**⭐ Star this repo if you found it helpful!**
-->

Made with ❤️ by [Ritanjit](https://github.com/ritanjit)

</div>
