# Problem Statement

The goal of this project is to develop an automated Computer Vision system capable of identifying and classifying geometric shapes (circles, squares, triangles) from noisy images. This project demonstrates the application of classical Computer Vision and Machine Learning algorithms without relying on heavy deep-learning frameworks.

## Scope of the Project

The system is designed to:
- Generate a synthetic dataset of geometric shapes with controlled variations (noise, rotation, scale, position).
- Preprocess images using grayscale conversion and Gaussian filtering to reduce noise.
- Segment shapes using Otsu's thresholding and morphological operations.
- Extract descriptive geometric features including area, perimeter, circularity, aspect ratio, and Hu Moments.
- Classify the extracted features into predefined shape categories using a K-Nearest Neighbors (KNN) classifier.
- Provide a clear, modular Command Line Interface (CLI) for dataset preparation, model training, and evaluation.

## Target Users

The primary target users are students, educators, and computer vision beginners who want a clear, reproducible example of classical image analysis and object classification pipelines. It is designed to be easily runnable on standard hardware without GPU acceleration.

## High-level Features

1. **Synthetic Data Generator**: Reproducibly creates noisy images with varying parameters to test the robustness of the CV pipeline.
2. **Modular CV Pipeline**: Clearly separated modules for preprocessing, segmentation, feature extraction, and classification.
3. **Classical Feature Engineering**: Calculates 13 distinct numerical features (geometric and Hu Moments) to represent shape independent of scale and rotation.
4. **Custom KNN Classifier**: A lightweight K-Nearest Neighbors implementation to categorize shapes based on extracted features.
5. **Comprehensive CLI**: A terminal interface for end-to-end execution, saving models, and running batch evaluations.
