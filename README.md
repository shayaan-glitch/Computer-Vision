# Image Analysis and Object Classification Using Classical Computer Vision

## Overview
This project identifies geometric shapes (circles, squares, and triangles) from noisy images. It uses OpenCV to process the images and extract numerical features, and then classifies the shapes using a custom K-Nearest Neighbors (KNN) algorithm written in NumPy.

The pipeline takes a color image, converts it to grayscale, applies a Gaussian blur to reduce noise, and segments the shape using Otsu's thresholding. The detected contour is measured to generate 13 specific features, which are then used by the KNN classifier to predict the shape.

## Features
- Generates a local dataset of noisy geometric shapes.
- Preprocesses images using grayscale and Gaussian blur.
- Segments shapes using Otsu's thresholding and morphological operations.
- Extracts 13 features (area, perimeter, circularity, aspect ratio, extent, solidity, and 7 Hu Moments).
- Classifies shapes using a custom K=5 K-Nearest Neighbors classifier.
- Includes a terminal interface for training, testing, and single-image prediction.

## Technologies Used
- Python 3
- OpenCV (`cv2`)
- NumPy
- Pytest
- Pillow

## Project Structure
- `data/dataset/`: Generated training, validation, and testing images.
- `docs/`: Documentation and diagrams.
- `outputs/`: Saved models, predictions, and evaluation results.
- `scripts/`: Data generation and diagram scripts.
- `src/`: Core Python modules (preprocessing, segmentation, feature extraction, classification, evaluation, pipeline).
- `tests/`: Automated unit tests.
- `main.py`: Command-line interface.

## Requirements
The dependencies are listed in `requirements.txt`. Only basic, pre-compiled wheels are required.

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
Generate the 600 synthetic images locally:
```bash
python scripts/prepare_data.py --num 600
```
This splits the images into `train`, `val`, and `test` directories inside `data/dataset/`.

## How to Run

**1. Train the Model**
```bash
python main.py --train data/dataset/train
```
This trains the KNN classifier and saves it to `outputs/model.pkl`.

**2. Evaluate the Model**
```bash
python main.py --test data/dataset/test
```
This evaluates the model on the test set and saves the confusion matrix and metrics to `outputs/`.

**3. Predict a Single Image**
```bash
python main.py --predict data/dataset/test/circle/circle_0000.jpg
```
This predicts the shape and saves a visualized image to `outputs/`.

## Testing
Run the unit tests:
```bash
python -m pytest tests/
```

## Results
The model currently achieves 60.00% accuracy on the test set. Because the images include random rotation, scaling, and Gaussian noise, the classical thresholding sometimes struggles to isolate the contour perfectly, leading to confusion primarily between circles and squares.

## Limitations
- The custom KNN uses raw Euclidean distance without feature normalization, which negatively impacts accuracy.
- Heavy background noise can sometimes cause the contour detector to select the wrong region.
