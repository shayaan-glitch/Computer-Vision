# Image Analysis and Object Classification Using Classical Computer Vision

This project implements a complete Computer Vision pipeline to identify and classify geometric shapes (circles, squares, triangles) from noisy images. It uses classical computer vision techniques (filtering, thresholding, contour extraction) and a custom K-Nearest Neighbors (KNN) classifier.

## Features
- **Synthetic Dataset Generation**: Creates varied, noisy images to test classical CV techniques robustly.
- **Image Preprocessing**: Grayscale conversion and Gaussian blur.
- **Image Segmentation**: Otsu thresholding and morphological closing/opening.
- **Feature Extraction**: 13 geometric features (Circularity, Aspect Ratio, Solidity, Extent, and Hu Moments).
- **Classification**: Pure NumPy KNN implementation.
- **Evaluation**: Calculates accuracy, precision, recall, F1-score, and confusion matrix.

## Technologies Used
- Python 3
- OpenCV (cv2)
- NumPy
- Pytest (for testing)
- Pillow (for diagram generation)

## Project Structure
```
.
├── data/
│   └── dataset/          # Generated images (train, val, test splits)
├── docs/                 # Documentation and diagrams
├── outputs/              # Trained models and evaluation results
├── scripts/
│   ├── prepare_data.py   # Dataset generator
│   └── generate_diagrams.py # Diagram generator
├── src/
│   ├── classification.py # KNN classifier
│   ├── evaluation.py     # Metrics calculation
│   ├── feature_extraction.py # Shape descriptors
│   ├── pipeline.py       # End-to-end integration
│   ├── preprocessing.py  # Image filtering
│   └── segmentation.py   # Thresholding and contours
├── tests/                # Pytest unit tests
├── main.py               # CLI entry point
├── requirements.txt      # Python dependencies
├── statement.md          # Problem statement
└── README.md             # This file
```

## Requirements
Ensure you have Python installed. The project relies strictly on simple wheels that work across standard environments.

## Installation
1. Clone this repository or navigate to the directory.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
Generate the synthetic dataset (600 images) using the data preparation script:
```bash
python scripts/prepare_data.py --num 600
```
This creates a `data/dataset` folder with `train`, `val`, and `test` subdirectories.

## How to Run

**1. Train the Model**
Train the KNN classifier on the generated training dataset:
```bash
python main.py --train data/dataset/train
```
The model will be saved to `outputs/model.pkl`.

**2. Evaluate the Model**
Test the trained model on the unseen test dataset:
```bash
python main.py --test data/dataset/test
```
Evaluation metrics and a confusion matrix image will be saved to the `outputs/` directory.

**3. Predict a Single Image**
Predict the shape of a single image and generate a visualization:
```bash
python main.py --predict data/dataset/test/circle/circle_0000.jpg
```
The result image with the prediction overlay will be saved to `outputs/`.

## How to Test
Execute the test suite using `pytest`:
```bash
python -m pytest tests/
```

## Example Output
```
=== Evaluation Report ===
Accuracy: 0.6111

Class Metrics:
- circle:
  Precision: 0.6176
  Recall: 0.6562
  F1-Score: 0.6364
...
```

## Limitations
- The system relies on classical contour extraction, meaning heavily overlapping shapes or extreme noise might cause segmentation failures.
- The KNN classifier is simple and uses Euclidean distance directly on features; normalizing the features could improve accuracy in future iterations.
