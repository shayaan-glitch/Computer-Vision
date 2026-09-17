import os
import cv2
import numpy as np
import pytest
from src.preprocessing import preprocess_image
from src.segmentation import segment_image
from src.feature_extraction import extract_all_features
from src.classification import ShapeClassifier
from src.pipeline import process_image

def test_preprocessing():
    # Create a dummy image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blurred = preprocess_image(img)
    
    assert blurred is not None
    assert len(blurred.shape) == 2  # Should be grayscale
    assert blurred.shape == (100, 100)

def test_segmentation():
    # Create a simple white circle on black background
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    
    contour, thresh = segment_image(img)
    assert contour is not None
    assert cv2.contourArea(contour) > 1000

def test_feature_extraction():
    # Simple square contour
    contour = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]])
    features = extract_all_features(contour)
    
    assert features is not None
    assert len(features) == 13 # 6 geometric + 7 Hu moments
    assert isinstance(features, np.ndarray)

def test_classification():
    classifier = ShapeClassifier()
    # Dummy data
    X = np.random.rand(10, 13).astype(np.float32)
    y = ["circle"] * 5 + ["square"] * 5
    
    classifier.train(X, y)
    
    # Predict
    pred = classifier.predict(X[0])
    assert pred in classifier.classes
