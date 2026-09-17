import cv2
import os
from src.preprocessing import preprocess_image
from src.segmentation import segment_image
from src.feature_extraction import extract_all_features

def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Process
    blurred = preprocess_image(image)
    contour, segmented = segment_image(blurred)
    
    # Feature extraction
    features = extract_all_features(contour)
    return features, image, segmented, contour

def load_dataset(dataset_dir):
    features = []
    labels = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory {dataset_dir} does not exist.")
        
    classes = ["circle", "square", "triangle"]
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        for filename in os.listdir(cls_dir):
            if filename.endswith(".jpg"):
                filepath = os.path.join(cls_dir, filename)
                feat, _, _, _ = process_image(filepath)
                if feat is not None:
                    features.append(feat)
                    labels.append(cls)
                    
    return features, labels
