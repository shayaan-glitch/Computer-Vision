import cv2
import numpy as np

def extract_geometric_features(contour):
    # Area and Perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return None
        
    # Circularity: 4 * pi * (Area / Perimeter^2)
    # 1.0 for a perfect circle, smaller for other shapes
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # Bounding Box features
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h)
    
    # Convex Hull features
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    return [area, perimeter, circularity, aspect_ratio, extent, solidity]

def extract_hu_moments(contour):
    # Calculate Moments
    moments = cv2.moments(contour)
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments)
    
    # Log transform Hu Moments for better scaling, keeping sign
    # Formula: -1 * sign(h) * log10(abs(h))
    log_hu_moments = []
    for h in hu_moments:
        h = h[0]
        if h == 0:
            log_hu_moments.append(0.0)
        else:
            log_hu_moments.append(-1 * np.sign(h) * np.log10(abs(h)))
            
    return log_hu_moments

def extract_all_features(contour):
    if contour is None:
        return None
        
    geom_features = extract_geometric_features(contour)
    if geom_features is None:
        return None
        
    hu_features = extract_hu_moments(contour)
    
    # Combine features into a single 1D numpy array
    return np.array(geom_features + hu_features, dtype=np.float32)
