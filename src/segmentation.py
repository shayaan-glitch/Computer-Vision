import cv2
import numpy as np

def apply_otsu_threshold(image):
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def apply_morphology(image):
    # Use closing to fill small holes and opening to remove noise
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening

def find_and_select_contour(thresh_image):
    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Select the largest contour, assuming it's the main shape
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out very small contours which might be noise
    if cv2.contourArea(largest_contour) < 100:
        return None
        
    return largest_contour

def segment_image(blurred_image):
    thresh = apply_otsu_threshold(blurred_image)
    morph = apply_morphology(thresh)
    contour = find_and_select_contour(morph)
    return contour, morph
