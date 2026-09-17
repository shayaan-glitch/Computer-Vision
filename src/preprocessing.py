import cv2

def convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, kernel_size, sigmaX)

def preprocess_image(image):
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray)
    return blurred
