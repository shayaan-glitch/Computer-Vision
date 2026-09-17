import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path

def create_background(size=(200, 200), intensity_range=(150, 255)):
    # Create background with random intensity and some gradient/noise
    base_intensity = random.randint(*intensity_range)
    bg = np.full((size[0], size[1], 3), base_intensity, dtype=np.uint8)
    
    # Add noise to background
    noise = np.random.normal(0, 10, (size[0], size[1], 3)).astype(np.uint8)
    bg = cv2.add(bg, noise)
    return bg

def draw_shape(img, shape_type):
    h, w = img.shape[:2]
    
    # Random properties
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)) # Dark colors
    thickness = -1  # Filled
    
    # Random size (scale) and center
    max_radius = min(h, w) // 4
    min_radius = max_radius // 2
    r = random.randint(min_radius, max_radius)
    
    center_x = random.randint(r + 10, w - r - 10)
    center_y = random.randint(r + 10, h - r - 10)
    
    if shape_type == "circle":
        # Introduce small deformation (ellipse)
        axes = (r + random.randint(-5, 5), r + random.randint(-5, 5))
        angle = random.randint(0, 360)
        cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, color, thickness)
        
    elif shape_type == "square":
        # Calculate square points and rotate
        side = r * 1.8
        pts = np.array([
            [-side/2, -side/2],
            [side/2, -side/2],
            [side/2, side/2],
            [-side/2, side/2]
        ])
        
        # Deformation
        pts += np.random.randint(-3, 3, pts.shape)
        
        # Rotation
        angle = random.uniform(0, 2 * np.pi)
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        pts = np.dot(pts, rot_mat.T)
        pts += [center_x, center_y]
        cv2.fillPoly(img, [np.int32(pts)], color)
        
    elif shape_type == "triangle":
        # Calculate triangle points
        side = r * 2.2
        h_tri = side * np.sqrt(3) / 2
        pts = np.array([
            [0, -2/3 * h_tri],
            [side/2, 1/3 * h_tri],
            [-side/2, 1/3 * h_tri]
        ])
        
        # Deformation
        pts += np.random.randint(-4, 4, pts.shape)
        
        # Rotation
        angle = random.uniform(0, 2 * np.pi)
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        pts = np.dot(pts, rot_mat.T)
        pts += [center_x, center_y]
        cv2.fillPoly(img, [np.int32(pts)], color)
        
    return img

def apply_effects(img):
    # Small blur
    if random.random() > 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
    # Gaussian noise
    noise_std = random.uniform(5, 20)
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img

def generate_dataset(output_dir, num_images=600):
    shapes = ["circle", "square", "triangle"]
    splits = {"train": 0.7, "val": 0.15, "test": 0.15}
    
    # Create directories
    for split in splits.keys():
        for shape in shapes:
            os.makedirs(os.path.join(output_dir, split, shape), exist_ok=True)
            
    # Generate images
    counts = {s: 0 for s in shapes}
    for i in range(num_images):
        shape = shapes[i % 3]
        
        # Generate image
        img = create_background()
        img = draw_shape(img, shape)
        img = apply_effects(img)
        
        # Determine split
        rand_val = random.random()
        if rand_val < splits["train"]:
            split = "train"
        elif rand_val < splits["train"] + splits["val"]:
            split = "val"
        else:
            split = "test"
            
        filename = f"{shape}_{counts[shape]:04d}.jpg"
        filepath = os.path.join(output_dir, split, shape, filename)
        
        cv2.imwrite(filepath, img)
        counts[shape] += 1
        
    print(f"Generated {sum(counts.values())} images in {output_dir}")
    print(f"Distribution: {counts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic shape dataset.")
    parser.add_argument("--output", type=str, default="data/dataset", help="Output directory")
    parser.add_argument("--num", type=int, default=600, help="Number of images to generate")
    args = parser.parse_args()
    
    generate_dataset(args.output, args.num)
