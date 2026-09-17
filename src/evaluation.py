import numpy as np
import cv2
import os

def calculate_metrics(y_true, y_pred, classes):
    # Initialize confusion matrix
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        cm[true_idx][pred_idx] += 1
        
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    # Calculate precision, recall, F1 for each class
    metrics = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    return accuracy, metrics, cm

def draw_confusion_matrix(cm, classes, output_path):
    # Draw a confusion matrix using OpenCV
    cell_size = 100
    margin = 150
    num_classes = len(classes)
    
    img_size = margin * 2 + num_classes * cell_size
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
    
    # Draw axes labels
    cv2.putText(img, "Predicted", (img_size // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # Y-axis label needs rotation, just place it simply
    cv2.putText(img, "Actual", (10, img_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # Draw grid and text
    for i in range(num_classes):
        # Draw class labels
        cv2.putText(img, classes[i], (margin + i * cell_size + 20, margin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        cv2.putText(img, classes[i], (margin - 100, margin + i * cell_size + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
        
        for j in range(num_classes):
            x1 = margin + j * cell_size
            y1 = margin + i * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Draw cell
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 1)
            
            # Put value
            val = str(cm[i][j])
            text_size = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = x1 + (cell_size - text_size[0]) // 2
            text_y = y1 + (cell_size + text_size[1]) // 2
            cv2.putText(img, val, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
    cv2.imwrite(output_path, img)
    print(f"Confusion matrix image saved to {output_path}")

def evaluate_and_report(y_true, y_pred, classes, output_dir):
    accuracy, metrics, cm = calculate_metrics(y_true, y_pred, classes)
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=== Evaluation Report ===\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Class Metrics:\n")
        for cls in classes:
            f.write(f"- {cls}:\n")
            f.write(f"  Precision: {metrics[cls]['precision']:.4f}\n")
            f.write(f"  Recall: {metrics[cls]['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics[cls]['f1']:.4f}\n")
            
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm) + "\n")
        
    print(f"Evaluation report saved to {report_path}")
    
    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    draw_confusion_matrix(cm, classes, cm_img_path)
