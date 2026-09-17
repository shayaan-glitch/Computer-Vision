from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'VITyarthi - Computer Vision Project Report', border=False, ln=1, align='C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 8, title, ln=1, fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

def generate_report():
    pdf = PDF()
    pdf.add_page()
    
    # 1. Cover Page
    pdf.set_font('helvetica', 'B', 24)
    pdf.ln(40)
    pdf.cell(0, 10, 'Image Analysis and Object Classification', ln=1, align='C')
    pdf.set_font('helvetica', '', 16)
    pdf.cell(0, 10, 'Using Classical Computer Vision', ln=1, align='C')
    pdf.ln(20)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Student Project Report', ln=1, align='C')
    pdf.add_page()
    
    # Content
    sections = [
        ("2. Introduction", "This project is a Computer Vision pipeline that identifies and classifies geometric shapes (circles, squares, triangles) from noisy images. It uses OpenCV to process the images and extract numerical features, which are then passed to a classifier."),
        ("3. Problem Statement", "The goal of this project is to build a system that can identify shapes using classical computer vision techniques rather than deep learning. It demonstrates how image filtering, contour extraction, and geometric feature calculation can be combined with a K-Nearest Neighbors (KNN) classifier."),
        ("4. Functional Requirements", "- Generate a dataset of 600 synthetic shape images.\n- Process images using grayscale conversion and Gaussian blur.\n- Segment shapes using Otsu's thresholding.\n- Extract 13 numerical shape features.\n- Classify shapes using a K=5 KNN algorithm.\n- Provide a command-line interface for training and testing."),
        ("5. Non-functional Requirements", "- Performance: The system should run on a standard CPU without requiring GPU acceleration.\n- Maintainability: The code is split into distinct modules for preprocessing, feature extraction, and classification.\n- Reproducibility: The dataset is generated locally via a script so anyone can reproduce the environment.\n- Usability: The terminal interface uses clear arguments for all operations."),
    ]
    
    for title, body in sections:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    # Diagrams Section
    pdf.chapter_title("6. System Architecture & 7. Design Diagrams")
    pdf.chapter_body("The pipeline consists of Preprocessing, Segmentation, Feature Extraction, and Classification modules. Below are the design diagrams.")
    
    if os.path.exists('docs/images/architecture_diagram.png'):
        pdf.image('docs/images/architecture_diagram.png', w=160)
        pdf.ln(5)
    
    if os.path.exists('docs/images/workflow_diagram.png'):
        pdf.image('docs/images/workflow_diagram.png', w=100)
        pdf.ln(5)
        
    pdf.add_page()
    if os.path.exists('docs/images/use_case_diagram.png'):
        pdf.image('docs/images/use_case_diagram.png', w=120)
        pdf.ln(5)
        
    if os.path.exists('docs/images/class_diagram.png'):
        pdf.image('docs/images/class_diagram.png', w=120)
        pdf.ln(5)
        
    sections_2 = [
        ("8. Design Decisions & Rationale", "A synthetic dataset was used because it guarantees that the images will test the specific noise removal and thresholding logic written for this project. A custom KNN classifier was implemented using NumPy so that the classification step could be kept simple and directly show how distance-based classification works. The 13 selected features include Hu Moments to help account for the random rotation of the shapes."),
        ("9. Implementation Details", "The pipeline executes in this order:\n1. Image Loading: The image is loaded via OpenCV.\n2. Grayscale Conversion: Color channels are removed.\n3. Gaussian Blur: A blur is applied to reduce the synthetic noise.\n4. Thresholding: Otsu's method isolates the darker shape from the lighter background.\n5. Morphological Processing: Closing and opening operations fill small holes and remove artifacts.\n6. Contour Extraction: The largest external contour is selected.\n7. Feature Extraction: The code calculates area, perimeter, circularity, aspect ratio, extent, solidity, and 7 Hu Moments.\n8. KNN Prediction: The 13 features are compared against the training set using Euclidean distance (K=5)."),
    ]
    
    for title, body in sections_2:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    pdf.chapter_title("10. Screenshots / Results")
    pdf.chapter_body("The model achieved 60.00% accuracy on the test set. The confusion matrix shows that circles and squares were frequently confused with each other. This indicates that the current feature set and segmentation process still have room for improvement. Below is the actual confusion matrix.")
    
    if os.path.exists('outputs/confusion_matrix.png'):
        pdf.image('outputs/confusion_matrix.png', w=120)
        pdf.ln(5)
        
    sections_3 = [
        ("11. Testing Approach", "The code is tested using pytest. The tests use simulated dummy inputs (like an empty image with a single drawn circle) to verify that the preprocessing, segmentation, feature extraction, and classification modules execute mathematically without crashing."),
        ("12. Challenges Faced", "One issue was that the Gaussian noise sometimes caused the contour detector to find extra, incorrect boundaries in the background. Morphological operations were adjusted to reduce these extra detections, and a filter was added to select only the largest contour. Additionally, keeping the KNN features balanced was a challenge because the raw area feature is much larger than the Hu Moments."),
        ("13. Learnings & Key Takeaways", "One thing I learned from the project was that segmentation directly affects the final classifier result. If the contour is incorrect due to remaining noise, the extracted features like area and perimeter are also incorrect, causing the KNN to fail."),
        ("14. Future Enhancements", "The most necessary enhancement is feature normalization. Because Euclidean distance is used, large numbers like contour area dominate the smaller Hu Moments. Standardizing the features before KNN prediction would likely improve the 60.00% accuracy significantly."),
        ("15. References", "1. OpenCV Documentation: https://docs.opencv.org/\n2. Python NumPy Documentation: https://numpy.org/doc/")
    ]
    
    for title, body in sections_3:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    pdf.output('report.pdf')
    print("Generated report.pdf")

if __name__ == '__main__':
    generate_report()
