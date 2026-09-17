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
        ("2. Introduction", "This project implements a classical Computer Vision pipeline to identify and classify geometric shapes (circles, squares, triangles). It avoids deep learning wrappers to emphasize fundamental image processing techniques like filtering, thresholding, and morphological operations."),
        ("3. Problem Statement", "The goal is to develop a system capable of identifying geometric shapes from noisy images using classical CV techniques, demonstrating the application of feature extraction and classical machine learning (KNN) in a modular pipeline."),
        ("4. Functional Requirements", "- Generate a synthetic dataset of shapes with noise.\n- Process images using grayscale conversion and Gaussian filtering.\n- Segment shapes using Otsu's thresholding.\n- Extract 13 geometric features.\n- Classify shapes using a custom K-Nearest Neighbors classifier.\n- Provide a CLI interface for training and testing."),
        ("5. Non-functional Requirements", "- Performance: Processing pipeline must execute rapidly on CPU.\n- Maintainability: Code must be modular and well-structured.\n- Reliability: Handling varied noise gracefully during segmentation.\n- Usability: Simple and intuitive command-line interface."),
    ]
    
    for title, body in sections:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    # Diagrams Section
    pdf.chapter_title("6. System Architecture & 7. Design Diagrams")
    pdf.chapter_body("The system consists of Preprocessing, Segmentation, Feature Extraction, and Classification modules. Below are the design diagrams.")
    
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
        ("8. Design Decisions & Rationale", "A synthetic dataset was used to ensure reproducibility and guarantee that the classical techniques (noise removal, thresholding) are explicitly required to achieve good performance. A custom KNN classifier was implemented because external ML libraries like scikit-learn were unavailable on the target 32-bit Python environment. 13 features (geometric + Hu Moments) were selected to provide scale and rotation invariance."),
        ("9. Implementation Details", "OpenCV was heavily utilized for image operations. Images are converted to grayscale and blurred. Otsu's thresholding is used to separate shapes from the background, followed by morphological closing and opening. Contours are found, and features like Circularity, Aspect Ratio, Extent, Solidity, and 7 Hu Moments are computed. These 13 features form the input to a custom K=5 KNN algorithm utilizing Euclidean distance."),
    ]
    
    for title, body in sections_2:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    pdf.chapter_title("10. Screenshots / Results")
    pdf.chapter_body("Evaluation Results:\nAccuracy: 61.11%\nPrecision (Triangles): 80.77%\nRecall (Triangles): 67.74%\n\nBelow is the confusion matrix generated from the test set evaluation.")
    
    if os.path.exists('outputs/confusion_matrix.png'):
        pdf.image('outputs/confusion_matrix.png', w=120)
        pdf.ln(5)
        
    sections_3 = [
        ("11. Testing Approach", "Automated tests were written using pytest to verify each module independently. The test_pipeline.py checks preprocessing, segmentation, feature extraction, and classification modules using simulated dummy inputs to ensure the mathematical operations function without runtime errors."),
        ("12. Challenges Faced", "The primary challenge was setting up a stable environment for classical machine learning on a 32-bit Python 3.14 installation. Due to the lack of precompiled scikit-learn wheels, a custom K-Nearest Neighbors algorithm had to be implemented from scratch in pure NumPy. Additionally, tuning the morphological operations to correctly segment the shapes despite Gaussian noise required careful adjustment of kernel sizes."),
        ("13. Learnings & Key Takeaways", "I learned that classical CV techniques require careful tuning of parameters (like kernel size and threshold methods) to work effectively. I also learned that feature engineering (like calculating Hu Moments for rotation invariance) is critical when relying on simpler classifiers like KNN instead of deep learning feature extractors."),
        ("14. Future Enhancements", "Future improvements could include implementing feature normalization before KNN classification, which would likely increase accuracy. Additionally, implementing an SVM classifier or integrating more advanced contour filtering logic could improve the robustness to heavy noise."),
        ("15. References", "1. OpenCV Documentation: https://docs.opencv.org/\n2. Python NumPy Documentation: https://numpy.org/doc/")
    ]
    
    for title, body in sections_3:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
        
    pdf.output('report.pdf')
    print("Generated report.pdf")

if __name__ == '__main__':
    generate_report()
