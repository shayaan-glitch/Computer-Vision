import os
from PIL import Image, ImageDraw, ImageFont

def create_box(draw, x, y, w, h, text):
    draw.rectangle([x, y, x+w, y+h], outline="black", width=2, fill="lightblue")
    draw.text((x + 10, y + h//2 - 5), text, fill="black")

def draw_arrow(draw, x1, y1, x2, y2):
    draw.line([x1, y1, x2, y2], fill="black", width=2)
    # Arrow head
    draw.polygon([(x2, y2), (x2-5, y2-5), (x2-5, y2+5)], fill="black")

def generate_architecture():
    img = Image.new('RGB', (800, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    create_box(draw, 50, 50, 100, 50, "Input Image")
    draw_arrow(draw, 150, 75, 200, 75)
    
    create_box(draw, 200, 50, 120, 50, "Preprocessing")
    draw_arrow(draw, 320, 75, 370, 75)
    
    create_box(draw, 370, 50, 120, 50, "Segmentation")
    draw_arrow(draw, 490, 75, 540, 75)
    
    create_box(draw, 540, 50, 120, 50, "Feature Ext.")
    draw_arrow(draw, 660, 75, 710, 75)
    
    create_box(draw, 710, 50, 80, 50, "KNN Class.")
    
    img.save("docs/images/architecture_diagram.png")

def generate_use_case():
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.text((50, 150), "User", fill="black")
    draw_arrow(draw, 80, 150, 150, 50)
    draw_arrow(draw, 80, 150, 150, 100)
    draw_arrow(draw, 80, 150, 150, 150)
    draw_arrow(draw, 80, 150, 150, 200)
    
    draw.ellipse([150, 30, 300, 70], outline="black")
    draw.text((180, 45), "Prepare Data", fill="black")
    
    draw.ellipse([150, 80, 300, 120], outline="black")
    draw.text((180, 95), "Train Model", fill="black")
    
    draw.ellipse([150, 130, 300, 170], outline="black")
    draw.text((180, 145), "Test Model", fill="black")
    
    draw.ellipse([150, 180, 300, 220], outline="black")
    draw.text((180, 195), "Predict Image", fill="black")
    
    img.save("docs/images/use_case_diagram.png")
    
def generate_class_diagram():
    img = Image.new('RGB', (400, 250), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 50, 350, 200], outline="black", width=2)
    draw.text((60, 60), "ShapeClassifier", fill="black")
    draw.line([50, 80, 350, 80], fill="black", width=1)
    draw.text((60, 90), "+ k: int\n+ X_train\n+ y_train\n+ classes", fill="black")
    draw.line([50, 150, 350, 150], fill="black", width=1)
    draw.text((60, 160), "+ train()\n+ predict()\n+ save()\n+ load()", fill="black")
    
    img.save("docs/images/class_diagram.png")
    
def generate_workflow():
    img = Image.new('RGB', (300, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    create_box(draw, 100, 20, 100, 40, "Start")
    draw_arrow(draw, 150, 60, 150, 90)
    
    create_box(draw, 100, 90, 100, 40, "Process Image")
    draw_arrow(draw, 150, 130, 150, 160)
    
    create_box(draw, 100, 160, 100, 40, "Extract Feat")
    draw_arrow(draw, 150, 200, 150, 230)
    
    create_box(draw, 100, 230, 100, 40, "Predict KNN")
    draw_arrow(draw, 150, 270, 150, 300)
    
    create_box(draw, 100, 300, 100, 40, "End")
    
    img.save("docs/images/workflow_diagram.png")

if __name__ == '__main__':
    os.makedirs('docs/images', exist_ok=True)
    generate_architecture()
    generate_use_case()
    generate_class_diagram()
    generate_workflow()
    print("Generated all diagrams via PIL.")
