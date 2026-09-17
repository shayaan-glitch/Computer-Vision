import argparse
import os
import cv2
from src.pipeline import process_image, load_dataset
from src.classification import ShapeClassifier
from src.evaluation import evaluate_and_report

def main():
    parser = argparse.ArgumentParser(description="Computer Vision Shape Analysis")
    parser.add_argument("--train", type=str, help="Path to training data directory")
    parser.add_argument("--test", type=str, help="Path to test data directory")
    parser.add_argument("--predict", type=str, help="Path to single image to predict")
    parser.add_argument("--model-path", type=str, default="outputs/model.pkl", help="Path to save/load model")
    parser.add_argument("--output", type=str, default="outputs/", help="Directory to save outputs")
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    classifier = ShapeClassifier()
    
    if args.train:
        print(f"Loading training data from {args.train}...")
        try:
            X_train, y_train = load_dataset(args.train)
            if not X_train:
                print("Error: No valid training data found.")
                return
            print(f"Training on {len(X_train)} samples...")
            classifier.train(X_train, y_train)
            classifier.save(args.model_path)
            print(f"Model saved to {args.model_path}")
        except Exception as e:
            print(f"Error during training: {e}")
            
    if args.test:
        print(f"Loading test data from {args.test}...")
        try:
            if not os.path.exists(args.model_path):
                print(f"Error: Model not found at {args.model_path}. Train first.")
                return
            classifier.load(args.model_path)
            
            X_test, y_test = load_dataset(args.test)
            if not X_test:
                print("Error: No valid test data found.")
                return
                
            y_pred = [classifier.predict(x) for x in X_test]
            evaluate_and_report(y_test, y_pred, classifier.classes, args.output)
        except Exception as e:
            print(f"Error during testing: {e}")
            
    if args.predict:
        try:
            if not os.path.exists(args.model_path):
                print(f"Error: Model not found at {args.model_path}. Train first.")
                return
            classifier.load(args.model_path)
            
            feat, img, segmented, contour = process_image(args.predict)
            if feat is not None:
                prediction = classifier.predict(feat)
                print(f"Prediction for {args.predict}: {prediction}")
                
                # Draw prediction on image
                cv2.putText(img, f"Pred: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if contour is not None:
                    cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)
                    
                out_path = os.path.join(args.output, f"pred_{os.path.basename(args.predict)}")
                cv2.imwrite(out_path, img)
                print(f"Saved visualization to {out_path}")
            else:
                print("Error: Could not extract features from image.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    if not any([args.train, args.test, args.predict]):
        parser.print_help()

if __name__ == "__main__":
    main()
