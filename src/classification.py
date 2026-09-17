import numpy as np
import pickle

class ShapeClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes = ["circle", "square", "triangle"]
        
    def train(self, features, labels):
        self.X_train = np.array(features, dtype=np.float32)
        # Store labels as strings directly for simplicity
        self.y_train = np.array(labels)
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
        
    def predict(self, feature_vector):
        if self.X_train is None:
            raise ValueError("Model not trained.")
            
        x = np.array(feature_vector, dtype=np.float32)
        
        # Calculate distances to all training points
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common label
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
        
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'X_train': self.X_train, 'y_train': self.y_train, 'k': self.k}, f)
            
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.k = data['k']
