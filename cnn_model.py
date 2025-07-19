import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle

# Set parameters
img_height, img_width = 32, 32  # Smaller for faster processing
num_classes = 4  # Defender, Midfielder, Forward, Goalkeeper

data_dir = 'sample_images'

def load_and_preprocess_images():
    """Load images from folders and convert to feature vectors"""
    features = []
    labels = []
    class_names = ['Defender', 'Midfielder', 'Forward', 'Goalkeeper']
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping {class_name}.")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(class_dir, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((img_height, img_width))
                    # Convert to grayscale and flatten
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray).flatten() / 255.0
                    
                    features.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return np.array(features), np.array(labels), class_names

def train_model():
    """Train the MLP classifier"""
    print("Loading and preprocessing images...")
    features, labels, class_names = load_and_preprocess_images()
    
    if len(features) == 0:
        print("No images found! Please add images to the sample_images folders.")
        return None, None, None
    
    print(f"Loaded {len(features)} images from {len(set(labels))} classes")
    
    # Split data (simple approach - use last 20% for validation)
    split_idx = int(0.8 * len(features))
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP classifier
    print("Training model...")
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Save model and scaler
    with open('cnn_model.pkl', 'wb') as f:
        pickle.dump((model, scaler, class_names), f)
    
    print("Model saved as cnn_model.pkl")
    return model, scaler, class_names

if __name__ == "__main__":
    train_model() 