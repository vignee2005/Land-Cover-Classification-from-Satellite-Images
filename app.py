# app.py
import os
import numpy as np
from flask import Flask, render_template, url_for
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\vigne\OneDrive\Desktop\aiml_pro\flask_app\cnn_model.h5'
model = load_model(model_path)

# Ensure the model is compiled
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the class names used in your model
class_names = ['Forest', 'Highway', 'Industrial', 'Pasture', 'Residential', 'River', 'SeaLake']

# Define image paths to classify
image_paths_to_classify = [
    'static/uploads/F1.jpg',
    'static/uploads/Ind1.jpg',
    'static/uploads/Res1.jpg',
    'static/uploads/River1.jpg',
    'static/uploads/Highway1.jpg',
    'static/uploads/Highway2.jpg',
    'static/uploads/Pasture1.jpg',
    'static/uploads/Res2.jpg',
    'static/uploads/River2.jpg',
    'static/uploads/F2.jpg',
    'static/uploads/SeaLake.jpg',
    'static/uploads/Ind2.jpg'
]

# Function to preprocess a single image
def preprocess_image(image_path, image_size=(64, 64)):
    try:
        print(f"Loading image from path: {image_path}")  # Debug statement
        image = Image.open(image_path).convert('RGB')
        image = image.resize(image_size)
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        print(f"Error loading image: {image_path}. Error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html', image_paths=image_paths_to_classify)

@app.route('/classify/<path:image_path>')
def classify_image(image_path):
    full_image_path = os.path.join('static', 'uploads', image_path)
    print(f"Classifying image at full path: {full_image_path}")  # Debug statement
    image = preprocess_image(full_image_path)
    if image is None:
        return f"Failed to classify image '{full_image_path}'. Please check the image path."
    
    # Reshape and expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)
    
    # Predict class probabilities
    predictions = model.predict(image)
    
    # Get predicted class index
    predicted_class_index = np.argmax(predictions)
    
    # Map index to class name
    predicted_class_name = class_names[predicted_class_index]
    
    return render_template('classify.html', image_path=image_path, predicted_class=predicted_class_name)

if __name__ == '__main__':
    app.run(debug=True)
