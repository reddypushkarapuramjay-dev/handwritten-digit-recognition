"""
Handwritten Digit Recognition - Flask Web Application

This is the main Flask application that serves the web interface and handles
digit prediction requests from drawn images or file uploads.

Run with: python app.py
"""

import os
import sys
import base64
import io
import json
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DigitRecognizer, create_model
from utils import (preprocess_base64_image, preprocess_image, validate_image_file, 
                   load_model_for_inference, get_device, set_seed)


# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = os.path.join('models', 'handwritten_digit_model.pth')

# Global model variable (loaded once at startup per worker)
model = None
device = None
model_loaded = False
model_error = None


def load_model_at_startup():
    """
    Load the trained model when the application starts.
    This avoids loading the model for every prediction request.
    Called at module import time so it works with both Flask dev server and Gunicorn.
    """
    global model, device, model_loaded, model_error
    
    try:
        set_seed(42)
        device = get_device()
        
        model_path = app.config['MODEL_PATH']
        
        # Check if model file exists
        if not os.path.exists(model_path):
            model_error = (f"Model file not found at '{model_path}'. "
                          f"Please train the model first by running: python train.py")
            print(f"ERROR: {model_error}")
            return False
        
        # Load model for inference
        model, checkpoint_info = load_model_for_inference(
            DigitRecognizer, 
            model_path, 
            device,
            input_size=784,
            hidden1_size=128,
            hidden2_size=64,
            output_size=10,
            dropout_rate=0.2
        )
        
        model_loaded = True
        print(f"Model loaded successfully from {model_path}")
        print(f"  Epoch: {checkpoint_info['epoch']}")
        print(f"  Accuracy: {checkpoint_info['accuracy']:.4f}")
        return True
        
    except Exception as e:
        model_error = f"Failed to load model: {str(e)}"
        print(f"ERROR: {model_error}")
        import traceback
        traceback.print_exc()
        return False


# Load model at module import time so it works with both
# Flask development server and Gunicorn workers.
# Each Gunicorn worker will load the model once at startup.
load_model_at_startup()


def predict_digit(image_tensor):
    """
    Run inference on a preprocessed image tensor.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image of shape (1, 784)
        
    Returns:
        dict: Prediction results including digit, confidence, and all probabilities
    """
    if not model_loaded or model is None:
        raise RuntimeError("Model not loaded")
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Get probabilities
    with torch.no_grad():
        probabilities = model.predict_proba(image_tensor)
        predicted_class = model.predict(image_tensor)
    
    # Convert to numpy
    probs = probabilities.cpu().numpy().flatten()
    pred_digit = int(predicted_class.item())
    confidence = float(probs[pred_digit] * 100)
    
    # Format all probabilities
    all_probs = {str(i): float(probs[i] * 100) for i in range(10)}
    
    return {
        'predicted_digit': pred_digit,
        'confidence': confidence,
        'probabilities': all_probs,
        'success': True
    }


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', 
                          model_loaded=model_loaded,
                          model_error=model_error)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle digit prediction from drawn canvas (base64 image).
    
    Expected JSON payload:
    {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    }
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': model_error or 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        base64_image = data['image']
        
        # Preprocess the base64 image
        image_tensor = preprocess_base64_image(base64_image, invert=True)
        
        # Run prediction
        result = predict_digit(image_tensor)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """
    Handle digit prediction from file upload.
    
    Expected form data:
    - file: image file (PNG, JPG, JPEG)
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': model_error or 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Validate file
        is_valid, error_msg, image = validate_image_file(file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Preprocess the image
        image_tensor = preprocess_image(image, invert=True)
        
        # Run prediction
        result = predict_digit(image_tensor)
        
        # Add image preview (base64 encoded)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        result['image_preview'] = f"data:image/png;base64,{img_base64}"
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Upload prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring."""
    return jsonify({
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'model_error': model_error,
        'device': str(device) if device else 'unknown'
    })


@app.route('/model-info')
def model_info():
    """Get model information."""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': model_error or 'Model not loaded'
        }), 500
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'success': True,
        'architecture': 'Fully Connected ANN',
        'layers': [
            {'name': 'Input', 'size': 784, 'activation': 'None'},
            {'name': 'Hidden 1', 'size': 128, 'activation': 'ReLU'},
            {'name': 'Hidden 2', 'size': 64, 'activation': 'ReLU'},
            {'name': 'Output', 'size': 10, 'activation': 'Softmax (via CrossEntropyLoss)'}
        ],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': [1, 784],
        'output_shape': [1, 10],
        'dataset': 'MNIST',
        'framework': 'PyTorch'
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16 MB.'
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again.'
    }), 500


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION - FLASK APP")
    print("=" * 60)
    
    # Model is already loaded at module import time (above)
    # This block only runs with Flask development server (not Gunicorn)
    
    if not model_loaded:
        print("\nWARNING: Model not loaded!")
        print(f"Error: {model_error}")
        print("Please run 'python train.py' first to train the model.")
        print("The web app will start but predictions will fail until model is trained.")
    else:
        print("\nModel loaded successfully!")
    
    print("\nStarting Flask development server...")
    # Use PORT environment variable (for Render compatibility) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Access the app at: http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run Flask app (development only)
    # For production deployment, use gunicorn instead:
    # gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app
    app.run(host='0.0.0.0', port=port, debug=True)