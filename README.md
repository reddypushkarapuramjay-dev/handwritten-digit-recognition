# Handwritten Digit Recognition Using Neural Network

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete, beginner-to-intermediate Machine Learning project that recognizes handwritten digits (0-9) using a custom Artificial Neural Network built with PyTorch, deployed as a web application with Flask.

## 🎯 Features

- **Draw & Predict**: Draw digits on an HTML5 canvas with mouse or touchscreen
- **Upload & Predict**: Upload image files (PNG, JPG, JPEG, GIF, BMP)
- **Real-time Predictions**: Get instant predictions with confidence scores
- **Probability Visualization**: See probability distribution for all 10 digits
- **Professional UI**: Modern, responsive design with Bootstrap 5
- **Educational Content**: Detailed explanations of neural network concepts
- **Model Evaluation**: Training curves, confusion matrices, and architecture diagrams

## 🏗️ Architecture

```
Input Layer (784) → Hidden Layer 1 (128, ReLU) → Hidden Layer 2 (64, ReLU) → Output Layer (10, Softmax)
```

- **Framework**: PyTorch 2.0+
- **Model**: Fully Connected ANN (~109K parameters)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Dataset**: MNIST (60K train, 10K test)
- **Accuracy**: ~98% on test set
- **Inference**: CPU-only (free hosting compatible)

## 📁 Project Structure

```
Handwritten_Digit_Recognition/
├── app.py                 # Flask web application
├── train.py               # Model training script
├── model.py               # Neural network definition
├── utils.py               # Utilities (data, preprocessing, evaluation)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── Procfile               # Deployment configuration
├── dataset/               # MNIST dataset (auto-downloaded)
├── models/
│   └── handwritten_digit_model.pth  # Trained model weights
├── graphs/                # Generated evaluation graphs
├── templates/
│   └── index.html         # Main HTML template
└── static/
    ├── css/
    │   └── style.css      # Custom styles
    ├── js/
    │   └── script.js      # Frontend JavaScript
    └── images/            # Static images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

```powershell
# 1. Navigate to project directory
cd C:\Users\Puramjay\OneDrive\Desktop\Handwritten_Digit_Recognition

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

### Training the Model

```powershell
# Train the neural network (takes ~2-5 minutes on CPU)
python train.py
```

This will:
- Download MNIST dataset automatically
- Train the model for 10 epochs
- Save the model to `models/handwritten_digit_model.pth`
- Generate evaluation graphs in `graphs/`

### Running the Web App

```powershell
# Start Flask development server
python app.py
```

Open your browser to **http://localhost:5000**

## 🎨 Using the Application

1. **Draw a Digit**: Use the canvas to draw a digit (0-9) with your mouse or finger
2. **Adjust Brush Size**: Use the slider to change brush thickness
3. **Predict**: Click "Predict Digit" to see the AI's prediction
4. **Or Upload**: Click the upload area to select an image file
5. **View Results**: See the predicted digit, confidence, and probability bars
6. **Try Again**: Click "Try Another" to clear and make a new prediction

## 📊 Model Evaluation

After training, the following graphs are generated in `graphs/`:

| Graph | Description |
|-------|-------------|
| `training_curves.png` | Training loss and accuracy over epochs |
| `confusion_matrix.png` | Test set confusion matrix |
| `confusion_matrix_normalized.png` | Normalized confusion matrix |
| `digit_predictions.png` | Sample predictions with true/predicted labels |
| `architecture_diagram.png` | Neural network architecture visualization |
| `probability_distribution.png` | Probability distribution for a sample prediction |

## 🧠 Neural Network Details

### Architecture

```python
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # Input → Hidden 1
        self.fc2 = nn.Linear(128, 64)    # Hidden 1 → Hidden 2
        self.fc3 = nn.Linear(64, 10)     # Hidden 2 → Output
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Logits (CrossEntropyLoss applies Softmax)
        return x
```

### Parameter Count
- Input → Hidden 1: 784 × 128 + 128 = **100,480**
- Hidden 1 → Hidden 2: 128 × 64 + 64 = **8,256**
- Hidden 2 → Output: 64 × 10 + 10 = **650**
- **Total: ~109,386 trainable parameters**

### Training Configuration
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-5
- **Dropout**: 0.2
- **LR Scheduler**: StepLR (step=5, gamma=0.5)

## 🌐 Deployment

### Render (Free Tier)

1. **Push to GitHub**:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/Handwritten_Digit_Recognition.git
   git push -u origin main
   ```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com) and create a free account
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `digit-recognizer` (or your choice)
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app`
   - Click "Create Web Service"

3. **Access**: Your app will be live at `https://your-app-name.onrender.com`

### Important Notes for Deployment
- The trained model (`models/handwritten_digit_model.pth`) must be in the repository
- Free tier spins down after 15 min inactivity (cold start ~30-60s)
- CPU-only inference (no GPU needed)
- 512 MB RAM limit on free tier (model is ~1 MB)

## 📦 Requirements

```
torch==2.3.0
torchvision==0.18.0
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
flask==3.0.2
gunicorn==21.2.0
pillow==10.2.0
python-dotenv==1.0.1
```

## 🔬 How It Works (Educational)

The application demonstrates these core deep learning concepts:

1. **Input Layer**: 784 neurons receive normalized pixel values (28×28 image)
2. **Hidden Layers**: Two fully connected layers with ReLU activation learn hierarchical features
3. **Neurons**: Basic computing units that apply weighted sum + bias + activation
4. **Weights & Biases**: ~109K learnable parameters adjusted during training
5. **ReLU Activation**: `max(0, x)` introduces non-linearity
6. **Forward Propagation**: Data flows from input to output through the network
7. **Loss Function**: CrossEntropyLoss measures prediction error
8. **Backpropagation**: Gradients flow backward to update weights via Adam optimizer
9. **Prediction**: Softmax converts logits to probabilities; highest probability wins

## 🖼️ Screenshots

### Home Page
![Home Page](graphs/home_screenshot.png)

### Drawing & Prediction
![Drawing](graphs/drawing_screenshot.png)

### How It Works Section
![How It Works](graphs/how_it_works_screenshot.png)

*Add your own screenshots to the `graphs/` folder and update the paths above.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **PyTorch Team**: For the excellent deep learning framework
- **Flask Team**: For the lightweight web framework
- **Bootstrap Team**: For the responsive CSS framework

## 📚 Learning Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

**Built with ❤️ for learning Deep Learning**

*This is an educational project. For production digit recognition, consider CNN-based architectures (LeNet-5, ResNet) which achieve >99% accuracy on MNIST.*