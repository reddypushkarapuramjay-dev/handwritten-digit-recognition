"""
Handwritten Digit Recognition - Utility Functions

This module contains utility functions for:
- Data loading and preprocessing
- Image preprocessing for inference
- Model evaluation metrics
- Visualization helpers
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
import io
import base64
import os


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def get_data_loaders(batch_size=64, num_workers=0, data_dir='./dataset'):
    """
    Download and load MNIST dataset with train/test splits.
    
    Args:
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of subprocesses for data loading (0 for Windows)
        data_dir (str): Directory to store/download dataset
        
    Returns:
        tuple: (train_loader, test_loader, train_dataset, test_dataset)
    """
    # Define transforms: Convert to tensor and normalize
    # MNIST mean=0.1307, std=0.3081 (precomputed dataset statistics)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training dataset
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test dataset
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def get_inference_transform():
    """
    Get the transform pipeline for inference (preprocessing uploaded/drawn images).
    
    Returns:
        transforms.Compose: Transform pipeline matching training preprocessing
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((28, 28)),  # Resize to MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as training
    ])


def preprocess_image(image, invert=True):
    """
    Preprocess a PIL Image for model inference.
    
    Handles:
    - Grayscale conversion
    - Resize to 28x28
    - Color inversion (MNIST: white digit on black background)
    - Normalization
    - Flatten to 784 features
    
    Args:
        image (PIL.Image): Input image
        invert (bool): Whether to invert colors (default True for MNIST format)
        
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 784)
    """
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28 using high-quality resampling
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)
    
    # Invert colors if needed (MNIST has white digits on black background)
    # Most user drawings are black on white, so we invert
    if invert:
        img_array = 255.0 - img_array
    
    # Normalize to [0, 1] range
    img_array = img_array / 255.0
    
    # Apply MNIST normalization (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081
    
    # Flatten to 784 features and add batch dimension
    img_tensor = torch.from_numpy(img_array).view(1, -1).float()
    
    return img_tensor


def preprocess_base64_image(base64_string, invert=True):
    """
    Preprocess a base64 encoded image string for inference.
    
    Args:
        base64_string (str): Base64 encoded image (data URL or raw base64)
        invert (bool): Whether to invert colors
        
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 784)
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    return preprocess_image(image, invert=invert)


def validate_image_file(file):
    """
    Validate uploaded image file.
    
    Args:
        file: Flask FileStorage object
        
    Returns:
        tuple: (is_valid, error_message, pil_image)
    """
    # Check if file exists
    if not file or file.filename == '':
        return False, "No file selected", None
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}", None
    
    try:
        # Try to open and validate image
        image = Image.open(file.stream)
        image.verify()  # Verify it's a valid image
        file.stream.seek(0)  # Reset stream position
        image = Image.open(file.stream)  # Reopen for use
        return True, None, image
    except Exception as e:
        return False, f"Invalid image file: {str(e)}", None


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test dataset.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics including accuracy, predictions, targets
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    avg_loss = total_loss / len(test_loader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probabilities)
    }


def get_confusion_matrix(targets, predictions, normalize=False):
    """
    Compute confusion matrix.
    
    Args:
        targets (array-like): True labels
        predictions (array-like): Predicted labels
        normalize (bool): Whether to normalize by row (true class)
        
    Returns:
        np.ndarray: Confusion matrix
    """
    cm = confusion_matrix(targets, predictions)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


def get_classification_report(targets, predictions):
    """
    Generate detailed classification report.
    
    Args:
        targets (array-like): True labels
        predictions (array-like): Predicted labels
        
    Returns:
        str: Classification report
    """
    return classification_report(targets, predictions, digits=4)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_curves(train_losses, train_accuracies, test_accuracies=None, save_path=None):
    """
    Plot training loss and accuracy curves.
    
    Args:
        train_losses (list): Training loss per epoch
        train_accuracies (list): Training accuracy per epoch
        test_accuracies (list): Test accuracy per epoch (optional)
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    if test_accuracies:
        ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None, normalize=False):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Class labels
        save_path (str): Path to save the plot
        normalize (bool): Whether matrix is normalized
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    
    if normalize:
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_digit_predictions(images, targets, predictions, probabilities, num_samples=10, save_path=None):
    """
    Plot sample digit predictions with probabilities.
    
    Args:
        images (np.ndarray): Images (N, 28, 28) or (N, 784)
        targets (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels
        probabilities (np.ndarray): Prediction probabilities (N, 10)
        num_samples (int): Number of samples to display
        save_path (str): Path to save the plot
    """
    # Reshape images if flattened
    if images.ndim == 2 and images.shape[1] == 784:
        images = images.reshape(-1, 28, 28)
    
    num_samples = min(num_samples, len(images))
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Display image
        ax.imshow(images[i], cmap='gray')
        
        # Title with true and predicted labels
        true_label = targets[i]
        pred_label = predictions[i]
        confidence = probabilities[i][pred_label] * 100
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)', 
                     color=color, fontsize=11)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Digit predictions saved to {save_path}")
    
    plt.close()


def plot_probability_distribution(probabilities, true_label=None, save_path=None):
    """
    Plot probability distribution for a single prediction.
    
    Args:
        probabilities (np.ndarray): Probability array of shape (10,)
        true_label (int): True label (optional)
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 5))
    
    classes = list(range(10))
    colors = ['green' if i == np.argmax(probabilities) else 'skyblue' for i in classes]
    if true_label is not None:
        colors[true_label] = 'orange'
    
    bars = plt.bar(classes, probabilities * 100, color=colors, edgecolor='black', alpha=0.7)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        if height > 1:  # Only label if visible
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Digit Class', fontsize=12)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.title('Model Prediction Probabilities', fontsize=14)
    plt.xticks(classes)
    plt.ylim(0, max(probabilities * 100) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability distribution saved to {save_path}")
    
    plt.close()


def plot_architecture_diagram(save_path=None):
    """
    Create a visual diagram of the neural network architecture.
    
    Args:
        save_path (str): Path to save the diagram
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Layer positions and sizes
    layers = [
        {'name': 'Input Layer', 'neurons': 784, 'x': 1, 'y': 3, 'color': '#4CAF50', 'desc': '28×28 = 784 pixels'},
        {'name': 'Hidden Layer 1', 'neurons': 128, 'x': 3.5, 'y': 3, 'color': '#2196F3', 'desc': '128 neurons + ReLU'},
        {'name': 'Hidden Layer 2', 'neurons': 64, 'x': 6, 'y': 3, 'color': '#FF9800', 'desc': '64 neurons + ReLU'},
        {'name': 'Output Layer', 'neurons': 10, 'x': 8.5, 'y': 3, 'color': '#F44336', 'desc': '10 neurons (0-9)'}
    ]
    
    # Draw layers as circles
    for layer in layers:
        # Draw representative circles
        num_circles = min(layer['neurons'], 8)
        radius = 0.3
        spacing = 0.6
        start_y = layer['y'] - (num_circles - 1) * spacing / 2
        
        for i in range(num_circles):
            circle = plt.Circle((layer['x'], start_y + i * spacing), radius,
                              facecolor=layer['color'], edgecolor='black', alpha=0.7, linewidth=1)
            ax.add_patch(circle)
        
        # Add ellipsis if more neurons
        if layer['neurons'] > 8:
            ax.text(layer['x'], layer['y'] + 2, '⋮', ha='center', va='center', fontsize=20)
        
        # Layer label
        ax.text(layer['x'], layer['y'] - 1.8, layer['name'], ha='center', va='center',
                fontsize=12, fontweight='bold', color=layer['color'])
        ax.text(layer['x'], layer['y'] - 2.2, layer['desc'], ha='center', va='center',
                fontsize=9, color='gray')
    
    # Draw arrows between layers
    for i in range(len(layers) - 1):
        start_x = layers[i]['x'] + 0.5
        end_x = layers[i+1]['x'] - 0.5
        y = layers[i]['y']
        
        # Draw multiple arrows to show dense connections
        for offset in [-0.4, 0, 0.4]:
            ax.annotate('', xy=(end_x, y + offset), xytext=(start_x, y + offset),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
    
    # Add title
    ax.text(5, 5.5, 'Neural Network Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold')
    ax.text(5, 5.1, 'Fully Connected ANN for MNIST Digit Recognition', ha='center', va='center',
            fontsize=11, color='gray')
    
    # Add parameter counts
    param_text = ("Parameters:\n"
                  "• Input→Hidden1: 784×128 + 128 = 100,480\n"
                  "• Hidden1→Hidden2: 128×64 + 64 = 8,256\n"
                  "• Hidden2→Output: 64×10 + 10 = 650\n"
                  "• Total: ~109,386 trainable parameters")
    ax.text(5, 0.5, param_text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
    
    plt.close()


def save_all_graphs(train_losses, train_accuracies, test_accuracies, 
                    targets, predictions, probabilities, images, graphs_dir='./graphs'):
    """
    Generate and save all evaluation graphs.
    
    Args:
        train_losses (list): Training losses per epoch
        train_accuracies (list): Training accuracies per epoch
        test_accuracies (list): Test accuracies per epoch
        targets (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels
        probabilities (np.ndarray): Prediction probabilities
        images (np.ndarray): Test images
        graphs_dir (str): Directory to save graphs
    """
    os.makedirs(graphs_dir, exist_ok=True)
    
    print("\nGenerating evaluation graphs...")
    
    # 1. Training curves
    plot_training_curves(train_losses, train_accuracies, test_accuracies,
                        save_path=os.path.join(graphs_dir, 'training_curves.png'))
    
    # 2. Confusion matrix
    cm = get_confusion_matrix(targets, predictions)
    plot_confusion_matrix(cm, save_path=os.path.join(graphs_dir, 'confusion_matrix.png'))
    
    # 3. Normalized confusion matrix
    cm_norm = get_confusion_matrix(targets, predictions, normalize=True)
    plot_confusion_matrix(cm_norm, save_path=os.path.join(graphs_dir, 'confusion_matrix_normalized.png'),
                         normalize=True)
    
    # 4. Digit prediction examples
    plot_digit_predictions(images, targets, predictions, probabilities, 
                          num_samples=15, save_path=os.path.join(graphs_dir, 'digit_predictions.png'))
    
    # 5. Architecture diagram
    plot_architecture_diagram(save_path=os.path.join(graphs_dir, 'architecture_diagram.png'))
    
    # 6. Sample probability distribution (for first test sample)
    plot_probability_distribution(probabilities[0], targets[0],
                                 save_path=os.path.join(graphs_dir, 'probability_distribution.png'))
    
    print(f"All graphs saved to {graphs_dir}/")


# ============================================================
# MODEL SAVING/LOADING
# ============================================================

def save_model(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        accuracy (float): Current accuracy
        filepath (str): Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device='cpu', optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): Model instance to load weights into
        filepath (str): Path to checkpoint
        device (str): Device to load model on
        optimizer (Optimizer): Optional optimizer to load state into
        
    Returns:
        dict: Checkpoint info (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0),
        'accuracy': checkpoint.get('accuracy', 0)
    }


def load_model_for_inference(model_class, filepath, device='cpu', **model_kwargs):
    """
    Load model for inference (creates model instance and loads weights).
    
    Args:
        model_class: Model class (e.g., DigitRecognizer)
        filepath (str): Path to checkpoint
        device (str): Device to load model on
        **model_kwargs: Arguments for model constructor
        
    Returns:
        tuple: (model, checkpoint_info)
    """
    model = model_class(**model_kwargs)
    checkpoint_info = load_model(model, filepath, device)
    model.eval()
    return model, checkpoint_info


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """
    Get the best available device (CPU for free hosting compatibility).
    
    Returns:
        torch.device: Device to use
    """
    # Force CPU for free hosting compatibility
    device = torch.device('cpu')
    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test data loading
    train_loader, test_loader, train_ds, test_ds = get_data_loaders(batch_size=32, data_dir='./dataset')
    
    # Test preprocessing
    sample_image, sample_label = train_ds[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample label: {sample_label}")
    
    # Test seed setting
    set_seed(42)
    
    # Test device
    device = get_device()
    
    print("Utilities test passed!")