"""
Handwritten Digit Recognition - Model Training Script

This script trains the neural network on the MNIST dataset.
Run this script to train the model and generate evaluation graphs.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DigitRecognizer, create_model, count_parameters, get_model_summary
from utils import (get_data_loaders, get_device, set_seed, evaluate_model, 
                   save_model, save_all_graphs, plot_training_curves)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Reproducibility
SEED = 42

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
WEIGHT_DECAY = 1e-5  # L2 regularization

# Model architecture
INPUT_SIZE = 784
HIDDEN1_SIZE = 128
HIDDEN2_SIZE = 64
OUTPUT_SIZE = 10
DROPOUT_RATE = 0.2

# Paths
DATA_DIR = './dataset'
MODEL_DIR = './models'
GRAPHS_DIR = './graphs'
MODEL_PATH = os.path.join(MODEL_DIR, 'handwritten_digit_model.pth')


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, test_loader, criterion, device):
    """
    Validate model on test set.
    
    Args:
        model (nn.Module): Model to validate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device (torch.device): Device to validate on
        
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training function."""
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Get device (CPU for free hosting)
    device = get_device()
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,  # 0 for Windows compatibility
        data_dir=DATA_DIR
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        input_size=INPUT_SIZE,
        hidden1_size=HIDDEN1_SIZE,
        hidden2_size=HIDDEN2_SIZE,
        output_size=OUTPUT_SIZE,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)
    
    # Print model summary
    get_model_summary(model)
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch:2d}/{EPOCHS} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")
    save_model(model, optimizer, EPOCHS, train_losses[-1], train_accuracies[-1] / 100, MODEL_PATH)
    
    # Detailed evaluation
    print("\nPerforming detailed evaluation...")
    eval_results = evaluate_model(model, test_loader, device)
    
    print(f"\nDetailed Test Results:")
    print(f"  Test Accuracy: {eval_results['accuracy'] * 100:.2f}%")
    print(f"  Test Loss: {eval_results['loss']:.4f}")
    
    # Print classification report
    from sklearn.metrics import classification_report
    report = classification_report(eval_results['targets'], eval_results['predictions'], digits=4)
    print(f"\nClassification Report:\n{report}")
    
    # Generate and save all graphs
    print("\nGenerating evaluation graphs...")
    
    # Get some test images for visualization
    test_images = []
    test_labels = []
    for data, target in test_loader:
        test_images.append(data.cpu().numpy())
        test_labels.append(target.cpu().numpy())
        if len(test_images) * BATCH_SIZE >= 15:
            break
    test_images = np.vstack(test_images)[:15].reshape(-1, 28, 28)
    test_labels = np.hstack(test_labels)[:15]
    test_preds = eval_results['predictions'][:15]
    test_probs = eval_results['probabilities'][:15]
    
    save_all_graphs(
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        targets=eval_results['targets'],
        predictions=eval_results['predictions'],
        probabilities=eval_results['probabilities'],
        images=test_images,
        graphs_dir=GRAPHS_DIR
    )
    
    # For digit predictions graph, we need some test images
    # Get a batch of test images
    test_images = []
    test_labels = []
    for data, target in test_loader:
        test_images.append(data.cpu().numpy())
        test_labels.append(target.cpu().numpy())
        if len(test_images) * BATCH_SIZE >= 15:
            break
    test_images = np.vstack(test_images)[:15].reshape(-1, 28, 28)
    test_labels = np.hstack(test_labels)[:15]
    test_preds = eval_results['predictions'][:15]
    test_probs = eval_results['probabilities'][:15]
    
    from utils import plot_digit_predictions
    plot_digit_predictions(test_images, test_labels, test_preds, test_probs,
                          num_samples=15, save_path=os.path.join(GRAPHS_DIR, 'digit_predictions.png'))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Graphs saved to: {GRAPHS_DIR}/")
    print(f"\nTo run the web app: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()