"""
Handwritten Digit Recognition - Neural Network Model Definition

This module defines the Artificial Neural Network (ANN) architecture for
handwritten digit recognition using PyTorch.

Architecture:
- Input Layer: 784 neurons (28x28 flattened pixels)
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9) with logits (used with CrossEntropyLoss)

The model uses:
- ReLU activation for hidden layers
- CrossEntropyLoss (includes Softmax internally)
- Adam optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitRecognizer(nn.Module):
    """
    A simple fully connected Artificial Neural Network for MNIST digit classification.
    
    Architecture:
    - Input: 784 features (28x28 flattened)
    - Hidden 1: 128 neurons + ReLU
    - Hidden 2: 64 neurons + ReLU
    - Output: 10 neurons (logits for 10 classes)
    """
    
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10, dropout_rate=0.2):
        """
        Initialize the neural network layers.
        
        Args:
            input_size (int): Number of input features (784 for 28x28 images)
            hidden1_size (int): Neurons in first hidden layer (128)
            hidden2_size (int): Neurons in second hidden layer (64)
            output_size (int): Number of output classes (10 for digits 0-9)
            dropout_rate (float): Dropout probability for regularization
        """
        super(DigitRecognizer, self).__init__()
        
        # First fully connected layer: 784 -> 128
        self.fc1 = nn.Linear(input_size, hidden1_size)
        # Second fully connected layer: 128 -> 64
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        # Output layer: 64 -> 10
        self.fc3 = nn.Linear(hidden2_size, output_size)
        
        # Dropout layer for regularization (prevents overfitting)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 784)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10)
        """
        # Flatten input if it's not already flat (batch_size, 1, 28, 28) -> (batch_size, 784)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # First hidden layer: Linear -> ReLU -> Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second hidden layer: Linear -> ReLU -> Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer: Linear (logits, no activation - CrossEntropyLoss applies Softmax)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x):
        """
        Get prediction probabilities for input.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Probabilities for each class (after softmax)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x):
        """
        Get predicted class for input.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            _, predicted = torch.max(logits, 1)
        return predicted


def create_model(input_size=784, hidden1_size=128, hidden2_size=64, output_size=10, dropout_rate=0.2):
    """
    Factory function to create a DigitRecognizer model.
    
    Args:
        input_size (int): Input features count
        hidden1_size (int): First hidden layer size
        hidden2_size (int): Second hidden layer size
        output_size (int): Output classes count
        dropout_rate (float): Dropout rate
        
    Returns:
        DigitRecognizer: Initialized model
    """
    return DigitRecognizer(input_size, hidden1_size, hidden2_size, output_size, dropout_rate)


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model, input_size=(1, 784)):
    """
    Print a summary of the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input tensor shape
    """
    print("=" * 60)
    print("NEURAL NETWORK ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<20} {'Type':<15} {'Output Shape':<15} {'Params':<10}")
    print("-" * 60)
    
    total_params = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            params = in_features * out_features + out_features  # weights + bias
            total_params += params
            print(f"{name:<20} {'Linear':<15} {f'({out_features})':<15} {params:<10}")
        elif isinstance(module, nn.Dropout):
            print(f"{name:<20} {'Dropout':<15} {'-':<15} {'0':<10}")
        elif isinstance(module, nn.ReLU):
            print(f"{name:<20} {'ReLU':<15} {'-':<15} {'0':<10}")
    
    print("-" * 60)
    print(f"{'Total Parameters:':<35} {total_params:<10}")
    print(f"{'Trainable Parameters:':<35} {sum(p.numel() for p in model.parameters() if p.requires_grad):<10}")
    print("=" * 60)


if __name__ == "__main__":
    # Test model creation and summary
    model = create_model()
    get_model_summary(model)
    
    # Test forward pass
    test_input = torch.randn(1, 784)
    output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Test probabilities
    probs = model.predict_proba(test_input)
    print(f"Probabilities: {probs}")
    print(f"Sum of probabilities: {probs.sum().item():.4f}")
    
    # Test prediction
    pred = model.predict(test_input)
    print(f"Predicted class: {pred.item()}")