"""
Train a CNN model on the CIFAR10 dataset with early stopping and save the learning curves.

This script trains a CNN model on the CIFAR10 dataset for a specified number of epochs.
It uses early stopping to stop training when the validation loss does not improve for a
specified number of epochs. The learning curves are saved as PNG images in the
"outputs/learning_curves" directory.

The script also prints the final training and validation accuracy and loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os

from models.cnn import CIFAR10CNN
from utils.data_loader import get_data_loaders
from utils.visualization import plot_learning_curves


# Set the random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set the hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 30
patience = 5

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the output directories
os.makedirs("outputs/learning_curves", exist_ok=True)
os.makedirs("outputs/confusion_matrix", exist_ok=True)
os.makedirs("outputs/misclassified", exist_ok=True)

def train_model():
    """
    Train the model and return the trained model and the learning curves.
    """
    # Get the data loaders
    train_loader, val_loader, _, class_names = get_data_loaders(batch_size=batch_size)   
    
    # Initialize the model, loss function, and optimizer
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, factor=0.1, verbose=True)
    
    # Lists to store the metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float("inf")
    patience_counter = 0
     
    # Training loop
    for epoch in range(num_epochs):
        # Training phase 
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, labels in train_loop:
            images , labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loop.set_postfix({"Loss": loss.item()})
            
        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save learning curves
        plot_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # Early stopping
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return model, train_losses, val_losses, train_accuracies, val_accuracies 
    
    
    
def evaluate(model, data_loader, criterion):
    """
    Evaluate the model on the given data loader and return the loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            
    final_loss = running_loss / total
    final_acc = 100 * correct / total
    return final_loss, final_acc


if __name__ == "__main__":
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model()
    print("\nTraining Complete.")