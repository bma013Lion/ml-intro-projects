import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
import os

from models.cnn import CIFAR10CNN
from utils.data_loader import get_data_loaders
from utils.visualization import plot_confusion_matrix, visualize_misclassified


# initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the model
def evaluate_model(model_path='outputs/best_model.pth'):
    # get the data loaders
    _, _, test_loader, class_names = get_data_loaders(batch_size=64)
    
    # initialize the model
    model = CIFAR10CNN().to(device)
    
    # load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    # initialize variables
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_true= []
    misclassified_preds = []
    confidences = []
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # evaluate loop
    with torch.no_grad():
       for images, labels in tqdm(test_loader, desc="Evaluating"):
           images, labels = images.to(device), labels.to(device)
           
           # forward pass
           outputs = model(images)
           loss = criterion(outputs, labels)
           
           # update metrics
           _, predicted = torch.max(outputs.data, 1)
           probs = torch.softmax(outputs, dim=1)
           confidence, _ = torch.max(probs, 1)
           
           # update statistics
           running_loss += loss.item() * images.size(0)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
           
           # store predictions and labels
           all_preds.extend(predicted.cpu().numpy())
           all_labels.extend(labels.cpu().numpy())
            
           # find misclassified examples
           mask = predited != labels
           for i in range(images.size(0)):
               if mask[i]:
                   misclassified_images.append(images[i].cpu().numpy())
                   misclassified_true.append(labels[i].item())
                   misclassified_preds.append(predicted[i].item())
                   confidences.append(confidence[i].item())
                   
                   
    # calculate final metrics
    total_loss = running_loss / total
    total_acc = 100 * correct / total       

    print(f"\nTest Loss: {total_loss:.4f}, Test Accuracy: {total_acc:.2f}%")
    
    # generate visualizations
    plot_confusion_matrix(all_preds, all_labels, class_names)
    
    # visualize mislassified examples (up to 10 examples
    if misclassified_images:
        visualize_misclassified(
            misclassified_images[:10], 
            misclassified_true[:10],
            misclassified_preds[:10], 
            confidences[:10], 
            class_names
        )
    
    return total_loss, total_acc

if __name__ == "__main__":
    evaluate_model()
    
        
    