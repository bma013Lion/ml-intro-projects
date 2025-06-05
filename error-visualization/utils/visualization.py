import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir='outputs/learning_curves'):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves.png')
    plt.close()
    
def plot_confusion_matrix(true_labels, pred_labels, class_names, save_dir='outputs/confusion_matrix'): 
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
   
def visualiz_misclassified(images, true_labels, pred_labels, confidence, class_names, save_dir='outputs/misclassified', num_examples=10):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    for i in range(min(num_examples, len(images))):
        plt.subplot(2, 5, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5   
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f'True: {class_names[true_labels[i]]}\n'
                  f'Predicted: {class_names[pred_labels[i]]}\n'
                  f'Confidence: {confidence[i]:.2f}',
                  fontsize=10, color='red')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f'{save_dir}/misclassified_examples.png')
    plt.close()
    
        
       
      

    