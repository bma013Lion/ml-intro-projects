import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Generate synthetic data
def generate_medical_data(n_samples=1000):
    """Generate a synthetic medical diagnosis dataset."""
    # Create an imbalanced dataset (5% positive cases - rare disease)
    # 
    # make_classification: to generate a synthetic dataset 
    #                      for a binary classification problem
    # Parameters:
    #   - n_samples: number of samples to generate
    #   - n_features: number of features to generate
    #   - n_informative: number of informative features
    #   - n_redundant: number of redundant features
    #   - weights: class distribution
    #   - random_state: random seed for reproducibility 
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        weights=[0.95, 0.05],  # 5% positive class
        random_state=42
    )
    
    # Add some noise to make it more realistic
    
    # np.random.normal: to generate random numbers from a normal distribution
    # Parameters:
    #   - loc: mean of the distribution
    #   - scale: standard deviation of the distribution
    #   - size: shape of the output array
    
    # Why add noise?
    # - Realism: Real-world data is rarely perfect. 
    #           This simulates measurement errors or natural variations.
    # - Prevents Overfitting: It makes the data less "clean," helping models 
    #                        generalize better to real-world scenarios.
    # - Robustness Testing: Ensures the model doesn't rely too heavily on exact values.
    X += np.random.normal(0, 0.5, X.shape)
    
    return X, y

def eval_model(y_true, y_pred, y_pred_prob=None, model_name="Model"):
    """
    Evaluate a model with given ground truth labels and predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (1 = disease, 0 = healthy)
    y_pred : array-like
        Predicted labels (1 = disease, 0 = healthy)
    y_pred_prob : array-like, optional
        Predicted probabilities for the positive class (disease)
    model_name : str, optional
        Model name for display purposes (default is "Model")

    Returns
    -------
    None

    Notes
    -----
    This function prints out various evaluation metrics and plots the
    confusion matrix and ROC curve (if y_pred_prob is provided).
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Evalutaion")
    print('='*50)

    # metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # print metrics
    print("\nBasic Metrics:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification Report:
    # A classification report is a summary of the main classification metrics
    # for each label. It provides the precision, recall, f1 score and support
    # for each class. The precision is the ratio of true positives to the sum
    # of true positives and false positives. The recall is the ratio of true
    # positives to the sum of true positives and false negatives. The f1 score
    # is the harmonic mean of precision and recall. The support is the number
    # of actual occurrences of each class in the specified target.
    
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Disease']))
    
    # confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 50)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Healthy', 'Predicted Disease'],
                yticklabels=['Actual Healthy', 'Actual Disease'])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    if y_pred_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # print ROC AUC
        print("\nROC AUC:")
        print("-" * 50)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{model_name} - ROC Curve")
        plt.tight_layout()
        plt.savefig(f'{model_name}_roc_curve.png')
        plt.close()
        
def main():
    #set random seed
    np.random.seed(42)
    
    # generate data
    print("Generating synthetic medical diagnosis dataset...")
    X, y = generate_medical_data(2000)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    # Probabilities for positive class
    y_prob = model.predict_proba(X_test)[:, 1] 
    
    # Evaluate the model
    eval_model(y_test, y_pred, y_pred_prob=y_prob, model_name="Random Forest")
    
    # Demonstrate the impact of threshold adjustment
    print("\n" + "="*50)
    print("Impact of Threshold Adjustment")
    print("="*50)
    
    # Try different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        y_pred_adj = (y_prob >= threshold).astype(int)
        print(f"\nThreshold: {threshold:.1f}")
        print(f"Accuracy:  {accuracy_score(y_test, y_pred_adj):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred_adj, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y_test, y_pred_adj):.4f}")
        print(f"F1 Score:  {f1_score(y_test, y_pred_adj, zero_division=0):.4f}")
    
    print("\nEvaluation complete! Check the generated plots:")
    print("- confusion_matrix.png: Shows true/false positives/negatives")
    print("- roc_curve.png: Shows the ROC curve with AUC score")


if __name__ == "__main__":
    main()