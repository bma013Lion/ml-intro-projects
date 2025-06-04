import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)

# Data Preprocessing
# Convert 'Sex' to numerical (0 for female, 1 for male)
le = LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])

# Feature Engineering
# Create new feature: Family Size
# Family size = Siblings/Spouses Aboard + Parents/Children Aboard + 1 (for the passenger)
titanic_df['Family_Size'] = titanic_df['Siblings/Spouses Aboard'] + titanic_df['Parents/Children Aboard'] + 1

# Create new feature: Is Alone
# Is Alone = 1 if Family Size is 1, 0 otherwise
titanic_df['Is_Alone'] = (titanic_df['Family_Size'] == 1).astype(int)

# Define features and target
X = titanic_df.drop(['Survived', 'Name'], axis=1)  # Drop 'Name' as it's not useful
y = titanic_df['Survived']

# Define different models to try
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Define different feature sets
feature_sets = {
    'All Features': ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 
                    'Parents/Children Aboard', 'Fare', 'Family_Size', 'Is_Alone'],
    'Top 4 Features': ['Sex', 'Fare', 'Age', 'Pclass'],
    'Simple Set': ['Sex', 'Pclass', 'Fare'],
    'Family Features': ['Sex', 'Pclass', 'Family_Size', 'Is_Alone']
}

# Function to evaluate models

# Parameters:
#   X: DataFrame of features
#   y: Series of target variable
#   models: dictionary of models to evaluate
#   feature_sets: dictionary of feature sets to evaluate
#   cv: number of folds for cross-validation

def eval_models(X, y, models, feature_sets, cv=5):
    """
    Evaluate different models using cross-validation with different feature sets.
    
    Parameters
    ----------
    X: DataFrame of features
    y: Series of target variable
    models: dictionary of models to evaluate
    feature_sets: dictionary of feature sets to evaluate
    cv: number of folds for cross-validation
    
    Returns
    -------
    DataFrame of results with columns:
        Model: name of model
        Feature Set: name of feature set
        Mean CV Accuracy: mean accuracy across folds
        Std Dev: standard deviation of accuracy across folds
    """
    results = []
    
    # Iterate over each model
    for model_name, model in models.items():
        # Iterate over each feature set
        for set_name, features in feature_sets.items():
            # Select Features
            X_sub = X[features]
            
            # Cross Validation: split data into multiple subsets and evaluate model on each subset
            # - This helps to avoid overfitting by averaging performance across multiple subsets
            cv_scores = cross_val_score(model, X_sub, y, cv=cv, scoring='accuracy')
            
            # Store Results
            results.append({
                'Model': model_name,
                'Feature Set': set_name,
                'Mean CV Accuracy': cv_scores.mean().round(4),
                'Std Dev': cv_scores.std().round(4)
            })
            
    return pd.DataFrame(results)
            
      
# Run evaluation
results_df = eval_models(X, y, models, feature_sets)

# Display comparison results 
print("\nModel Comparison:")
print("=" * 60)
print(results_df.sort_values(by='Mean CV Accuracy', ascending=False).to_string(index=False))
     

# Train best model on full training set
best_model_info = results_df.loc[results_df['Mean CV Accuracy'].idxmax()]
best_model = models[best_model_info['Model']]
best_features = feature_sets[best_model_info['Feature Set']]

# Print out the best model and its feature set
print("\n" + "=" * 60)
print("Best Model and Feature Set:")
print(f"Model: {best_model_info['Model']}")
print(f"Feature Set: {best_model_info['Feature Set']}")
# Cross-Validated Accuracy is the mean accuracy of the model
# across multiple subsets of the data
print(f"Cross-Validated Accuracy: {best_model_info['Mean CV Accuracy']:.4f}")

# Train best model
best_model.fit(X[best_features], y)

 
# Get feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    # Create DataFrame with feature names and importances
    importance = pd.DataFrame({
        'Feature': best_features,
        'Importance': best_model.feature_importances_
    })
    
    # Sort by importance in descending order
    importance = importance.sort_values('Importance', ascending=False)
    
    # Print feature importance
    print("\nFeature Importance:")
    print(importance)
    
    
    
def train_best_model():
    """Train and return the best model and its features based on the analysis."""
    # Load and preprocess data (same as before)
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    titanic_df = pd.read_csv(url)

    # Preprocessing
    le = LabelEncoder()
    titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])
    titanic_df['Family_Size'] = titanic_df['Siblings/Spouses Aboard'] + titanic_df['Parents/Children Aboard'] + 1
    titanic_df['Is_Alone'] = (titanic_df['Family_Size'] == 1).astype(int)

    # Define features and target
    X = titanic_df.drop(['Survived', 'Name'], axis=1)
    y = titanic_df['Survived']

    # Based on your output, the best model was Decision Tree with Simple Set features
    best_model = DecisionTreeClassifier(random_state=42)
    best_features = ['Sex', 'Pclass', 'Fare']  # Simple Set features

    # Train the model
    best_model.fit(X[best_features], y)
    
    return best_model, best_features