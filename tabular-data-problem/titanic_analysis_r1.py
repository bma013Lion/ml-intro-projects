# Titanic Survival Prediction: Machine Learning Analysis

# Purpose: Pandas is the go-to library for 
#          data manipulation and analysis in Python.
# Usage: We'll use it to load, clean, and explore the Titanic dataset, 
#        and to create DataFrames for easy data handling.
import pandas as pd

# Purpose: NumPy is a fundamental library for scientific computing in Python.
# Usage: We'll use it for numerical operations and array manipulations.
import numpy as np

# Purpose: scikit-learn is a popular machine learning library in Python.
# Usage: We'll use it for building and evaluating machine learning models.
from sklearn.model_selection import train_test_split

# Purpose: LabelEncoder is a class in scikit-learn for encoding 
#          categorical variables.
# Usage: We'll use it to convert categorical variables into numeric format.
from sklearn.preprocessing import LabelEncoder

# Purpose: RandomForestClassifier is a class in scikit-learn for building 
#          random forest models.
# Usage: We'll use it to build a random forest classifier for the 
#        Titanic dataset.
from sklearn.ensemble import RandomForestClassifier

# Purpose: accuracy_score and classification_report are functions in 
#          scikit-learn for evaluating machine learning models.
# Usage: We'll use them to evaluate the performance of our random 
#        forest classifier.
from sklearn.metrics import accuracy_score, classification_report

# Purpose: matplotlib is a popular library for data visualization in Python.
# Usage: We'll use it to create visualizations of our data and models.
import matplotlib.pyplot as plt



# Loading in the Dataset
data_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(data_url)

# Displaying the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(titanic_df.head())

# Displaying dataset info
print("\nDataset info:")
print(titanic_df.info())

# Checking for missing values
print("\nMissing values:")
print(titanic_df.isnull().sum())


# Data Processing

# Handling Missing Values 
# though no missing values were found in this dataset)

# Fill missing values with median
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)

# No need to encode these as they're already numeric:
# - Survived (0/1)
# - Pclass (1/2/3)
# - Siblings/Spouses Aboard (count)
# - Parents/Children Aboard (count)

# Feature Engineering

# Create a new feature: Family Size
# Family size = Siblings/Spouses Aboard + Parents/Children Aboard + 1 (for the passenger)
titanic_df['Family Size'] = titanic_df['Siblings/Spouses Aboard'] + titanic_df['Parents/Children Aboard'] + 1

# Create a new feature: Is Alone
# Is Alone = 1 if Family Size is 1, 0 otherwise
titanic_df['Is Alone'] = np.where(titanic_df['Family Size'] == 1, 1, 0)

# Convert only the necessary categorical variable to numeric
le = LabelEncoder()
titanic_df['Sex'] = le.fit_transform(titanic_df['Sex'])  # Converts to 0/1

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare', 'Family Size', 'Is Alone']
target = 'Survived'

# Split the dataset into training and testing sets
X = titanic_df[features]
y = titanic_df[target]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)



# Evaluate the model

# Calculate accuracy
print("\nModel Evaluation:")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print classification report
print("\nClassification Report:")
print("=" * 50)
print(classification_report(y_test, y_pred))

# Plot feature importances
#  - Feature importances are a measure of how important each feature is to the model.
#  - Features with higher importances are more important for making predictions.
#  - Feature importances are often used for feature selection, which is the process
#    of selecting which features to use in a model.

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print("=" * 50)
print(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

# Sample Prediction
print("\nSample Predictions (First 5 test samples):")
print("=" * 50)
sample_results = X_test.head().copy()
sample_results['Actual'] = y_test.head().values
sample_results['Predicted'] = y_pred[:5]
print(sample_results[['Actual', 'Predicted']])
