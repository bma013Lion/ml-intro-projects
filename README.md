# Introduction to Machine Learning Projects

This repository contains a collection of introductory machine learning projects, each designed to demonstrate fundamental ML concepts and techniques.

## Projects

### 1. Evaluation Metrics for Medical Diagnosis

This project demonstrates how to evaluate machine learning models for medical diagnosis, particularly focusing on imbalanced datasets where positive cases (diseases) are rare. It includes data generation, model training, and comprehensive evaluation metrics.

#### Data Generation
- **Synthetic Data**: Generates realistic medical data with configurable parameters
- **Class Imbalance**: Simulates a real-world scenario where only 5% of patients have the disease (positive cases)
- **Features**: Each patient record includes 10 measurements, with 8 informative features and 2 redundant ones for added realism

#### Model Training
- **Algorithm**: Uses Random Forest Classifier, which combines multiple decision trees for robust predictions
- **Data Splitting**: 70% of data for training, 30% for testing to evaluate model generalization
- **Handling Imbalance**: Demonstrates techniques to handle the imbalanced nature of medical data

#### Evaluation Metrics

**Basic Metrics:**
- **Accuracy**: Overall prediction correctness (can be misleading for imbalanced data)
- **Precision**: Of patients predicted to have the disease, what percentage actually have it? (Measures false alarms)
- **Recall**: What percentage of actual disease cases were correctly identified? (Measures how many sick patients were found)
- **F1 Score**: Harmonic mean of precision and recall, providing a balanced metric

**Visualizations:**
- **Confusion Matrix**: A clear visualization of:
  - True Positives (TP): Correctly identified diseases
  - False Positives (FP): Healthy patients incorrectly flagged
  - True Negatives (TN): Correctly identified healthy patients
  - False Negatives (FN): Missed disease cases
- **ROC Curve & AUC**:
  - Shows model's ability to distinguish between healthy and sick patients
  - AUC (Area Under Curve) closer to 1.0 indicates better performance

**Threshold Analysis**
- Demonstrates how changing the decision threshold affects model performance
- **Lower threshold** (e.g., 0.3):
  - Higher recall (finds more actual cases)
  - More false positives (false alarms)
- **Higher threshold** (e.g., 0.7):
  - Higher precision (more certain when predicting disease)
  - Might miss some actual cases

#### Medical Relevance

**False Positives (Type I Errors):**
- Can cause unnecessary stress and additional testing
- Wastes medical resources
- May lead to unnecessary treatments

**False Negatives (Type II Errors):**
- Could delay critical treatment
- Potentially life-threatening in serious conditions
- Erodes trust in the healthcare system

**Threshold Selection Strategy:**
- **Serious Conditions** (e.g., cancer): Lower threshold to catch all possible cases, accepting more false positives
- **Less Critical Conditions**: Higher threshold to minimize false alarms, requiring stronger evidence for diagnosis

#### Files:
- **medical_diagnosis_metrics.py**: Main script containing data generation, model training, and evaluation logic

#### Example Usage:

```bash
# Run the medical diagnosis metrics evaluation
python evaluation-metrics/medical_diagnosis_metrics.py
```

#### Output Includes:
- Comprehensive metrics report (Accuracy, Precision, Recall, F1)
- Detailed classification metrics for each class
- Visualizations:
  - Confusion matrix (saved as PNG)
  - ROC curve with AUC score (saved as PNG)
- Performance analysis across different classification thresholds

### 2. Image Classification with PyTorch and TensorFlow

This project demonstrates how to implement an image classifier using both PyTorch and TensorFlow frameworks to classify the FashionMNIST dataset. The implementation includes data loading, model architecture definition, training, evaluation, and visualization of results.

#### Features

**Data Handling:**
- Loading and preprocessing the FashionMNIST dataset
- Data normalization and transformation
- Visualization of sample images with labels

**Model Architecture (Both Implementations):**
- Multi-layer neural network with ReLU activations
- Dropout for regularization
- Cross-entropy loss with Adam optimizer

**Training Process:**
- Configurable batch size and learning rate
- Training/validation split
- Progress tracking with loss and accuracy metrics
- Early stopping to prevent overfitting

**Evaluation:**
- Test set accuracy calculation
- Confusion matrix visualization
- Sample predictions with confidence scores

#### Implementation Details

**PyTorch Implementation (`img_classifier_pytorch.py`):**
- Uses `torch.nn.Module` for model definition
- Custom training loop with manual gradient updates
- GPU support with automatic device detection
- Model checkpointing

**TensorFlow Implementation (`img_classifier_tensorflow.py`):**
- Uses Keras Sequential API
- Built-in training loop with `model.fit()`
- TensorBoard integration for visualization
- Automatic differentiation with `GradientTape`

#### Skills Demonstrated

- **Deep Learning Fundamentals:**
  - Neural network architecture design
  - Forward and backward propagation
  - Activation functions and optimization

- **PyTorch & TensorFlow:**
  - Data loading and preprocessing
  - Model definition and training
  - GPU acceleration

- **Machine Learning Best Practices:**
  - Train/validation/test split
  - Hyperparameter tuning
  - Model evaluation metrics
  - Overfitting prevention techniques

#### Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision tensorflow matplotlib numpy
   ```

2. Run the PyTorch implementation:
   ```bash
   python image-classification/img_classifier_pytorch.py
   ```

3. Run the TensorFlow implementation:
   ```bash
   python image-classification/img_classifier_tensorflow.py
   ```

#### Output
- Training progress (loss and accuracy)
- Sample images with predictions
- Model performance on test set
- Saved model checkpoints in `outputs/` directory

### 3. Tabular Data Problem: Titanic Survival Prediction

This project demonstrates a complete machine learning workflow using the classic Titanic dataset. It includes data exploration, preprocessing, model training, evaluation, and a prediction interface.

#### Files

1. **Image Classification**
   - `img_classifier_pytorch.py`: PyTorch implementation
   - `img_classifier_tensorflow.py`: TensorFlow implementation
   - `requirements.txt`: Project dependencies
   - `outputs/`: Directory containing saved models and visualizations

2. **Titanic Survival Prediction**
   - `titanic_analysis_r1.py`
   - Initial data exploration and model training
   - Basic model evaluation and feature importance analysis

2. **titanic_analysis_r2.py**
   - Enhanced version with model comparison
   - Cross-validation and feature set analysis
   - Includes `train_best_model()` function for model persistence

3. **titanic_run.py**
   - Command-line interface for making predictions
   - Interactive prompt for user input
   - Shows survival probability based on passenger details

#### Example Usage:

```bash
# Run the prediction interface
python tabular-data-problem/titanic_run.py
```

#### Sample Output:

```
Titanic Survival Predictor
==========================
Please enter the passenger's details:

Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd): 1
Sex (male/female): female
Age: 25
Fare amount (suggested $30.0-$512.0 for 1st class): 120.50

Analyzing passenger data...

=== Prediction Results ===

Passenger Details:
------------------------------
Class: 1 (1st class)
Sex: Female
Fare: $120.50

Prediction:
------------------------------
Survival Probability: 98.4%
Prediction: Survived

This passenger had a high chance of survival.
```

#### Key Features:

- **Data Preprocessing**: Handles missing values and encodes categorical variables
- **Feature Engineering**: Creates new features like family size and alone status
- **Model Comparison**: Evaluates multiple models and feature sets
- **Interactive Prediction**: User-friendly interface for making predictions
- **Input Validation**: Ensures valid input with helpful error messages

#### Model Performance:

The best performing model was a Decision Tree using a simple feature set (Sex, Pclass, Fare) with 81.18% cross-validated accuracy.

```
Model Comparison:
============================================================
              Model     Feature Set  Mean CV Accuracy  Std Dev
      Decision Tree      Simple Set            0.8118   0.0325
                SVM Family Features            0.8072   0.0085
      Random Forest    All Features            0.8061   0.0276
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-intro-projects
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   Using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

   For minimal installation (core packages only):
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

4. **Verify the installation**
   ```bash
   python -c "import pandas as pd; import sklearn; print('All required packages are installed!')"
   ```

5. **Run the desired script**
   ```bash
   # For example, to run the Titanic predictor:
   python tabular-data-problem/titanic_run.py
   ```

## Dependencies

### Core Dependencies
- Python 3.8+
- numpy (≥1.21.0)
- pandas (≥1.3.0)
- scikit-learn (≥1.0.0)
- matplotlib (≥3.4.0)

### Development Dependencies (included in requirements.txt)
- pytest (for testing)
- jupyter (for interactive development)
- seaborn (for enhanced visualizations)
- mypy (for type checking)
- pylint (for code quality)


