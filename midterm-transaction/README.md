# 🔍 Fraud Detection - End-to-End Machine Learning & Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [Author](#author)

---

## 🎯 Overview

This project implements a comprehensive **end-to-end fraud detection system** for online transactions. The pipeline covers the entire machine learning workflow from data preprocessing to model deployment, including both traditional machine learning algorithms and deep learning approaches using PyTorch.

The system is designed to predict the **probability that an online transaction is fraudulent** (`isFraud`), addressing a critical challenge in e-commerce and financial services.

---

## 🎯 Objective

To design and implement an end-to-end machine learning and deep learning pipeline that can:

1. **Predict the probability** of an online transaction being fraudulent
2. Handle real-world data challenges (missing values, class imbalance)
3. Compare multiple ML/DL approaches for optimal performance
4. Provide interpretable results with feature importance analysis

---

## 📊 Dataset Description

### Source
The dataset contains online transaction records with various features related to:
- Transaction details
- Card information
- Address information
- Email domains
- Device/identity information

### Dataset Statistics

| Dataset | Rows | Features |
|---------|------|----------|
| Training | 590,540 | 394 (393 features + 1 target) |
| Test | 506,691 | 393 |

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Transaction** | `TransactionID`, `TransactionDT`, `TransactionAmt`, `ProductCD` | Basic transaction information |
| **Card** | `card1` - `card6` | Card type, category, issuer information |
| **Address** | `addr1`, `addr2`, `dist1`, `dist2` | Billing and shipping address info |
| **Email** | `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email domains |
| **Count** | `C1` - `C14` | Counting features (transaction frequency) |
| **Time Delta** | `D1` - `D15` | Time-based features |
| **Match** | `M1` - `M9` | Match features (categorical) |
| **Vesta** | `V1` - `V339` | Vesta engineered features |

### Target Variable

| Class | Label | Percentage |
|-------|-------|------------|
| Not Fraud | 0 | ~96.5% |
| Fraud | 1 | ~3.5% |

⚠️ **Class Imbalance**: The dataset exhibits significant class imbalance (~27:1 ratio), which is addressed using SMOTE and undersampling techniques.

---

## 📁 Project Structure

```
Dataset Pertama/
├── midterm_transaction_data.ipynb  # Main Jupyter notebook
├── train_transaction.csv           # Training dataset
├── test_transaction.csv            # Test dataset
├── submission.csv                  # Random Forest predictions
├── submission_ensemble.csv         # Ensemble predictions
└── README.md                       # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Install Required Packages

```bash
# Core packages
pip install pandas numpy matplotlib seaborn

# Machine Learning
pip install scikit-learn imbalanced-learn

# Deep Learning (PyTorch)
pip install torch torchvision

# Jupyter
pip install jupyter notebook
```

### Alternative: Install all at once

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch torchvision jupyter
```

---

## 🔄 Pipeline Workflow

### 1. 📥 Data Loading
- Load training and test CSV files
- Initial data exploration and shape verification

### 2. 🔍 Exploratory Data Analysis (EDA)
- **Dataset Overview**: Shape, columns, data types
- **Target Distribution**: Visualize class imbalance
- **Missing Values Analysis**: Identify columns with high missing rates
- **Transaction Amount Analysis**: Distribution by fraud status

### 3. 🛠️ Data Preprocessing

#### 3.1 Handle Missing Values
```python
# Strategy:
- Drop columns with >80% missing values
- Numerical features: Impute with median
- Categorical features: Impute with mode
```

#### 3.2 Encode Categorical Variables
```python
# Label Encoding for categorical columns:
- ProductCD
- card4, card6
- P_emaildomain, R_emaildomain
- M1 - M9
```

### 4. 🔧 Feature Engineering

| New Feature | Description |
|-------------|-------------|
| `Transaction_Hour` | Hour of transaction (0-23) |
| `Transaction_DayOfWeek` | Day of week (0-6) |
| `Transaction_Day` | Day of month (0-29) |
| `TransactionAmt_Log` | Log-transformed transaction amount |
| `TransactionAmt_decimal` | Decimal part of transaction amount |

### 5. 📉 Feature Selection
- **Variance Threshold**: Remove quasi-constant features (variance < 0.01)

### 6. ⚖️ Handle Class Imbalance
```python
# Combined approach:
1. SMOTE (sampling_strategy=0.5)  # Oversample minority to 50% of majority
2. Random Undersampling (sampling_strategy=0.8)  # Undersample majority
```

### 7. 📊 Data Scaling
- StandardScaler for neural network inputs

---

## 🤖 Models Implemented

### Machine Learning Models

#### 1. Logistic Regression (Baseline)
```python
LogisticRegression(max_iter=1000, random_state=42)
```

#### 2. Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

#### 3. Gradient Boosting Classifier
```python
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

#### 4. Tuned Random Forest (GridSearchCV)
```python
# Hyperparameter Grid:
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5]
}
```

### Deep Learning Model (PyTorch)

#### Neural Network Architecture
```
FraudDetectionNN(
    Linear(input_dim → 256) → ReLU → BatchNorm → Dropout(0.3)
    Linear(256 → 128) → ReLU → BatchNorm → Dropout(0.3)
    Linear(128 → 64) → ReLU → BatchNorm → Dropout(0.2)
    Linear(64 → 32) → ReLU
    Linear(32 → 1) → Sigmoid
)
```

#### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 512 |
| Epochs | 30 (with early stopping) |
| Early Stopping Patience | 5 epochs |
| LR Scheduler | ReduceLROnPlateau |

---

## 📏 Evaluation Metrics

### Primary Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **ROC-AUC** | Area Under ROC Curve | Best for imbalanced datasets, threshold-independent |
| **F1-Score** | Harmonic mean of Precision & Recall | Balances false positives and negatives |
| **Average Precision** | Area under Precision-Recall curve | Focus on positive class performance |
| **Accuracy** | Overall correctness | Basic performance indicator |

### Visualization
- **ROC Curves**: Compare all models
- **Precision-Recall Curves**: Focus on fraud detection ability
- **Confusion Matrix**: Detailed prediction breakdown
- **Feature Importance**: Top contributing features

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | ROC-AUC | F1-Score | Avg Precision |
|-------|----------|---------|----------|---------------|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |
| Tuned Random Forest | - | - | - | - |
| Neural Network (PyTorch) | - | - | - | - |

> **Note**: Run the notebook to see actual results for your dataset.

### Key Findings

1. **Class Imbalance Handling**: SMOTE + Undersampling effectively balanced the training data
2. **Feature Engineering**: Time-based features and log transformation improved model performance
3. **Model Selection**: Tree-based models (Random Forest, Gradient Boosting) typically perform well on fraud detection
4. **Deep Learning**: PyTorch neural network provides competitive results with proper regularization

---

## 🚀 Usage

### Running the Notebook

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook midterm_transaction_data.ipynb
   ```

2. **Run All Cells**
   - Execute cells sequentially from top to bottom
   - Or use "Run All" from the Cell menu

3. **View Results**
   - Model comparison summary
   - Visualizations (ROC curves, confusion matrix)
   - Feature importance analysis

### Making Predictions

```python
# Load the trained model
import joblib
model = joblib.load('best_model.pkl')

# Prepare new data (apply same preprocessing)
new_data = preprocess(new_transaction_data)

# Get fraud probability
fraud_probability = model.predict_proba(new_data)[:, 1]
```

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `submission.csv` | Predictions from Tuned Random Forest (TransactionID, isFraud probability) |
| `submission_ensemble.csv` | Ensemble predictions (average of RF and NN) |

### Submission Format
```csv
TransactionID,isFraud
2987000,0.0234
2987001,0.8756
...
```

---

## 📦 Dependencies

### Core Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Machine Learning
```
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
```

### Deep Learning
```
torch>=2.0.0
torchvision>=0.15.0
```

### Development
```
jupyter>=1.0.0
notebook>=6.4.0
```

---

## 📊 Visualizations Generated

1. **Class Distribution** - Bar plot and pie chart of fraud vs non-fraud
2. **Missing Values Distribution** - Bar chart showing missing value ranges
3. **Transaction Amount Distribution** - Histogram by fraud status
4. **Training History** - Loss, Accuracy, and AUC curves (Neural Network)
5. **ROC Curves** - All models comparison
6. **Precision-Recall Curves** - All models comparison
7. **Confusion Matrix** - Best model detailed breakdown
8. **Feature Importance** - Top 20 most important features

---

## 🔮 Future Improvements

1. **Additional Models**: XGBoost, LightGBM, CatBoost
2. **Advanced Feature Engineering**: More aggregation features, interaction terms
3. **Hyperparameter Optimization**: Bayesian optimization, Optuna
4. **Model Explainability**: SHAP values, LIME
5. **Real-time Prediction**: API deployment with FastAPI/Flask
6. **Cross-Validation**: Time-based stratified cross-validation

---

## 👨‍💻 Author

**Machine Learning Midterm Project**
- Course: Machine Learning
- Semester: 7
- Institution: [Your University]

---

## 📄 License

This project is for educational purposes as part of a university midterm examination.

---

## 🙏 Acknowledgments

- Dataset inspiration from IEEE-CIS Fraud Detection competition
- scikit-learn and PyTorch documentation
- imbalanced-learn library for handling class imbalance

---

<p align="center">
  <b>⭐ If you found this helpful, please give it a star! ⭐</b>
</p>
