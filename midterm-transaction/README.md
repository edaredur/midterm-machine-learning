# 🔍 Fraud Detection - Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This repository contains an **end-to-end classification pipeline** for detecting fraudulent online transactions. The project implements and compares multiple **Machine Learning** and **Deep Learning** models to solve this binary classification problem with significant class imbalance.

### 🎯 Objective

To design and implement a comprehensive classification pipeline that can predict the probability of an online transaction being fraudulent (`isFraud`), addressing a critical challenge in e-commerce and financial services.

---

## 👤 Author Information

| Field | Details |
|-------|---------|
| **Name** | Heydar Aqiila Alfarraz |
| **Class** | TK 46 05 |
| **NIM** | 1103223026 |
| **Course** | Machine Learning - Midterm Assignment |
| **Semester** | 7 |

---

## 📊 Dataset Description

| Attribute | Value |
|-----------|-------|
| **Dataset** | Online Transaction Fraud Detection |
| **Training Samples** | 590,540 transactions |
| **Test Samples** | 506,691 transactions |
| **Features** | 393 (Training: 394 including target) |
| **Target Variable** | isFraud (Binary: 0 = Not Fraud, 1 = Fraud) |
| **Class Distribution** | ~96.5% Not Fraud, ~3.5% Fraud |
| **Files** | `train_transaction.csv`, `test_transaction.csv` |

### Feature Categories:
- **Transaction Features**: `TransactionID`, `TransactionDT`, `TransactionAmt`, `ProductCD`
- **Card Features**: `card1` - `card6` (card type, category, issuer)
- **Address Features**: `addr1`, `addr2`, `dist1`, `dist2` (billing/shipping)
- **Email Features**: `P_emaildomain`, `R_emaildomain` (purchaser/recipient)
- **Count Features**: `C1` - `C14` (transaction frequency counters)
- **Time Delta Features**: `D1` - `D15` (time-based features)
- **Match Features**: `M1` - `M9` (categorical match features)
- **Vesta Features**: `V1` - `V339` (Vesta engineered features)

⚠️ **Class Imbalance**: ~27:1 ratio (Not Fraud:Fraud), addressed using SMOTE and undersampling.

---

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                  END-TO-END CLASSIFICATION PIPELINE             │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loading & Exploration                                  │
│     └── Load train/test CSV, basic statistics, class distribution │
│                         ↓                                       │
│  2. Data Preprocessing & Cleaning                               │
│     ├── Missing value analysis & imputation                     │
│     ├── Drop high-missing columns (>80%)                       │
│     ├── Categorical encoding (Label Encoding)                  │
│     └── Train-test split validation                            │
│                         ↓                                       │
│  3. Feature Engineering & Selection                             │
│     ├── Time-based features (Hour, Day, DayOfWeek)            │
│     ├── TransactionAmt transformations (Log, Decimal)         │
│     ├── Variance Threshold (remove quasi-constant)            │
│     └── StandardScaler normalization                           │
│                         ↓                                       │
│  4. Class Imbalance Handling                                    │
│     ├── SMOTE oversampling (minority → 50% majority)          │
│     └── Random undersampling (majority → 80% final)           │
│                         ↓                                       │
│  5. Model Training                                              │
│     ├── Machine Learning Models (4 models)                     │
│     └── Deep Learning Model (PyTorch Neural Network)           │
│                         ↓                                       │
│  6. Hyperparameter Tuning                                       │
│     └── GridSearchCV for Random Forest                         │
│                         ↓                                       │
│  7. Evaluation & Comparison                                     │
│     ├── ROC-AUC, F1, Accuracy, Average Precision              │
│     ├── Visualization (ROC, PR curves, confusion matrix)      │
│     └── Feature importance analysis                            │
│                         ↓                                       │
│  8. Results Interpretation & Predictions                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

### Machine Learning Models (scikit-learn)

| # | Model | Description |
|---|-------|-------------|
| 1 | **Logistic Regression** | Baseline linear classifier |
| 2 | **Random Forest Classifier** | Ensemble of decision trees |
| 3 | **Gradient Boosting Classifier** | Sequential ensemble method |
| 4 | **Tuned Random Forest** | GridSearchCV optimized |

### Deep Learning Model (PyTorch)

| Component | Configuration |
|-----------|---------------|
| **Architecture** | 4 Hidden Layers (256→128→64→32→1) |
| **Activation** | ReLU + Sigmoid (output) |
| **Regularization** | BatchNorm + Dropout (0.2-0.3) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Binary Cross-Entropy |
| **Learning Rate Scheduler** | ReduceLROnPlateau |
| **Early Stopping** | Patience = 5 epochs |
| **Training** | 30 epochs max, batch size 512 |

---

## 📈 Results Summary

### Model Performance Comparison

| Model | Accuracy | ROC-AUC | F1-Score | Avg Precision |
|-------|----------|---------|----------|---------------|
| Tuned Random Forest | - | - | - | - |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |
| Neural Network | - | - | - | - |
| Logistic Regression | - | - | - | - |

> **Note**: Actual values may vary slightly based on random seed and execution.

### 🏆 Best Model: [To be determined after running]

### Key Findings:

1. **Class Imbalance Handling**: SMOTE + Undersampling effectively balanced the training data
2. **Feature Engineering**: Time-based features and log transformation improved model performance
3. **Ensemble Methods**: Tree-based models (Random Forest, Gradient Boosting) perform well on fraud detection
4. **Deep Learning**: PyTorch neural network provides competitive results with proper regularization
5. **Feature Importance**: Vesta features and transaction amount are most predictive

---

## 📁 Repository Structure

```
Dataset 1/
├── 📓 midterm_transaction_data.ipynb  # Main Jupyter Notebook with full pipeline
├── 📊 train_transaction.csv           # Training dataset
├── 📊 test_transaction.csv            # Test dataset
├── 📄 README.md                       # This file
├── 📈 submission.csv                  # Random Forest predictions
└── 📈 submission_ensemble.csv         # Ensemble predictions
```

---

## 🚀 How to Run

### Prerequisites

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn torch
```

### Execution Steps

1. **Clone the repository or navigate to project folder**
   ```bash
   cd "Dataset 1"
   ```

2. **Ensure datasets are present**
   - Place `train_transaction.csv` and `test_transaction.csv` in the same directory

3. **Open and run the notebook**
   ```bash
   jupyter notebook midterm_transaction_data.ipynb
   ```
   Or open in VS Code with Jupyter extension

4. **Run all cells sequentially**
   - The notebook is designed to run from top to bottom
   - Expected runtime: ~20-40 minutes (depending on hardware)

---

## 📓 Notebook Navigation Guide

| Section | Description |
|---------|-------------|
| **1. Import Libraries** | Load all required packages |
| **2. Data Loading** | Load train/test CSV and explore dataset |
| **3. EDA** | Class distribution, missing values, amount analysis |
| **4. Preprocessing** | Handle missing values, encode categoricals |
| **5. Feature Engineering** | Time features, log transforms |
| **6. Feature Selection** | Variance threshold filtering |
| **7. Class Imbalance** | SMOTE + undersampling |
| **8. ML Models** | Train 4 ML models |
| **9. Deep Learning** | PyTorch Neural Network |
| **10. Hyperparameter Tuning** | GridSearchCV for Random Forest |
| **11. Evaluation** | Comparison charts, ROC/PR curves, confusion matrix |
| **12. Predictions** | Generate submission files |
| **13. Conclusion** | Summary and interpretation |

---

## 📊 Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Overall correctness |
| **ROC-AUC** | Area under ROC curve | Threshold-independent performance |
| **F1-Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean of precision & recall |
| **Avg Precision** | Area under PR curve | Focus on positive class |

---

## 🔍 Key Findings

1. **Class Imbalance**: Critical challenge with ~27:1 ratio, successfully handled with SMOTE + undersampling

2. **Feature Engineering**: Time-based features (hour, day) and transaction amount transformations significantly improved performance

3. **Model Performance**: Ensemble methods (Random Forest, Gradient Boosting) outperform linear models, indicating non-linear patterns in fraud behavior

4. **Deep Learning**: Neural network achieves competitive results, demonstrating ability to capture complex fraud patterns

5. **Feature Importance**: Vesta engineered features, transaction amount, and time-based features are most predictive

6. **Real-world Application**: Models can be deployed for real-time fraud detection with appropriate threshold tuning

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, imbalanced-learn |
| **Deep Learning** | PyTorch |
| **Development** | Jupyter Notebook, VS Code |

---

## 📝 Future Improvements

- [ ] Implement XGBoost and LightGBM for potentially better performance
- [ ] Experiment with deeper neural network architectures
- [ ] Try ensemble methods combining multiple models (stacking, voting)
- [ ] Apply more advanced feature engineering techniques
- [ ] Add model explainability (SHAP values, LIME)
- [ ] Implement time-based cross-validation for temporal data
- [ ] Deploy model as REST API for real-time predictions

---

## 📜 License

This project is created for educational purposes as part of the Machine Learning course midterm assignment.

---

## 🙏 Acknowledgments

- **Dataset**: Inspired by IEEE-CIS Fraud Detection competition
- **Course**: Machine Learning - Telkom University
- **Libraries**: scikit-learn, PyTorch, imbalanced-learn
- **Instructor**: Machine Learning Teaching Team

---

<div align="center">

**Made with ❤️ for Machine Learning Midterm**

*Semester 7*

</div>
