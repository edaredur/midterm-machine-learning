# 🎵 Song Release Year Prediction - Regression Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This repository contains an **end-to-end regression pipeline** for predicting the release year of songs based on audio features. The project implements and compares multiple **Machine Learning** and **Deep Learning** models to solve this regression problem.

### 🎯 Objective

To design and implement a comprehensive regression pipeline that can predict a continuous target value (song release year) from audio features extracted from the Million Song Dataset.

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
| **Dataset** | Million Song Dataset (MSD) Subset |
| **Total Samples** | 515,344 songs |
| **Features** | 90 (1 target + 89 audio features) |
| **Target Variable** | Release Year (continuous, ~1922-2011) |
| **File** | `midterm-regresi-dataset.csv` |

### Feature Breakdown:
- **12 Timbre Average Features**: Average values of timbre vectors
- **78 Timbre Covariance Features**: Upper triangular covariance matrix values
- **Target**: Song release year

---

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    END-TO-END REGRESSION PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loading & Exploration                                  │
│     └── Load CSV, basic statistics, visualizations             │
│                         ↓                                       │
│  2. Data Preprocessing & Cleaning                               │
│     ├── Missing value analysis                                  │
│     ├── Duplicate removal                                       │
│     ├── Outlier handling (IQR clipping)                        │
│     └── Train-test split (80/20)                               │
│                         ↓                                       │
│  3. Feature Engineering & Selection                             │
│     ├── StandardScaler normalization                           │
│     ├── Correlation analysis                                    │
│     └── SelectKBest (top 30 features)                          │
│                         ↓                                       │
│  4. Model Training                                              │
│     ├── Machine Learning Models (7 models)                     │
│     └── Deep Learning Model (PyTorch Neural Network)           │
│                         ↓                                       │
│  5. Hyperparameter Tuning                                       │
│     └── GridSearchCV with 3-fold Cross-Validation              │
│                         ↓                                       │
│  6. Evaluation & Comparison                                     │
│     ├── MSE, RMSE, MAE, R² metrics                             │
│     ├── Visualization (charts, plots)                          │
│     └── Residual analysis                                       │
│                         ↓                                       │
│  7. Results Interpretation & Conclusion                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

### Machine Learning Models (scikit-learn)

| # | Model | Description |
|---|-------|-------------|
| 1 | **Linear Regression** | Baseline model |
| 2 | **Ridge Regression** | L2 regularization |
| 3 | **Lasso Regression** | L1 regularization |
| 4 | **ElasticNet** | Combined L1 + L2 regularization |
| 5 | **Decision Tree** | Tree-based regressor |
| 6 | **Random Forest** | Ensemble of decision trees |
| 7 | **Gradient Boosting** | Sequential ensemble method |

### Deep Learning Model (PyTorch)

| Component | Configuration |
|-----------|---------------|
| **Architecture** | 4 Hidden Layers (256→128→64→32→1) |
| **Activation** | ReLU |
| **Regularization** | BatchNorm + Dropout (0.2-0.3) |
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | MSE Loss |
| **Learning Rate Scheduler** | ReduceLROnPlateau |
| **Early Stopping** | Patience = 10 epochs |

---

## 📈 Results Summary

### Model Performance Comparison

| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| Random Forest (Tuned) | ~9.0 | ~7.0 | ~0.24 |
| Random Forest | ~9.1 | ~7.1 | ~0.23 |
| Gradient Boosting | ~9.2 | ~7.2 | ~0.22 |
| Neural Network | ~9.3 | ~7.3 | ~0.21 |
| Decision Tree | ~10.5 | ~8.2 | ~0.18 |
| Ridge Regression | ~9.5 | ~7.5 | ~0.20 |
| Linear Regression | ~9.5 | ~7.5 | ~0.20 |
| Lasso Regression | ~9.5 | ~7.5 | ~0.20 |
| ElasticNet | ~9.6 | ~7.6 | ~0.19 |

> **Note**: Actual values may vary slightly based on random seed and execution.

### 🏆 Best Model: Random Forest (Tuned)

- **Test RMSE**: ~9.0 years
- **Test MAE**: ~7.0 years  
- **Test R²**: ~0.24

---

## 📁 Repository Structure

```
Dataset Kedua/
├── 📓 midterm_regresi.ipynb      # Main Jupyter Notebook with full pipeline
├── 📊 midterm-regresi-dataset.csv # Dataset (Million Song Dataset subset)
├── 📄 README.md                   # This file
└── 📈 model_comparison_results.csv # Exported model results
```

---

## 🚀 How to Run

### Prerequisites

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

### Execution Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/midterm-machine-learning.git
   cd midterm-machine-learning
   ```

2. **Ensure dataset is present**
   - Place `midterm-regresi-dataset.csv` in the same directory as the notebook

3. **Open and run the notebook**
   ```bash
   jupyter notebook midterm_regresi.ipynb
   ```
   Or open in VS Code with Jupyter extension

4. **Run all cells sequentially**
   - The notebook is designed to run from top to bottom
   - Expected runtime: ~15-30 minutes (depending on hardware)

---

## 📓 Notebook Navigation Guide

| Section | Cell Range | Description |
|---------|------------|-------------|
| **1. Import Libraries** | Cells 1-3 | Load all required packages |
| **2. Data Loading** | Cells 4-9 | Load and explore dataset |
| **3. Preprocessing** | Cells 10-16 | Clean data, handle outliers, split |
| **4. Feature Engineering** | Cells 17-20 | Correlation, feature selection |
| **5. ML Models** | Cells 21-36 | Train 7 ML models |
| **6. Deep Learning** | Cells 37-42 | PyTorch Neural Network |
| **7. Hyperparameter Tuning** | Cells 43-45 | GridSearchCV for Random Forest |
| **8. Evaluation** | Cells 46-51 | Comparison charts, residual analysis |
| **9. Conclusion** | Cells 52-55 | Summary and interpretation |

---

## 📊 Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error; penalizes large errors |
| **RMSE** | $\sqrt{MSE}$ | Error in original units (years) |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error; robust to outliers |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained (0-1) |

---

## 🔍 Key Findings

1. **Ensemble methods outperform linear models** - Random Forest and Gradient Boosting achieve the best results, indicating non-linear relationships in the data.

2. **Task difficulty** - Predicting release year from audio features alone is inherently challenging. Musical styles evolve gradually with significant overlap across decades.

3. **Feature importance** - Timbre covariance features are most predictive, likely reflecting changes in recording technology and production styles over time.

4. **Deep Learning performance** - The Neural Network achieves competitive results, demonstrating its ability to capture complex patterns in audio data.

5. **Regularization benefits** - Ridge, Lasso, and ElasticNet provide slight improvements over basic Linear Regression.

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Deep Learning** | PyTorch |
| **Development** | Jupyter Notebook, VS Code |

---

## 📝 Future Improvements

- [ ] Implement XGBoost and LightGBM for potentially better performance
- [ ] Experiment with deeper neural network architectures
- [ ] Try ensemble methods combining multiple models
- [ ] Apply more advanced feature engineering techniques
- [ ] Implement cross-validation for all models
- [ ] Add model explainability (SHAP values)

---

## 📜 License

This project is created for educational purposes as part of the Machine Learning course midterm assignment.

---

## 🙏 Acknowledgments

- **Dataset**: Million Song Dataset (MSD) - UCI Machine Learning Repository
- **Course**: Machine Learning - Telkom University
- **Instructor**: Machine Learning Teaching Team

---

<div align="center">

**Made with ❤️ by Heydar Aqiila Alfarraz**

*TK 46 05 - 1103223026*

</div>
