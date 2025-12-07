# 🎓 Machine Learning Midterm Examination

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Examination Overview

This repository contains comprehensive solutions for the **Machine Learning Midterm Examination**, demonstrating proficiency in three fundamental machine learning paradigms: **Classification**, **Regression**, and **Clustering**. Each problem implements a complete end-to-end pipeline from data preprocessing to model evaluation.

---

## 👤 Student Information

| Field | Details |
|-------|---------|
| **Name** | Heydar Aqiila Alfarraz |
| **Class** | TK 46 05 |
| **NIM** | 1103223026 |
| **Course** | Machine Learning |
| **Exam Type** | Midterm Examination |
| **Semester** | 7 |
| **Institution** | Telkom University |

---

## 📚 Examination Structure

The midterm consists of **three comprehensive problems**, each targeting a different machine learning task:

### Problem 1: 🔍 Fraud Detection (Classification)
**Objective**: Binary classification to detect fraudulent online transactions

| Aspect | Details |
|--------|---------|
| **Task Type** | Supervised Learning - Binary Classification |
| **Dataset** | Online Transaction Fraud Detection |
| **Samples** | 590,540 training, 506,691 test |
| **Features** | 393 features (transaction, card, address, email, Vesta) |
| **Target** | isFraud (0 = Not Fraud, 1 = Fraud) |
| **Challenge** | Severe class imbalance (~27:1 ratio) |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting, Neural Network |
| **Key Techniques** | SMOTE, Undersampling, Feature Engineering, GridSearchCV |

### Problem 2: 🎵 Song Release Year Prediction (Regression)
**Objective**: Predict the release year of songs from audio features

| Aspect | Details |
|--------|---------|
| **Task Type** | Supervised Learning - Regression |
| **Dataset** | Million Song Dataset (MSD) Subset |
| **Samples** | 515,344 songs |
| **Features** | 90 (12 timbre averages + 78 timbre covariances) |
| **Target** | Release Year (continuous, ~1922-2011) |
| **Challenge** | Complex non-linear relationships in audio features |
| **Models** | Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, Neural Network |
| **Key Techniques** | Outlier Handling, Feature Selection, Hyperparameter Tuning |

### Problem 3: 👥 Customer Segmentation (Clustering)
**Objective**: Segment customers based on credit card usage and payment behavior

| Aspect | Details |
|--------|---------|
| **Task Type** | Unsupervised Learning - Clustering |
| **Dataset** | Credit Card Customer Data |
| **Samples** | ~9,000 customers |
| **Features** | 18 (balance, purchases, cash advance, credit limit, payments) |
| **Target** | None (unsupervised) |
| **Challenge** | Determining optimal number of clusters |
| **Models** | K-Means, Hierarchical (Ward, Complete, Average), DBSCAN |
| **Key Techniques** | Elbow Method, Silhouette Analysis, PCA, Feature Engineering |

---

## 🗂️ Repository Structure

```
UTS/
├── 📄 README.md                    # This file - Midterm summary
│
├── 📁 Dataset 1/                   # Problem 1: Fraud Detection
│   ├── 📓 midterm_transaction_data.ipynb
│   ├── 📊 train_transaction.csv
│   ├── 📊 test_transaction.csv
│   ├── 📈 submission.csv
│   ├── 📈 submission_ensemble.csv
│   └── 📄 README.md
│
├── 📁 Dataset 2/                   # Problem 2: Song Year Prediction
│   ├── 📓 midterm_regresi.ipynb
│   ├── 📊 midterm-regresi-dataset.csv
│   ├── 📈 model_comparison_results.csv
│   └── 📄 README.md
│
├── 📁 Dataset 3/                   # Problem 3: Customer Segmentation
│   ├── 📓 clustering_midterm.ipynb
│   ├── 📊 clusteringmidterm.csv
│   ├── 📈 customer_clustering_results.csv
│   ├── 📈 clustering_model_comparison.csv
│   └── 📄 README.md
│
└── 📁 Soal 1-3/                    # Organized solutions
    ├── Soal 1/ (Classification)
    ├── Soal 2/ (Regression)
    └── Soal 3/ (Clustering)
```

---

## 🔄 Common Pipeline Methodology

Each problem follows a systematic **end-to-end machine learning pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│              STANDARDIZED ML PIPELINE (All Problems)            │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loading & Exploration                                  │
│     └── Load datasets, statistical analysis, visualizations     │
│                         ↓                                       │
│  2. Exploratory Data Analysis (EDA)                             │
│     └── Distribution, correlation, outliers, missing values     │
│                         ↓                                       │
│  3. Data Preprocessing & Cleaning                               │
│     └── Missing values, outliers, duplicates, encoding         │
│                         ↓                                       │
│  4. Feature Engineering & Selection                             │
│     └── Derived features, scaling, dimensionality reduction    │
│                         ↓                                       │
│  5. Problem-Specific Handling                                   │
│     ├── Classification: Class imbalance (SMOTE)                │
│     ├── Regression: Outlier clipping (IQR)                     │
│     └── Clustering: Optimal K determination                    │
│                         ↓                                       │
│  6. Model Training & Selection                                  │
│     └── Multiple ML/DL models, cross-validation                │
│                         ↓                                       │
│  7. Hyperparameter Tuning                                       │
│     └── GridSearchCV, RandomizedSearchCV                        │
│                         ↓                                       │
│  8. Evaluation & Comparison                                     │
│     └── Metrics, visualizations, model selection               │
│                         ↓                                       │
│  9. Results Interpretation & Deployment                         │
│     └── Insights, predictions, export results                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented Across All Problems

### Classification (Problem 1)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Tuned Random Forest (GridSearchCV)
- PyTorch Neural Network (4 layers)

### Regression (Problem 2)
- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)
- ElasticNet (L1+L2)
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- PyTorch Neural Network (4 layers)

### Clustering (Problem 3)
- K-Means Clustering
- Hierarchical Clustering (Ward linkage)
- Hierarchical Clustering (Complete linkage)
- Hierarchical Clustering (Average linkage)
- DBSCAN (Density-based)

**Total Models Implemented: 20**

---

## 📊 Evaluation Metrics Summary

### Classification Metrics (Problem 1)
| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness |
| **ROC-AUC** | Threshold-independent performance |
| **F1-Score** | Balance precision & recall |
| **Average Precision** | Focus on positive class |

### Regression Metrics (Problem 2)
| Metric | Purpose |
|--------|---------|
| **MSE** | Mean Squared Error |
| **RMSE** | Root Mean Squared Error (original units) |
| **MAE** | Mean Absolute Error (robust to outliers) |
| **R²** | Proportion of variance explained |

### Clustering Metrics (Problem 3)
| Metric | Purpose |
|--------|---------|
| **Silhouette Score** | Cluster cohesion (higher = better) |
| **Calinski-Harabasz** | Variance ratio (higher = better) |
| **Davies-Bouldin** | Cluster similarity (lower = better) |

---

## 🏆 Key Results & Performance

### Problem 1: Fraud Detection
- **Best Model**: Tuned Random Forest / Ensemble
- **Key Achievement**: Successfully handled 27:1 class imbalance
- **Performance**: High ROC-AUC and F1-Score on validation set
- **Deliverables**: 
  - `submission.csv` - Random Forest predictions
  - `submission_ensemble.csv` - Ensemble predictions

### Problem 2: Song Year Prediction
- **Best Model**: Tuned Random Forest
- **Key Achievement**: ~9.0 years RMSE, ~0.24 R²
- **Performance**: Outperformed linear models significantly
- **Deliverables**: 
  - `model_comparison_results.csv` - Complete model comparison

### Problem 3: Customer Segmentation
- **Best Model**: K-Means (optimal K determined)
- **Key Achievement**: 4 distinct customer segments identified
- **Performance**: High Silhouette and Calinski-Harabasz scores
- **Deliverables**: 
  - `customer_clustering_results.csv` - Cluster assignments
  - `clustering_model_comparison.csv` - Model metrics

---

## 🛠️ Technologies & Tools Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, imbalanced-learn |
| **Deep Learning** | PyTorch |
| **Statistical Analysis** | SciPy |
| **Development** | Jupyter Notebook, VS Code |

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Install all required packages
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn torch scipy
```

### Running the Solutions

**Problem 1 - Fraud Detection:**
```bash
cd "Dataset 1"
jupyter notebook midterm_transaction_data.ipynb
```

**Problem 2 - Song Year Prediction:**
```bash
cd "Dataset 2"
jupyter notebook midterm_regresi.ipynb
```

**Problem 3 - Customer Segmentation:**
```bash
cd "Dataset 3"
jupyter notebook clustering_midterm.ipynb
```

---

## 🔍 Key Learning Outcomes

### Technical Skills Demonstrated

1. ✅ **Data Preprocessing**: Missing values, outliers, encoding, scaling
2. ✅ **Feature Engineering**: Derived features, transformations, selection
3. ✅ **Class Imbalance**: SMOTE, undersampling, stratified sampling
4. ✅ **Model Selection**: Comparing multiple algorithms systematically
5. ✅ **Hyperparameter Tuning**: GridSearchCV, cross-validation
6. ✅ **Deep Learning**: PyTorch neural networks with regularization
7. ✅ **Evaluation**: Appropriate metrics for each problem type
8. ✅ **Visualization**: Comprehensive charts and plots
9. ✅ **Documentation**: Clear, professional README files
10. ✅ **Code Quality**: Well-structured, commented, reproducible

### Problem-Solving Approaches

**Classification Challenge**: Handled severe class imbalance using SMOTE + undersampling, achieving balanced performance across both classes.

**Regression Challenge**: Predicted continuous values with complex non-linear patterns using ensemble methods and feature engineering.

**Clustering Challenge**: Identified optimal number of clusters through multiple validation methods and created actionable customer segments.

---

## 📈 Performance Summary

| Problem | Dataset Size | Models Tested | Best Model | Key Metric | Performance |
|---------|-------------|---------------|------------|------------|-------------|
| **Classification** | 590K | 5 | Tuned RF / Ensemble | ROC-AUC | High |
| **Regression** | 515K | 8 | Tuned Random Forest | RMSE | ~9.0 years |
| **Clustering** | 9K | 5 | K-Means | Silhouette | Optimal |

---

## 📝 Individual Problem Details

For detailed information about each problem, please refer to the respective README files:

- 📖 [**Problem 1: Fraud Detection**](Dataset%201/README.md) - Classification pipeline with class imbalance handling
- 📖 [**Problem 2: Song Year Prediction**](Dataset%202/README.md) - Regression pipeline with feature selection
- 📖 [**Problem 3: Customer Segmentation**](Dataset%203/README.md) - Clustering pipeline with multiple algorithms

---

## 💡 Key Insights & Findings

### Cross-Problem Observations

1. **Ensemble Methods Excel**: Random Forest and Gradient Boosting consistently performed well across classification and regression tasks

2. **Feature Engineering Matters**: Derived features (time-based, ratios, transformations) significantly improved model performance

3. **Deep Learning Competitive**: PyTorch neural networks achieved competitive results with proper architecture and regularization

4. **Data Quality Critical**: Proper handling of missing values, outliers, and preprocessing was essential for all tasks

5. **Metric Selection Important**: Choosing appropriate evaluation metrics for each problem type (ROC-AUC for imbalanced classification, RMSE for regression, Silhouette for clustering)

6. **Visualization Aids Understanding**: Comprehensive visualizations helped identify patterns, validate preprocessing steps, and interpret results

---

## 🎯 Conclusion

This midterm examination successfully demonstrates comprehensive understanding and practical application of machine learning concepts across three fundamental paradigms:

- **Supervised Learning (Classification)**: Binary classification with imbalanced data
- **Supervised Learning (Regression)**: Continuous value prediction with complex features
- **Unsupervised Learning (Clustering)**: Customer segmentation without labels

Each solution implements industry-standard practices including:
- Systematic exploratory data analysis
- Robust preprocessing pipelines
- Multiple model comparisons
- Hyperparameter optimization
- Comprehensive evaluation
- Clear documentation

The complete pipeline approach ensures reproducibility, maintainability, and real-world applicability of all solutions.

---

## 📜 License

This project is created for educational purposes as part of the Machine Learning course midterm examination at Telkom University.

---

## 🙏 Acknowledgments

- **Course**: Machine Learning - Telkom University
- **Instructor**: Machine Learning Teaching Team
- **Datasets**: 
  - IEEE-CIS Fraud Detection (inspired)
  - Million Song Dataset (MSD)
  - Credit Card Customer Data
- **Libraries**: scikit-learn, PyTorch, imbalanced-learn, SciPy

---

<div align="center">

**Made with ❤️ by Heydar Aqiila Alfarraz**

*TK 46 05 - 1103223026*

**Machine Learning Midterm Examination - Semester 7**

*Telkom University*

---

⭐ *Complete End-to-End Machine Learning Solutions* ⭐

</div>
