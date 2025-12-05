# Customer Segmentation using Clustering Analysis

## 📋 Project Overview

This project implements a comprehensive end-to-end machine learning pipeline for **customer clustering/segmentation** based on credit card usage and payment behavior. The goal is to identify distinct customer groups to enable targeted marketing strategies and personalized services.

## 🎯 Objective

Design and implement a comprehensive end-to-end machine learning pipeline dedicated to a clustering task, specifically for customer segmentation based on spending and payment behavior patterns.

## 📊 Dataset Description

The dataset (`clusteringmidterm.csv`) contains information about each customer's credit card usage and payment behavior with the following features:

| Feature | Description |
|---------|-------------|
| `CUST_ID` | Unique customer identifier |
| `BALANCE` | Current/average outstanding balance on the card |
| `BALANCE_FREQUENCY` | How often the balance is updated |
| `PURCHASES` | Total purchase amount |
| `ONEOFF_PURCHASES` | Large, one-time purchases |
| `INSTALLMENTS_PURCHASES` | Purchases paid in installments |
| `CASH_ADVANCE` | Total cash withdrawn using the card |
| `PURCHASES_FREQUENCY` | Frequency of purchases |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-off purchases |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash advances |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Maximum credit available |
| `PAYMENTS` | Amount paid back |
| `MINIMUM_PAYMENTS` | Total minimum payments made |
| `PRC_FULL_PAYMENT` | Proportion of full balance payments |
| `TENURE` | Duration of card ownership (months) |

## 🔧 Pipeline Steps

### 1. Data Loading & Exploration
- Load dataset and examine structure
- Identify data types and missing values
- Generate statistical summary

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of numerical features
- Correlation heatmap
- Outlier detection using boxplots

### 3. Data Preprocessing & Cleaning
- Handle missing values using median imputation
- Outlier treatment using IQR-based Winsorization
- Feature scaling using StandardScaler

### 4. Feature Engineering
- Credit Utilization Ratio
- Payment to Balance Ratio
- Cash Advance Ratio
- Purchase to Credit Limit Ratio
- Average Purchase per Transaction
- One-off to Installment Ratio
- Monthly Average Balance
- Monthly Average Purchases

### 5. Dimensionality Reduction
- PCA for visualization
- Variance explained analysis

### 6. Clustering Models Implementation
- **K-Means Clustering**: With Elbow method and Silhouette analysis for optimal K
- **Hierarchical Clustering**: Agglomerative with Ward, Complete, and Average linkage
- **DBSCAN**: Density-based clustering with eps parameter tuning

### 7. Model Evaluation & Comparison
Metrics used:
- Silhouette Score (higher is better)
- Calinski-Harabasz Index (higher is better)
- Davies-Bouldin Index (lower is better)

### 8. Cluster Interpretation & Profiling
- Cluster characteristic analysis
- Radar charts for profile visualization
- Customer segment descriptions

## 📈 Results Summary

### Model Comparison

| Model | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|-------|------------------|-------------------|----------------|
| K-Means | ✓ | ✓ | ✓ |
| Hierarchical | ✓ | ✓ | ✓ |
| DBSCAN | ✓ | ✓ | ✓ |

*Note: Actual values will be generated upon running the notebook*

### Customer Segments Identified

Based on K-Means clustering, customers are segmented into distinct groups with unique characteristics:

1. **Low Activity/New Customers**: Low balance, minimal purchases
2. **Regular Purchasers**: Moderate spending, regular payment patterns
3. **High-Value Customers**: High credit limits, significant purchases
4. **Cash Advance Users**: Primarily use card for cash advances

## 📁 Repository Structure

```
Dataset Ketiga/
├── clustering_midterm.ipynb    # Main Jupyter notebook with complete analysis
├── clusteringmidterm.csv       # Original dataset
├── customer_clustering_results.csv  # Output: Customers with cluster labels
├── clustering_model_comparison.csv  # Output: Model metrics comparison
└── README.md                   # This file
```

## 🚀 How to Run

1. **Prerequisites**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

2. **Run the Notebook**:
   - Open `clustering_midterm.ipynb` in Jupyter Notebook/Lab or VS Code
   - Run all cells sequentially

3. **Output Files**:
   - `customer_clustering_results.csv`: Contains original data with cluster assignments
   - `clustering_model_comparison.csv`: Contains performance metrics for all models

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms and metrics
- **scipy**: Hierarchical clustering dendrogram

## 📝 Key Findings

1. **Optimal Number of Clusters**: Determined using Elbow method and Silhouette analysis
2. **Best Performing Model**: Based on clustering metrics comparison
3. **Distinct Customer Segments**: Each cluster exhibits unique spending and payment patterns
4. **Actionable Insights**: Clusters can be used for targeted marketing and risk assessment

## 👤 Author Information

- **Name**: [Your Name]
- **Class**: [Your Class]
- **NIM**: [Your Student ID]

## 📄 License

This project is submitted as part of the Machine Learning Midterm Examination.

---

*Last Updated: December 2024*
