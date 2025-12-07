# 👥 Customer Segmentation - Clustering Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This repository contains an **end-to-end clustering pipeline** for customer segmentation based on credit card usage and payment behavior. The project implements and compares multiple **Clustering** algorithms to identify distinct customer groups for targeted marketing strategies.

### 🎯 Objective

To design and implement a comprehensive clustering pipeline that can segment customers into distinct groups based on their spending and payment behavior patterns, enabling personalized services and targeted marketing.

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
| **Dataset** | Credit Card Customer Data |
| **Total Samples** | ~9,000 customers |
| **Features** | 18 (17 behavioral features + 1 ID) |
| **Target Variable** | None (Unsupervised Learning) |
| **File** | `clusteringmidterm.csv` |

### Feature Breakdown:
- **Identification**: `CUST_ID` - Unique customer identifier
- **Balance Features**: `BALANCE`, `BALANCE_FREQUENCY` - Outstanding balance metrics
- **Purchase Features**: `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES` - Purchase behavior
- **Purchase Frequency**: `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`
- **Cash Advance**: `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX` - Cash withdrawal behavior
- **Transaction Count**: `PURCHASES_TRX` - Number of purchase transactions
- **Credit & Payment**: `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS` - Credit and payment info
- **Payment Behavior**: `PRC_FULL_PAYMENT` - Proportion of full payments
- **Account Age**: `TENURE` - Duration of card ownership (months)

---

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    END-TO-END CLUSTERING PIPELINE               │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loading & Exploration                                  │
│     └── Load CSV, basic statistics, visualizations             │
│                         ↓                                       │
│  2. Data Preprocessing & Cleaning                               │
│     ├── Missing value imputation (median)                      │
│     ├── Outlier handling (IQR-based Winsorization)            │
│     └── Feature scaling (StandardScaler)                       │
│                         ↓                                       │
│  3. Feature Engineering                                         │
│     ├── Credit Utilization Ratio                               │
│     ├── Payment to Balance Ratio                               │
│     ├── Cash Advance Ratio                                      │
│     ├── Purchase to Credit Limit Ratio                         │
│     ├── Average Purchase per Transaction                       │
│     ├── One-off to Installment Ratio                          │
│     ├── Monthly Average Balance                                │
│     └── Monthly Average Purchases                              │
│                         ↓                                       │
│  4. Dimensionality Reduction                                    │
│     ├── PCA for visualization                                   │
│     └── Variance explained analysis                            │
│                         ↓                                       │
│  5. Clustering Models                                           │
│     ├── K-Means (with Elbow & Silhouette)                     │
│     ├── Hierarchical (Ward, Complete, Average)                │
│     └── DBSCAN (density-based)                                 │
│                         ↓                                       │
│  6. Evaluation & Comparison                                     │
│     ├── Silhouette Score, Calinski-Harabasz, Davies-Bouldin  │
│     ├── Model comparison visualization                         │
│     └── Optimal cluster determination                          │
│                         ↓                                       │
│  7. Cluster Interpretation & Profiling                          │
│     ├── Cluster characteristics analysis                       │
│     ├── Radar charts for profiles                             │
│     └── Customer segment descriptions                          │
│                         ↓                                       │
│  8. Results Export & Conclusion                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

### Clustering Algorithms

| # | Model | Description |
|---|-------|-------------|
| 1 | **K-Means Clustering** | Partition-based, optimal K via Elbow & Silhouette |
| 2 | **Hierarchical Clustering (Ward)** | Agglomerative with Ward linkage |
| 3 | **Hierarchical Clustering (Complete)** | Agglomerative with Complete linkage |
| 4 | **Hierarchical Clustering (Average)** | Agglomerative with Average linkage |
| 5 | **DBSCAN** | Density-based clustering with eps tuning |

### Optimal Cluster Selection

| Method | Technique |
|--------|-----------|
| **Elbow Method** | Identify knee point in inertia curve |
| **Silhouette Analysis** | Maximize average silhouette score |
| **Dendrogram** | Visual hierarchy for hierarchical clustering |

---

## 📈 Results Summary

### Model Performance Comparison

| Model | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|-------|------------------|-------------------|----------------|
| K-Means | ✓ | ✓ | ✓ |
| Hierarchical (Ward) | ✓ | ✓ | ✓ |
| Hierarchical (Complete) | ✓ | ✓ | ✓ |
| Hierarchical (Average) | ✓ | ✓ | ✓ |
| DBSCAN | ✓ | ✓ | ✓ |

> **Note**: Actual values will be generated upon running the notebook.

### 🏆 Best Model: [To be determined after running]

### Customer Segments Identified

Based on optimal clustering, customers are segmented into distinct groups:

1. **Low Activity/New Customers**: Low balance, minimal purchases, potential growth targets
2. **Regular Purchasers**: Moderate spending, consistent payment patterns, loyal customers
3. **High-Value Customers**: High credit limits, significant purchases, premium segment
4. **Cash Advance Users**: Primarily use card for cash advances, different financial needs

---

## 📁 Repository Structure

```
Dataset 3/
├── 📓 clustering_midterm.ipynb          # Main Jupyter Notebook with full pipeline
├── 📊 clusteringmidterm.csv             # Dataset
├── 📄 README.md                         # This file
├── 📈 customer_clustering_results.csv   # Customers with cluster labels
└── 📈 clustering_model_comparison.csv   # Model metrics comparison
```

---

## 🚀 How to Run

### Prerequisites

```bash
# Required packages
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Execution Steps

1. **Clone the repository or navigate to project folder**
   ```bash
   cd "Dataset 3"
   ```

2. **Ensure dataset is present**
   - Place `clusteringmidterm.csv` in the same directory as the notebook

3. **Open and run the notebook**
   ```bash
   jupyter notebook clustering_midterm.ipynb
   ```
   Or open in VS Code with Jupyter extension

4. **Run all cells sequentially**
   - The notebook is designed to run from top to bottom
   - Expected runtime: ~10-20 minutes (depending on hardware)

---

## 📓 Notebook Navigation Guide

| Section | Description |
|---------|-------------|
| **1. Import Libraries** | Load all required packages |
| **2. Data Loading** | Load and explore dataset |
| **3. EDA** | Distribution analysis, correlation, outliers |
| **4. Preprocessing** | Missing values, outlier treatment, scaling |
| **5. Feature Engineering** | Create derived features |
| **6. Dimensionality Reduction** | PCA for visualization |
| **7. K-Means** | Elbow method, Silhouette analysis, clustering |
| **8. Hierarchical** | Dendrogram, different linkages |
| **9. DBSCAN** | Eps parameter tuning, clustering |
| **10. Evaluation** | Compare all models with metrics |
| **11. Cluster Profiling** | Interpret and visualize segments |
| **12. Export Results** | Save cluster assignments and metrics |
| **13. Conclusion** | Summary and actionable insights |

---

## 📊 Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Silhouette Score** | $\frac{b - a}{\max(a, b)}$ | Range: [-1, 1], higher is better, measures cluster cohesion |
| **Calinski-Harabasz** | $\frac{SS_{between}/(k-1)}{SS_{within}/(n-k)}$ | Higher is better, ratio of between/within cluster variance |
| **Davies-Bouldin** | $\frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$ | Lower is better, average similarity between clusters |

---

## 🔍 Key Findings

1. **Optimal Clusters**: Determined through multiple validation methods (Elbow, Silhouette, Dendrogram)

2. **Feature Engineering**: Derived ratios and monthly averages provide more meaningful segmentation than raw features alone

3. **Model Selection**: K-Means typically performs well for customer segmentation due to its spherical cluster assumption

4. **Outlier Handling**: IQR-based Winsorization prevents extreme values from dominating cluster formation

5. **Actionable Segments**: Each cluster exhibits distinct spending and payment patterns, enabling targeted strategies

6. **Business Applications**: 
   - Cluster 1: Acquisition campaigns
   - Cluster 2: Loyalty programs
   - Cluster 3: Premium services
   - Cluster 4: Credit counseling

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Statistical Analysis** | SciPy |
| **Development** | Jupyter Notebook, VS Code |

---

## 📝 Future Improvements

- [ ] Implement Gaussian Mixture Models (GMM) for soft clustering
- [ ] Try HDBSCAN for hierarchical density-based clustering
- [ ] Apply more advanced feature engineering (temporal patterns)
- [ ] Implement cluster stability analysis
- [ ] Add cluster profiling with statistical tests
- [ ] Create interactive visualizations with Plotly
- [ ] Develop customer lifetime value predictions per cluster

---

## 📜 License

This project is created for educational purposes as part of the Machine Learning course midterm assignment.

---

## 🙏 Acknowledgments

- **Dataset**: Credit Card Customer Data
- **Course**: Machine Learning - Telkom University
- **Libraries**: scikit-learn, SciPy, Pandas
- **Instructor**: Machine Learning Teaching Team

---

<div align="center">

**Made with ❤️ for Machine Learning Midterm**

*Semester 7*

</div>
