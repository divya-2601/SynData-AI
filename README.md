# SynData AI — Synthetic Data Generation & Evaluation Platform

An AI SaaS web platform for generating realistic tabular datasets using deep generative models (CTGAN and TVAE), with evaluation of statistical fidelity, privacy memorization risks, and machine learning utility.

---

## 🌟 Key Features

1. **Dashboard Overview**
   - High-level platform statistics (Original Records, Synthetic Records, Active Generative Model).
   - "How SynData AI Works" interactive 4-step workflow cards.
   - Core value propositions for privacy compliance, safe testing, and ML research.

2. **Upload Dataset**
   - Drag-and-drop CSV upload with 50 MB file validation.
   - 1-Click preloaded sample datasets (*Customer Churn* and *Heart Disease Clinical*).
   - Instant dataset summary: total rows, columns, numerical attributes, categorical attributes, and missing values.

3. **Data Overview**
   - Interactive feature profiling table with unique value counts, data types, and sample observations.
   - Descriptive numerical statistics table.
   - Interactive Plotly visualizations: Univariate histograms, categorical frequency bar charts, and missing value diagnostics.
   - Dual Pearson correlation heatmap.

4. **Generate Synthetic Data**
   - Deep Generative Models:
     - **CTGAN**: Conditional Generative Adversarial Networks for tabular data.
     - **TVAE**: Variational Autoencoders for tabular data.
   - Hyperparameter Controls: Training epochs, batch size, sample volume, and random seeds.
   - Progress bar with status updates during training.
   - Instant preview of generated artificial records and 1-click CSV download.

5. **Statistical Comparison & Goodness of Fit**
   - Side-by-side numerical moment comparison (Mean, Median, Standard Deviation, Min, Max, and absolute drift).
   - Overlaid distribution histograms (Real vs Synthetic).
   - Two-Sample **Kolmogorov-Smirnov (KS) tests** with exact statistics, p-values, and empirical distribution fit ratings.
   - Pearson correlation matrices for both original and synthetic datasets with Pairwise MAE difference.

6. **Privacy Risk Analysis**
   - High-dimensional **Nearest-Neighbor Distance (NND)** in normalized Euclidean space.
   - Identifies minimum NND, average NND, median NND, and exact duplicates.
   - Identifies suspiciously close records ($\text{distance} \le 0.05$) to detect training memorization.
   - Empirical distance distribution chart with configurable danger threshold.
   - Realistic privacy guidelines and compliance warnings.

7. **Machine Learning Utility Evaluation (TSTR)**
   - Implements **Train on Synthetic, Test on Real (TSTR)** protocol vs **Train on Real, Test on Real (TRTR)** baseline.
   - User-selectable target feature (auto-detects binary and multi-class target attributes).
   - Comparative evaluation of Accuracy, Precision, Recall, and F1-Score on an isolated real test set.
   - Calculates the **Relative Utility Retention Ratio** ($\text{F1}_{\text{TSTR}} / \text{F1}_{\text{TRTR}}$).

8. **Reports & Downloads**
   - Export generated synthetic tabular records in CSV format.
   - 4 Publication-grade PDF audit reports generated with ReportLab:
     - Statistical Quality Audit Report (PDF)
     - Privacy Risk Audit Report (PDF)
     - ML Utility Evaluation Report (PDF)
     - Complete Consolidated Generative AI Evaluation Dossier (PDF)

9. **About Project**
   - Capstone problem statement, mathematical solution, technology stack, and real-world industrial applications.

---

## 🎨 Design System

- **Main Background**: `#FFFFFF`
- **Sidebar Background**: `#241044`
- **Primary Purple**: `#6C3BDB`
- **Dark Purple**: `#3B176D`
- **Light Lavender**: `#F3EDFF`
- **Soft Purple**: `#E8DDFB`
- **Primary Text**: `#211A2C`
- **Secondary Text**: `#777085`
- **Borders**: `#E8E1F0`
- **Success / Warning / Error**: `#16A36A` / `#E5A12B` / `#DC4545`

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Technology Stack

- **Frontend**: Streamlit, Custom CSS/HTML, Plotly Express & Graph Objects
- **Generative AI**: CTGAN, TVAE, PyTorch, RDT
- **Data & Statistics**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn
- **PDF Engine**: ReportLab 5.0
