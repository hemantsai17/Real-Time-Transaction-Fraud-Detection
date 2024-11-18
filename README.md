# Real-Time Transaction Fraud Detection

## Objective
The goal of this project is to build a real-time fraud detection model that flags suspicious credit card transactions as they occur. The model leverages **Anomaly Detection** techniques, such as **Isolation Forest** and **DBSCAN**, to identify fraudulent transactions from a stream of real-time data.

## Dataset
The **Credit Card Fraud Detection Dataset** used in this project is publicly available on Kaggle. It contains credit card transactions labeled as **fraudulent (1)** or **non-fraudulent (0)**, where the fraudulent transactions need to be identified from a large number of genuine ones. The dataset includes anonymized features such as transaction amount, transaction time, and various anonymized variables.

- **Dataset Source**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
  
- **Size of Dataset**: ~284,807 transactions

## Project Features

- **Anomaly Detection Techniques**: This project uses two anomaly detection techniques:
  - **Isolation Forest**: A tree-based model designed for anomaly detection, effective for high-dimensional datasets.
  - **DBSCAN**: A density-based clustering algorithm that identifies outliers as anomalies in the data.

- **Real-Time Detection**: The system can be adapted to flag fraudulent transactions as they occur in real-time.

- **Metrics Used**:
  - **Precision**: The percentage of flagged transactions that are truly fraudulent.
  - **Recall**: The percentage of fraudulent transactions that were flagged by the model.
  - **False Positive Rate**: The percentage of non-fraudulent transactions that were incorrectly flagged as fraudulent.

## Methodology

### 1. **Data Preprocessing**
- **Standardization**: Standardized the dataset using `StandardScaler` to ensure the features are on the same scale, improving the performance of anomaly detection algorithms.
- **Data Exploration**: Performed exploratory data analysis (EDA) to understand the distribution of classes and detect imbalances.

### 2. **Anomaly Detection Models**
- **Isolation Forest**: 
  - An ensemble of tree-based models that isolates observations by randomly selecting features and splitting values, making it well-suited for detecting outliers.
  - Tuned `n_estimators`, `max_samples`, and `contamination` parameters for optimal performance.

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: 
  - Identified outliers in the dataset based on the density of data points. Points in low-density regions are considered anomalies.
  - Adjusted the `eps` (neighborhood size) and `min_samples` (minimum number of points required to form a dense region) parameters.

### 3. **Evaluation**
- **Precision, Recall, FPR**: Used **classification metrics** to evaluate model performance and tune thresholds.
  - **Precision**: True Positives / (True Positives + False Positives)
  - **Recall**: True Positives / (True Positives + False Negatives)
  - **False Positive Rate**: False Positives / (False Positives + True Negatives)

### 4. **Real-Time Integration**
- The model can be integrated with a real-time transaction processing system, where each incoming transaction is flagged as fraudulent or not.

## Project Workflow

1. **Data Import and Preprocessing**:
   - Load and clean the dataset.
   - Handle missing values and outliers.
   - Normalize features to bring all values into a comparable range.

2. **Model Training**:
   - Split the data into training and testing sets.
   - Train Isolation Forest and DBSCAN models.
   - Tune hyperparameters such as `eps`, `min_samples` for DBSCAN, and `contamination`, `n_estimators` for Isolation Forest.

3. **Model Evaluation**:
   - Evaluate models using **Precision**, **Recall**, and **False Positive Rate**.
   - Select the best model based on performance metrics.

4. **Fraud Detection**:
   - Implement a real-time transaction simulation where new transactions are fed into the model and flagged as fraudulent or not.

## Installation

### Prerequisites

- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  
### Installing Dependencies

To install the required libraries, run the following:

```bash
pip install -r requirements.txt
