Here is the complete README.md draft in English, incorporating all project details:

---

# **QSAR Model Development for Discovering New PTP1B Inhibitors**

This project focuses on developing a QSAR (Quantitative Structure-Activity Relationship) model to discover potential inhibitors for PTP1B. The pipeline involves molecular descriptor processing, feature selection, machine learning model training, and compound screening using XGBoost.

---

## **Project Overview**

- **Objective**: Build a QSAR model to classify active and inactive compounds for PTP1B and use it for screening potential inhibitors.
- **Dataset**: Data from PDBbinds, including PTP1B inhibitors categorized as positive (active) and negative (inactive) compounds.
- **Model**: XGBoost classifier optimized for binary classification tasks.

---

## **Key Components**

### **1. Data**
- **Raw Data**:
  - `data/raw/PTP1B_positive_compounds_BindingDB(200).csv`: known as active (positive) inhibitors of PTP1B.
  - `data/raw/PTP1B_negative_compounds_BindingDB(218).csv`: known as inactive (negative) compounds of PTP1B
- **Processed Data**:
  - **`Descriptor_preprocessing_results.csv`**: Preprocessed data after cleaning, variance filtering, and correlation handling.
  - **`Final_feature_selection_data_with_ylabel.csv`**: Data after feature selection with Logistic Regression.
- **Screening Results**:
  - **`final_screened_compounds.sdf`**: SDF file of compounds screened using the trained model.

### **2. Model**
- **`PTP1B_prediction_QSAR_model.pkl`**:
  - The final trained XGBoost model used for predicting compound activity.

### **3. Notebooks**
- **Descriptor_Calculation.ipynb**:
  - Generates molecular descriptors for positive and negative datasets using PaDEL-Descriptor.
- **Descriptor_Preprocessing.ipynb**:
  - Handles missing values, removes low-variance features, and resolves correlation.
- **Descriptor_Feature_Selection.ipynb**:
  - Selects relevant features using Logistic Regression with L1 regularization.
- **Model_Training_and_Validation.ipynb**:
  - Trains the XGBoost model, performs hyperparameter tuning, cross-validation, and evaluates model performance.

---

## **How to Run**

### **1. Environment Setup**
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **2. Descriptor calculation**
Generate molecular descriptors for the positive and negative datasets:
```plaintext
notebooks/Descriptor_Calculation.ipynb
```

### **3. Descriptor Preprocessing**
Run the preprocessing notebook to clean and prepare molecular descriptors:
```plaintext
notebooks/Descriptor_Preprocessing.ipynb
```

### **4. Feature Selection**
Run the feature selection notebook to identify relevant features:
```plaintext
notebooks/Feature_Selection.ipynb
```

### **5. Model Training and Validation**
Train and validate the XGBoost model:
```plaintext
notebooks/Model_Training_and_Validation.ipynb
```

### **6. Compound Screening**
Use the trained model to screen new compounds:
```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('./model/PTP1B_prediction_QSAR_model.pkl')

# Load screening data
screening_data = pd.read_csv('screening_data.csv')

# Predict activity
predictions = model.predict(screening_data)
```

---

## **Dependencies**

The required dependencies for this project are listed below:

```plaintext
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
xgboost==1.7.4
matplotlib==3.7.1
joblib==1.2.0
padelpy==0.1.10
```

**Note**: 
- **PadelPy** requires Java Runtime Environment (JRE) to be installed on your system.
- To check Java installation:
  ```bash
  java -version
  ```
- If Java is not installed:
  - **Linux**: `sudo apt-get install default-jre`
  - **Windows**: Download from [Java.com](https://www.java.com/download/).

---
