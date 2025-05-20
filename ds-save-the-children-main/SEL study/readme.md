# Project Overview

This project consists of two Jupyter Notebook files:

1. **`sel-pre.ipynb`**: Responsible for data preprocessing and prepare the data for dashboard construction.
2. **`sel.ipynb`**: Focused on model training, hyperparameter tuning, and generating predictions.

These notebooks work together to complete a pipeline from raw data preprocessing to prediction generation.

---

# File Descriptions

## 1. sel-pre.ipynb
- **Purpose**: Handles data loading, cleaning, and transformation.
- **Key Features**:
  - Load the raw dataset.
  - Handle missing values and outliers.
  - Convert the lable of some features.
  - Save the preprocessed data for subsequent use.

## 2. sel.ipynb
- **Purpose**: Loads preprocessed data for model training and prediction.
- **Key Features**:
  - Re-preprocess the raw dataset
  - Perform feature encoding and normalization.
  - Define and train like RandomForestRegressor.
  - Perform hyperparameter tuning using GridSearchCV.
  - Generate predictions.
  - Visualize model performance and prediction results.

---

# Dependencies

The required Python packages are listed in the `requirements.txt` file. Install them using the following command:

```bash
pip install -r requirements.txt
```

---

# Outputs
## 1. sel-pre.ipynb:
Outputs the preprocessed dataset saved as nigeria_sel_score.csv
## 2. sel.ipynb:
Outputs model predictions and feature importances as nigeria_sel_feature_importance.csv