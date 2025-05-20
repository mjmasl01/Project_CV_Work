# Predicting Depression Risk from NHANES Data  
**Author**: Matthew Maslow  
**Course**: CDS-DS-596 – Special Topics in Medical Science for Data Science  
**Date**: May 2025

---

## Project Overview

This project uses machine learning to predict depression risk using data from the 2021–2023 NHANES (National Health and Nutrition Examination Survey). Two key self-reported depression indicators from the PHQ-9 screener (DPQ020 and DPQ060) are modeled using **Logistic Regression** and **XGBoost**, with **SMOTE** applied to address class imbalance.

The project integrates behavioral, socioeconomic, and clinical features (e.g., sleep, income, CRP, Vitamin D) and explores explainability using **SHAP values**.

---

## Objectives

- Predict binary depression indicators (DPQ020 and DPQ060)
- Compare **Logistic Regression** and **XGBoost**
- Test the effect of **SMOTE** vs. native class weighting
- Interpret feature importance using **SHAP**

---

## 📁 Project Structure

Predicting Depression Risk from NHANES Data/
│
├── data/ # Raw XPT files (excluded from GitHub)
├── DataCSV/ # Cleaned merged dataset in CSV format
│
├── README.md # This file
├── data.zip # ZIP archive of raw data (6337 rows, 28 columns)
│
├── Final Project Report - mjmaslow.pdf # Final research paper writeup
├── Final Project Report.docx # Editable version of the report
├── Works Cited.docx # Citation file
│
├── finalProjCode_mjmaslow.pdf # Cleaned final model notebook (PDF)
├── finalProjCode.ipynb # Original modeling notebook
│
├── Final Project Notes.docx # Draft notes and ideas
├── Final Project Sp25.pdf # Older project doc (optional archive)

yaml
Copy
Edit

---

## Methodology

- Data merged from 10+ NHANES modules: demographics, income, sleep, alcohol, physical activity, clinical labs
- Invalid responses removed (codes: 7, 9, 77, 99, 9999, etc.)
- Binary targets:
  - `DPQ020_binary`: Felt down, depressed, or hopeless
  - `DPQ060_binary`: Felt bad about yourself
- Models trained:
  - Logistic Regression (`class_weight='balanced'`)
  - XGBoost (`scale_pos_weight`)
  - Both also trained with **SMOTE**
- Evaluation:
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix
  - 5-fold Cross-Validation
  - SHAP value visualizations
  - Validation Curves (hyperparameter tuning)

---

## Results Summary

| Target        | Model              | Accuracy | Recall (Depressed) | SHAP Top Features                    |
|---------------|-------------------|----------|---------------------|--------------------------------------|
| DPQ020        | Logistic Regression | 62%     | 0.69                | Age, Sleep, Income, CRP, Vitamin D   |
|               | XGBoost            | 65%     | 0.38                | Age, HbA1c, CRP, Income              |
| DPQ060        | Logistic Regression | 66%     | 0.65                | Age, CRP, Sleep, Income              |
|               | XGBoost            | 69%     | 0.27                | Age, Sleep, Vitamin D, HbA1c         |

---

## Interpretation

- **Logistic Regression** offered better sensitivity (recall) for detecting depression cases.
- **XGBoost** performed slightly better on overall accuracy but struggled with minority class recall.
- **SHAP analysis** identified consistent predictors: age, income, CRP (inflammation), and sleep patterns.

---

## Limitations

- Targets based on **self-reported** PHQ-9 symptoms, not clinical diagnoses
- NHANES is **cross-sectional**, limiting temporal prediction
- No external validation set used
- SMOTE helped but could introduce synthetic overfitting if not validated properly

---

## References

- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- NIMH (2023). Major Depression Statistics
- Shatte et al. (2019). *Machine Learning in Mental Health*
- Sofia et al. (2023). *Depression Detection During COVID-19 Using ML*

---

## Tech Stack

- Python, Jupyter Notebook  
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `pyreadstat`  
- SMOTE via `imblearn`

