# Clinical Readmission Prediction

This project predicts whether a patient is likely to be readmitted to the hospital based on clinical and visit-related information.

The model was trained using a structured machine learning pipeline and deployed using Streamlit on Hugging Face Spaces.

Live App:
https://huggingface.co/spaces/kishore-9/clinical-readmission-m

---

## Problem Statement

Hospital readmission is an important healthcare challenge.  
Missing high-risk patients can increase medical complications and cost.

The objective of this project is to build a machine learning model that predicts:

readmitted → yes / no

Primary focus:  
- Recall for the "yes" class (identify high-risk patients)

Secondary metric:  
- F1-score

---

## Dataset Overview

The dataset contains 25,000 hospital records with 17 features including:

- Age group  
- Time in hospital  
- Lab procedures  
- Number of medications  
- Outpatient, inpatient, and emergency visits  
- Diagnosis categories  
- Glucose and A1C test results  
- Medication change indicators  

Target variable:
- readmitted (yes / no)

Basic data cleaning performed:
- Replaced "Missing" in diagnosis columns with "Unknown"
- Capped extreme outliers using IQR method
- Verified no duplicate rows
- No null values present

---

## Modeling Approach

Steps followed:

1. Train-test split using stratification
2. Preprocessing using:
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features
3. Built three baseline models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
4. Applied Stratified 5-fold cross-validation
5. Performed hyperparameter tuning using GridSearchCV
6. Selected best-performing model based on recall
7. Saved final pipeline using joblib

The final deployed model is a tuned Decision Tree.

Final Test Performance (approx):
- Recall (readmitted = yes): ~0.55
- F1-score: ~0.52

---

## Deployment

The trained pipeline (preprocessing + model) is saved as:

hospital_model.pkl

Deployment stack:
- Streamlit
- Docker
- Hugging Face Spaces

The Streamlit application:
- Accepts patient details as input
- Generates prediction
- Displays probability of readmission
- Provides a simple clinical-style interface

---

## Project Structure
``` text
clinical-readmission-ml/
│
├── Dockerfile
├── requirements.txt
└── src/
├── streamlit_app.py
└── hospital_model.pkl

```


---

## Key Learning Outcomes

- End-to-end ML workflow implementation
- Avoiding data leakage using pipelines
- Stratified cross-validation
- Hyperparameter tuning
- Model serialization
- Deployment debugging in Docker environment
- Production path handling for model loading

---

## Note

This project is built for academic and learning purposes.  
It should not be used for real medical decision-making.

---
