# Clinical Readmission Prediction

This project predicts whether a patient is likely to be readmitted to the hospital based on clinical and visit-related information.

The model was trained using a structured machine learning pipeline and deployed using **Streamlit on Hugging Face Spaces**.

**Live App:**  
https://huggingface.co/spaces/kishore-9/clinical-readmission-m

---

# Problem Statement

Hospital readmission is an important healthcare challenge. Patients who return to the hospital shortly after discharge may indicate unresolved medical issues or inadequate follow-up care.

The objective of this project is to build a **binary classification model** that predicts whether a patient will be readmitted to the hospital.

Target variable:

**readmitted → yes / no**

Primary evaluation focus:

- **Recall for the "yes" class** (detect high-risk patients)

Secondary metric:

- **F1-score**

The focus on recall is important because missing high-risk patients can lead to increased medical complications and healthcare costs.

---

# Dataset

The dataset used in this project is available on Kaggle:

https://www.kaggle.com/datasets/dubradave/hospital-readmissions

It contains **25,000 hospital encounter records with 17 features** describing patient conditions and hospital visit details.

## Feature Categories

### Patient Information
- Age group  
- Medical specialty  

### Hospital Visit Details
- Time in hospital  
- Number of lab procedures  
- Number of medications  
- Number of procedures  
- Outpatient visits  
- Inpatient visits  
- Emergency visits  

### Clinical Information
- Diagnosis categories (primary, secondary, tertiary)  
- Glucose test results  
- A1C test results  
- Medication change indicators  
- Diabetes medication indicator  

Target variable:

**readmitted (yes / no)**

---

# Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand patterns related to hospital readmission.

## Key Observations

- The dataset is **moderately balanced**, with approximately **53% non-readmitted patients and 47% readmitted patients**.
- Patients with **multiple previous inpatient visits** show a higher probability of readmission.
- Higher **number of medications** and **longer hospital stays** often correlate with increased readmission likelihood.
- **Older age groups** tend to have slightly higher readmission rates.
- Certain diagnosis categories appear more frequently among readmitted patients.

## Data Cleaning Steps

- Replaced `"Missing"` values in diagnosis columns with `"Unknown"`
- Capped extreme values using the **IQR outlier method**
- Verified **no duplicate rows**
- Confirmed **no missing (NaN) values**

The cleaned dataset was saved as:

`hospital_readmissions_cleaned.csv`

---

# Model Building

The machine learning pipeline followed these steps:

## 1. Train-Test Split

- 80% training data  
- 20% testing data  
- Stratified split to preserve class distribution  

## 2. Preprocessing

- `StandardScaler` for numerical features  
- `OneHotEncoder` for categorical features  
- Implemented using `ColumnTransformer`

## 3. Baseline Models

- Logistic Regression  
- Decision Tree  
- Random Forest  

## 4. Cross-Validation

- Stratified **5-Fold Cross Validation**

## 5. Hyperparameter Tuning

- Performed using **GridSearchCV**
- Focused on improving recall for the readmitted class

## 6. Model Selection

- Decision Tree achieved the best recall performance
- Selected as the final deployed model

### Final Test Performance

Approximate results on the test dataset:

- **Recall (readmitted = yes): ~0.55**
- **F1-score: ~0.52**

The trained pipeline was saved using:

`joblib.dump(best_pipeline, "hospital_model.pkl")`

---

# Deployment

The trained model pipeline (preprocessing + classifier) was deployed using **Streamlit** on **Hugging Face Spaces**.

The application allows users to:

- Enter patient clinical details
- Generate a prediction
- View the estimated probability of readmission

The deployment environment is automatically managed by Hugging Face Spaces.

---

# Project Structure
```
clinical-readmission-m/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── src/
├── streamlit_app.py
└── hospital_model.pkl

```


---

# Key Learning Outcomes

- End-to-end machine learning workflow
- Data exploration and preprocessing
- Avoiding data leakage using pipelines
- Model comparison and evaluation
- Hyperparameter tuning with cross-validation
- Model serialization using joblib
- Deployment using Streamlit and Hugging Face Spaces

---

# Note

This project is built for educational purposes and demonstration of machine learning workflows.

It should **not be used for real medical decision-making**.
