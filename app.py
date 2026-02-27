import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("hospital_model.pkl")

st.set_page_config(page_title="Clinical Readmission Predictor", layout="wide")

st.title("Hospital Readmission Prediction")
st.caption("Predict the likelihood of patient readmission based on clinical information.")

st.markdown("---")

# ===============================
# Patient Details Section
# ===============================

with st.expander("Patient Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        )

        medical_specialty = st.selectbox(
            "Medical Specialty",
            [
                "InternalMedicine",
                "Emergency/Trauma",
                "Family/GeneralPractice",
                "Cardiology",
                "Surgery-General",
                "Orthopedics",
                "Other",
                "Unknown"
            ]
        )

    with col2:
        diagnosis_options = [
            "Circulatory",
            "Respiratory",
            "Digestive",
            "Diabetes",
            "Injury",
            "Musculoskeletal",
            "Genitourinary",
            "Other",
            "Unknown"
        ]

        diag_1 = st.selectbox("Primary Diagnosis", diagnosis_options)
        diag_2 = st.selectbox("Secondary Diagnosis", diagnosis_options)
        diag_3 = st.selectbox("Tertiary Diagnosis", diagnosis_options)

        glucose_test = st.selectbox(
            "Glucose Test Result",
            ["no", "normal", ">200", ">300"]
        )

        A1Ctest = st.selectbox(
            "A1C Test Result",
            ["no", "normal", ">7", ">8"]
        )

        change = st.selectbox("Medication Change", ["no", "yes"])
        diabetes_med = st.selectbox("Diabetes Medication", ["no", "yes"])


# ===============================
# Hospital Visit Details
# ===============================

st.markdown("### Hospital Visit Details")

col3, col4 = st.columns(2)

with col3:
    time_in_hospital = st.number_input("Time in Hospital (days)", 1, 20, 3)
    n_lab_procedures = st.number_input("Number of Lab Procedures", 1, 150, 40)
    n_procedures = st.number_input("Number of Procedures", 0, 10, 1)
    n_medications = st.number_input("Number of Medications", 1, 100, 10)

with col4:
    n_outpatient = st.number_input("Outpatient Visits", 0, 50, 0)
    n_inpatient = st.number_input("Inpatient Visits", 0, 20, 0)
    n_emergency = st.number_input("Emergency Visits", 0, 20, 0)

st.markdown("---")

# ===============================
# Prediction
# ===============================

if st.button("Predict Readmission Risk"):

    input_data = pd.DataFrame({
        "age": [age],
        "time_in_hospital": [time_in_hospital],
        "n_lab_procedures": [n_lab_procedures],
        "n_procedures": [n_procedures],
        "n_medications": [n_medications],
        "n_outpatient": [n_outpatient],
        "n_inpatient": [n_inpatient],
        "n_emergency": [n_emergency],
        "medical_specialty": [medical_specialty],
        "diag_1": [diag_1],
        "diag_2": [diag_2],
        "diag_3": [diag_3],
        "glucose_test": [glucose_test],
        "A1Ctest": [A1Ctest],
        "change": [change],
        "diabetes_med": [diabetes_med],
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## Prediction Result")

    st.progress(float(probability))

    if prediction == "yes":
        st.error("High Risk of Readmission")
    else:
        st.success("Low Risk of Readmission")

    st.write(f"Estimated Probability of Readmission: {probability:.2f}")