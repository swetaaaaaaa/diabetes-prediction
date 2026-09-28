import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load("diabetes_pipeline.pkl")
st.set_page_config(page_title="Diabetes Prediction",page_icon="🩺")


feature_columns = pipeline.feature_names_in_


st.title("🩺 Diabetes Risk Prediction")

st.write("Enter the patient's information below.")

st.info("This application is an educational machine-learning project and is not a medical diagnostic tool.")

st.header("Patient Information")


pregnancies = st.number_input("Pregnancies",min_value=0,max_value=20,value=1)

glucose = st.number_input("Glucose",min_value=0.0,max_value=300.0,value=120.0)

blood_pressure = st.number_input("Blood Pressure",min_value=0.0,max_value=200.0,value=70.0)

skin_thickness = st.number_input("Skin Thickness",min_value=0.0,max_value=100.0,value=20.0)

insulin = st.number_input("Insulin",min_value=0.0,max_value=1000.0,value=80.0)

bmi = st.number_input("BMI",min_value=0.0,max_value=70.0,value=25.0)

diabetes_pedigree = st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0,value=0.5)

age = st.number_input("Age",min_value=1,max_value=120,value=30)

input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI":[bmi],
        "DiabetesPedigreeFunction":[diabetes_pedigree],
        "Age": [age],
})

if st.button("Predict Diabetes Risk",type="primary"):
    prediction = pipeline.predict(input_data)[0]

    probability = pipeline.predict_proba(input_data)[0]
    prob_no_diabetes = probability[0]
    prob_diabetes = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts the diabetes-positive class.")
    else:
        st.success("The model predicts the diabetes-negative class.")
    st.write(f"Probability of no diabetes :{prob_no_diabetes*100:.2f}%")
    st.write(f"Probability of diabetes :{prob_diabetes*100:.2f}%")

