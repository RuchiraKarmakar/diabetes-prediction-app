import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details below:")

col1, col2 = st.columns(2)

with col1:
    preg = st.text_input("Pregnancies")
    glu = st.text_input("Glucose")
    bp = st.text_input("Blood Pressure")
    skin = st.text_input("Skin Thickness")

with col2:
    ins = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    dpf = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")

if st.button("Predict Diabetes Risk"):
    try:
        input_data = pd.DataFrame([[
            float(preg), float(glu), float(bp), float(skin),
            float(ins), float(bmi), float(dpf), float(age)
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]

        if prediction[0] == 1:
            st.error("High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

        st.write(f"Risk Probability: {probability*100:.2f}%")

    except:
        st.warning("Please enter valid numeric values.")

