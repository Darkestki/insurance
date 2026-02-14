import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# -----------------------------
# 🎨 Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Insurance Charge Predictor 💰",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# 💅 3D Card Style
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 📦 Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(BASE_DIR, "ann_insurance_model.h5")
    scaler_path = os.path.join(BASE_DIR, "standard_scaler.pkl")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

model, sc = load_artifacts()

# -----------------------------
# 🏥 App Title
# -----------------------------
st.title("🏥 Insurance Charge Prediction App")
st.write("Fill the details below to predict your insurance charges 💡")

# -----------------------------
# 👤 User Inputs
# -----------------------------
age = st.slider("🎂 Age", 18, 65, 30)
bmi = st.number_input("⚖ BMI", 15.0, 50.0, 25.0, step=0.1)
children = st.slider("👶 Number of Children", 0, 5, 1)
sex = st.selectbox("🧑 Gender", ["female", "male"])
smoker = st.selectbox("🚬 Smoker", ["no", "yes"])
region = st.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])

# -----------------------------
# 🔄 Preprocessing Function
# -----------------------------
def preprocess_input(age, bmi, children, sex, smoker, region, sc):

    input_encoded = pd.DataFrame(0, index=[0], columns=[
        'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
        'region_northwest', 'region_southeast', 'region_southwest'
    ])

    input_encoded['age'] = age
    input_encoded['bmi'] = bmi
    input_encoded['children'] = children

    if sex == 'male':
        input_encoded['sex_male'] = 1

    if smoker == 'yes':
        input_encoded['smoker_yes'] = 1

    if region == 'northwest':
        input_encoded['region_northwest'] = 1
    elif region == 'southeast':
        input_encoded['region_southeast'] = 1
    elif region == 'southwest':
        input_encoded['region_southwest'] = 1

    scaled_input = sc.transform(input_encoded)

    return scaled_input

# -----------------------------
# 🔮 Prediction Button
# -----------------------------
if st.button("🔍 Predict Insurance Charge"):

    scaled_input_data = preprocess_input(age, bmi, children, sex, smoker, region, sc)

    prediction = model.predict(scaled_input_data)
    predicted_charge = float(prediction[0][0])

    st.success(f"💰 Predicted Insurance Charge: ${predicted_charge:,.2f}")
    st.balloons()
