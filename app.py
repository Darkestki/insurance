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
# 💅 Dynamic Theme Style
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
}
.stButton>button {
    background: linear-gradient(90deg, #00c853, #64dd17);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.3);
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 📦 Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = load_model(os.path.join(BASE_DIR, "ann_insurance_model.h5"))
    scaler = joblib.load(os.path.join(BASE_DIR, "standard_scaler.pkl"))
    return model, scaler

model, sc = load_artifacts()

# -----------------------------
# 🏥 Title
# -----------------------------
st.title("🏥 Smart Insurance Charge Predictor")
st.markdown("### AI Based Medical Cost Estimator 💡")

# -----------------------------
# 👤 User Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 65, 30)
    bmi = st.number_input("⚖ BMI", 15.0, 50.0, 25.0, step=0.1)
    children = st.slider("👶 Children", 0, 5, 1)

with col2:
    sex = st.selectbox("🧑 Gender", ["female", "male"])
    smoker = st.selectbox("🚬 Smoker", ["no", "yes"])
    region = st.selectbox("🌍 Region", 
                          ["southwest", "southeast", "northwest", "northeast"])

# -----------------------------
# 🔄 Preprocessing
# -----------------------------
def preprocess_input(age, bmi, children, sex, smoker, region):

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

    return sc.transform(input_encoded)

# -----------------------------
# 🎯 Risk Calculator
# -----------------------------
def calculate_risk(age, bmi, smoker):
    score = 0
    if age > 50:
        score += 1
    if bmi > 30:
        score += 1
    if smoker == "yes":
        score += 2
    return score

# -----------------------------
# 🔮 Prediction
# -----------------------------
if st.button("🔍 Predict Insurance Charge"):

    scaled_input = preprocess_input(age, bmi, children, sex, smoker, region)
    prediction = model.predict(scaled_input)
    predicted_charge = float(prediction[0][0])

    risk_score = calculate_risk(age, bmi, smoker)

    # 🎨 Dynamic Risk Display
    if risk_score <= 1:
        st.success(f"💰 Estimated Charge: ${predicted_charge:,.2f}")
        st.info("🟢 Low Risk Profile")
    elif risk_score == 2:
        st.warning(f"💰 Estimated Charge: ${predicted_charge:,.2f}")
        st.warning("🟡 Medium Risk Profile")
    else:
        st.error(f"💰 Estimated Charge: ${predicted_charge:,.2f}")
        st.error("🔴 High Risk Profile")

    # 📊 Risk Meter
    st.progress(min(risk_score * 30, 100))

    # 🎈 Celebration
    if predicted_charge < 10000:
        st.balloons()
    else:
        st.snow()
