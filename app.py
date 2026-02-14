import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
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
# 💅 3D Styling
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
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
# 🏥 Title
# -----------------------------
st.title("🏥 Insurance Charge Prediction Dashboard")
st.write("Adjust the inputs below and see real-time premium prediction 💡")

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
    region = st.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])

# -----------------------------
# 🔄 Preprocessing
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
# 🔮 Dynamic Prediction
# -----------------------------
scaled_input_data = preprocess_input(age, bmi, children, sex, smoker, region, sc)

prediction = model.predict(scaled_input_data)
predicted_charge = float(prediction[0][0])

# -----------------------------
# 📊 Risk Level Logic
# -----------------------------
if predicted_charge < 10000:
    risk = "Low Risk ✅"
    color = "#2E8B57"
elif predicted_charge < 30000:
    risk = "Medium Risk ⚠"
    color = "#FFA500"
else:
    risk = "High Risk 🔥"
    color = "#FF4B4B"

# -----------------------------
# 💰 3D Result Card
# -----------------------------
st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}, #222);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 5px 5px 20px rgba(0,0,0,0.4);
        margin-top: 20px;
    ">
        💰 Predicted Insurance Charge <br><br>
        ${predicted_charge:,.2f} <br><br>
        {risk}
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# 📈 Comparison Chart
# -----------------------------
st.subheader("📊 Premium Comparison")

average_charge = 20000  # You can adjust based on dataset mean

fig = plt.figure()
plt.bar(["Your Premium", "Average Premium"], 
        [predicted_charge, average_charge])
plt.ylabel("Charge ($)")
plt.title("Premium Comparison")
st.pyplot(fig)
