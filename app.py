import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained model and scaler
@st.cache_resource
def load_artifacts():
    model = load_model('ann_insurance_model.h5')
    scaler = joblib.load('standard_scaler.pkl')
    return model, scaler

model, sc = load_artifacts()

st.title('Insurance Charge Prediction App')
st.write('Enter the details below to predict the insurance charge.')

# User input fields
age = st.slider('Age', 18, 65, 30)
bmi = st.number_input('BMI', 15.0, 50.0, 25.0, step=0.1)
children = st.slider('Number of Children', 0, 5, 1)
sex = st.selectbox('Sex', ['female', 'male'])
smoker = st.selectbox('Smoker', ['no', 'yes'])
region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

# Preprocess input
def preprocess_input(age, bmi, children, sex, smoker, region, sc):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    })

    # Apply one-hot encoding for categorical features
    # Ensure consistent column names and order as during training
    # The original training data X had columns: age, bmi, children, sex_male, smoker_yes, region_northwest, region_southeast, region_southwest
    # Initialize dummy variables to False (or 0)
    input_encoded = pd.DataFrame(0, index=input_data.index, columns=[
        'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
        'region_northwest', 'region_southeast', 'region_southwest'
    ])
    
    input_encoded['age'] = input_data['age']
    input_encoded['bmi'] = input_data['bmi']
    input_encoded['children'] = input_data['children']

    if sex == 'male':
        input_encoded['sex_male'] = True
    
    if smoker == 'yes':
        input_encoded['smoker_yes'] = True
        
    if region == 'northwest':
        input_encoded['region_northwest'] = True
    elif region == 'southeast':
        input_encoded['region_southeast'] = True
    elif region == 'southwest':
        input_encoded['region_southwest'] = True
    # 'northeast' implies all region dummies are False, which is the default

    # Scale numerical features (age, bmi, children already in input_encoded)
    # Note: StandardScaler expects numpy array, and transforms all columns it was fitted on.
    # The loaded 'sc' was fitted on X (which includes dummy variables). 
    # So we need to ensure input_encoded has the same columns as the X from training and in the same order.
    
    # Reorder columns to match the training data X if necessary.
    # Assuming the order is consistent with how get_dummies created X.
    # For robustness, you might want to store X.columns during training and use that here.
    # For now, let's assume the generated input_encoded has the correct order based on our construction.

    # Convert boolean columns to int for scaling if the scaler expects numerical input
    for col in ['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']:
        input_encoded[col] = input_encoded[col].astype(int)

    # Scale the input data
    scaled_input = sc.transform(input_encoded)
    
    return scaled_input


if st.button('Predict Insurance Charge'):
    scaled_input_data = preprocess_input(age, bmi, children, sex, smoker, region, sc)
    predicted_charge = model.predict(scaled_input_data)[0][0]
    
    st.success(f'Predicted Insurance Charge: ${predicted_charge:.2f}')
