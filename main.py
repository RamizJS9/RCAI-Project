import streamlit as st
import numpy as np
import pickle

# Load the trained model
import os
loaded_model = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Diabetesmodel.pkl')
with open(MODEL_PATH, 'rb') as f:
    loaded_model = pickle.load(f)

# Prediction function
def predict_diabetes(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_reshape)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

# Streamlit UI
st.set_page_config(page_title="AI Diabetes Risk Assessment", layout="centered")
st.title("🩺 AI Diabetes Risk Assessment")
st.write("Enter your health details below:")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, step=0.1)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Predict button
if st.button("Assess Diabetes Risk"):
    result = predict_diabetes((glucose, blood_pressure, bmi, age))
    st.success(result)
