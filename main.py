import pickle
import numpy as np
import streamlit as st

# Load the pre-trained model
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

def predict_purchase(gender, age, salary):
    gender_map = {'male': 0, 'female': 1}
    input_data = np.array([[gender_map[gender.lower()], age, salary]])
    prediction = svm_model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Purchase Prediction App")
st.write("Predict whether a user will make a purchase based on gender, age, and estimated salary.")

# Input fields
gender = st.radio("Select Gender:", ('Male', 'Female'))
age = st.number_input("Enter Age:", min_value=1, max_value=100, step=1)
salary = st.number_input("Enter Estimated Salary:", min_value=0.0, step=100.0, format="%.2f")

# Prediction button
if st.button("Predict"):
    prediction = predict_purchase(gender, age, salary)
    result = "Purchased" if prediction == 1 else "Not Purchased"
    st.success(f"Prediction: {result}")
