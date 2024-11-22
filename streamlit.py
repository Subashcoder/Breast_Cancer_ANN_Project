import pickle
import streamlit as st
import numpy as np

# Load the scaler, selector, and model
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('selector.pkl', 'rb') as selector_file:
    selector = pickle.load(selector_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# User input for all features
st.sidebar.header("Breast Cancer Prediction App")
st.sidebar.write("Enter values for features:")

features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension']

user_input = []
for feature in features: 
    value = st.sidebar.number_input(f"Enter value for {feature}", step=0.01)
    user_input.append(value)

if st.sidebar.button("Predict"):
    # Scale the input
    user_input_scaled = scaler.transform([user_input])

    # Apply feature selection
    user_input_selected = selector.transform(user_input_scaled)

    # Make a prediction
    prediction = model.predict(user_input_selected)
    prediction_proba = model.predict_proba(user_input_selected)

    # Display results
    st.write("### Prediction:")
    st.write("Malignant" if prediction[0] == 0 else "Benign")
    st.write("### Prediction Probability:")
    st.write(f"Malignant: {prediction_proba[0][0]:.2f}, Benign: {prediction_proba[0][1]:.2f}")