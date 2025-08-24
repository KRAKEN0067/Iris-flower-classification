import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classifier")

st.write("Adjust the sliders to set flower measurements:")

# Use sliders instead of number inputs
SepalLengthCm = st.slider("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
SepalWidthCm  = st.slider("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)
PetalLengthCm = st.slider("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
PetalWidthCm  = st.slider("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)

# Predict button
if st.button("Predict"):
    sample = np.array([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    prediction = model.predict(sample)

    # iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    st.success(f"🌼 The model predicts: **{prediction[0]}**")
