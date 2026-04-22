import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('iris_model.joblib')

# Page config
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Predict the type of Iris flower based on measurements")

# Sidebar inputs (clean UI)
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Prediction: {species[prediction[0]]}")

# Optional: show input values
st.subheader("Input Summary")
st.write({
    "Sepal Length": sepal_length,
    "Sepal Width": sepal_width,
    "Petal Length": petal_length,
    "Petal Width": petal_width
})