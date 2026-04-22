import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('iris_model.joblib')

# Page config
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e6f2ff;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🌸 Iris Flower Classifier")
st.caption("Enter the flower measurements to predict the species")

# Layout: 2 columns
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)

with col2:
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Predict button
st.write("")
if st.button("🔍 Predict"):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(data)

    species = ["Setosa", "Versicolor", "Virginica"]
    result = species[prediction[0]]

    # Display result nicely
    st.markdown(f"""
        <div class="result-box">
            🌼 Prediction: {result}
        </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown("---")

# Input summary (clean table)
st.subheader("📋 Input Summary")
st.table({
    "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Value": [sepal_length, sepal_width, petal_length, petal_width]
})
