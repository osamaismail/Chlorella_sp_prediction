# streamlit_app.py

import streamlit as st
import joblib

# Load the saved Random Forest model
model_path = "random_forest_model.pkl"
loaded_model = joblib.load(model_path)

def predict_growth_rate(day):
    """Predict the growth rate using the loaded model."""
    predicted_growth_rate = loaded_model.predict([[day]])
    return predicted_growth_rate[0]

# Streamlit app
st.title("Chlorella sp. Growth Rate Predictor")

# Get user input for the day
day = st.slider("Select the day for which you want to predict the growth rate of Chlorella sp. after treatment:", 0, 100)  # Adjust range as needed

if st.button("Predict"):
    prediction = predict_growth_rate(day)
    st.write(f"The predicted growth rate of Chlorella sp. after treatment for day {day} is: {prediction}")

