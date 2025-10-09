import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load trained model, scaler, and feature list ---
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_list = joblib.load("features.pkl")
except FileNotFoundError:
    st.error("Model, scaler, or feature list file not found. Please check your deployment files.")

# --- Prediction Function ---
def predict_price(area, rooms, bathrooms, floors, city_center_distance, has_elevator, has_parking, has_balcony):
    # Create input dictionary
    input_dict = {
        'area': area,
        'rooms': rooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'cityCenterDistance': city_center_distance,
        'hasElevator': has_elevator,
        'hasParking': has_parking,
        'hasBalcony': has_balcony
    }

    # Create DataFrame with all expected features
    input_data = pd.DataFrame([input_dict])

    # Add missing columns with default value
    for col in feature_list:
        if col not in input_data.columns:
            input_data[col] = 0  # Default value for missing features

    # Reorder columns to match training
    input_data = input_data[feature_list]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return f"🏠 Estimated House Price: €{prediction[0]:,.2f}"

# --- Streamlit Interface ---
st.set_page_config(page_title="Paris Housing Predictor", page_icon="🏡")
st.title("🏡 Paris Housing Price Predictor")
st.write("Enter the property details below to predict the estimated housing price.")

# --- Input Fields ---
area = st.number_input("Area (sq meters)", min_value=10.0, max_value=1000.0, value=100.0)
rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=5, value=2)
floors = st.number_input("Number of floors", min_value=1, max_value=10, value=1)
city_center_distance = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, value=5.0)
has_elevator = st.radio("Has Elevator?", ["No", "Yes"])
has_parking = st.radio("Has Parking?", ["No", "Yes"])
has_balcony = st.radio("Has Balcony?", ["No", "Yes"])

# Convert radio inputs to binary
has_elevator = 1 if has_elevator == "Yes" else 0
has_parking = 1 if has_parking == "Yes" else 0
has_balcony = 1 if has_balcony == "Yes" else 0

# --- Prediction Button ---
if st.button("Predict Price 💰"):
    result = predict_price(area, rooms, bathrooms, floors, city_center_distance, has_elevator, has_parking, has_balcony)
    st.success(result)

# --- Footer ---
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning")
