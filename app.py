import streamlit as st
import joblib
import pandas as pd

# --- Load trained model, scaler, and feature list ---
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_list = joblib.load("features.pkl")
except FileNotFoundError:
    st.error("Model, scaler, or feature list file not found. Please check your deployment files.")
    st.stop()

# --- Prediction Function ---
def predict_price(
    square_meters, number_of_rooms, has_yard, has_pool, floors, city_code,
    city_part_range, num_prev_owners, made, is_new_built, has_storm_protector,
    basement, attic, garage, has_storage_room, has_guest_room
):
    input_dict = {
        'squareMeters': square_meters,
        'numberOfRooms': number_of_rooms,
        'hasYard': has_yard,
        'hasPool': has_pool,
        'floors': floors,
        'cityCode': city_code,
        'cityPartRange': city_part_range,
        'numPrevOwners': num_prev_owners,
        'made': made,
        'isNewBuilt': is_new_built,
        'hasStormProtector': has_storm_protector,
        'basement': basement,
        'attic': attic,
        'garage': garage,
        'hasStorageRoom': has_storage_room,
        'hasGuestRoom': has_guest_room
    }

    input_data = pd.DataFrame([input_dict])

    # Align features
    for col in feature_list:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_list]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    # Convert Euro to INR
    euro_to_inr = 89.0
    price_in_inr = prediction[0] * euro_to_inr
    return f"🏠 Estimated House Price: ₹{price_in_inr:,.2f}"

# --- Streamlit Interface ---
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡")
st.title("🏡 Housing Price Predictor")
st.write("Enter the property details below to predict the estimated housing price in INR.")

# --- Input Fields ---
square_meters = st.number_input("Area (sq meters)", min_value=10, max_value=1000, value=100)
number_of_rooms = st.number_input("Number of rooms", min_value=1, max_value=10, value=3)
has_yard = st.radio("Has Yard?", ["No", "Yes"])
has_pool = st.radio("Has Pool?", ["No", "Yes"])
floors = st.number_input("Number of floors", min_value=1, max_value=10, value=1)
city_code = st.number_input("City Code", min_value=0, max_value=100, value=5)
city_part_range = st.number_input("City Part Range", min_value=0, max_value=10, value=3)
num_prev_owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1)
made = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
is_new_built = st.radio("Is Newly Built?", ["No", "Yes"])
has_storm_protector = st.radio("Has Storm Protector?", ["No", "Yes"])
basement = st.radio("Has Basement?", ["No", "Yes"])
attic = st.radio("Has Attic?", ["No", "Yes"])
garage = st.radio("Has Garage?", ["No", "Yes"])
has_storage_room = st.radio("Has Storage Room?", ["No", "Yes"])
has_guest_room = st.radio("Has Guest Room?", ["No", "Yes"])

# Convert radio inputs to binary
has_yard = 1 if has_yard == "Yes" else 0
has_pool = 1 if has_pool == "Yes" else 0
is_new_built = 1 if is_new_built == "Yes" else 0
has_storm_protector = 1 if has_storm_protector == "Yes" else 0
basement = 1 if basement == "Yes" else 0
attic = 1 if attic == "Yes" else 0
garage = 1 if garage == "Yes" else 0
has_storage_room = 1 if has_storage_room == "Yes" else 0
has_guest_room = 1 if has_guest_room == "Yes" else 0

# --- Prediction Button ---
if st.button("Predict Price 💰"):
    result = predict_price(
        square_meters, number_of_rooms, has_yard, has_pool, floors, city_code,
        city_part_range, num_prev_owners, made, is_new_built, has_storm_protector,
        basement, attic, garage, has_storage_room, has_guest_room
    )
    st.success(result)

# --- Footer ---
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Machine Learning")
