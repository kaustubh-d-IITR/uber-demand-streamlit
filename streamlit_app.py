import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Load the model and scaler
# ---------------------------
model = joblib.load("rf_demand_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🚖 Uber Demand Level Predictor")
st.markdown("Enter trip info to predict **demand level**: Medium (1) or High (2)")

# Input fields
hour = st.slider("Hour of Day", 0, 23, 12)
day = st.slider("Day of Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
lat = st.number_input("Latitude", value=40.7)
lon = st.number_input("Longitude", value=-73.9)
base = st.selectbox("Select Base", ["B02512", "B02598", "B02617", "B02682", "B02764"])

# ---------------------------
# Convert Base to One-Hot
# ---------------------------
base_cols = ["Base_B02598", "Base_B02617", "Base_B02682", "Base_B02764"]
base_values = [0, 0, 0, 0]

if base != "B02512":  # Because we dropped 'B02512' during one-hot encoding
    base_index = base_cols.index(f"Base_{base}")
    base_values[base_index] = 1

# ---------------------------
# Combine all features
# ---------------------------
input_data = np.array([[hour, day, day_of_week, month, lat, lon] + base_values])
input_scaled = scaler.transform(input_data)

# ---------------------------
# Make Prediction
# ---------------------------
if st.button("Predict Demand Level"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🧠 Predicted Demand Level: **{'High' if prediction == 2 else 'Medium'} ({prediction})**")
