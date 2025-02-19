import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")  # One-hot encoded feature names
scaler = joblib.load("scaler.pkl")

# Check if scaler has feature names
if hasattr(scaler, "feature_names_in_"):
    scaler_features = [f for f in scaler.feature_names_in_ if f != "Price_USD"]
else:
    scaler_features = ["Year", "Month"]  # Replace with actual numeric columns used during training

# Streamlit App UI
st.title("Jewelry Price Prediction App")
st.markdown("Enter the jewelry details below to get a price prediction.")

# User Inputs
category = st.selectbox("Category", ["jewelry.earring", "jewelry.pendant", "jewelry.necklace", "jewelry.ring", 
                                     "jewelry.brooch", "jewelry.bracelet", "jewelry.souvenir", "jewelry.stud"])
main_metal = st.selectbox("Main Metal", ["gold", "silver", "platinum", "diamond"])
target_gender = st.selectbox("Target Gender", ["f", "m"])
main_color = st.selectbox("Main Color", ["red", "blue", "green", "black", "white"])
main_gem = st.text_input("Main Gem (if any, else leave blank)")
brand_id = st.number_input("Brand ID", min_value=0, step=1)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# Convert user inputs into a DataFrame
input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                        columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem", "Brand_ID", "Year", "Month"])

# Factorize categorical values using mappings
for col in ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]:
    if col in factorized_mappings:
        input_df[col] = input_df[col].map(factorized_mappings[col]).fillna(-1).astype(int)

# Apply One-Hot Encoding
input_df = pd.get_dummies(input_df)

# Ensure it has the same columns as training data
for col in one_hot_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns

# Reorder columns to match model input
input_df = input_df[one_hot_columns]

# Scale only numerical features
try:
    input_df[scaler_features] = scaler.transform(input_df[scaler_features])
except ValueError as e:
    st.error(f"Error scaling data: {e}")
    st.stop()

# Make prediction
prediction = model.predict(input_df)

# Display result
st.subheader("Predicted Jewelry Price (USD):")
st.write(f"${prediction[0]:,.2f}")
