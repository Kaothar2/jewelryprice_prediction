import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")  # List of one-hot encoded feature names
scaler = joblib.load("scaler.pkl")

# Extract the correct feature names for scaling (excluding 'Price_USD' if it was included before)
scaler_features = [f for f in scaler.feature_names_in_ if f != "Price_USD"]

# Streamlit App UI
st.title("Jewelry Price Prediction App")
st.markdown("Enter the jewelry details below to get a price prediction.")

# User Inputs
category = st.selectbox("Category", ["jewelry.earring", "jewelry.pendant", "jewelry.necklace", "jewelry.ring", 
                                     "jewelry.brooch", "jewelry.bracelet", "jewelry.souvenir", "jewelry.stud"])
main_metal = st.selectbox("Main Metal", ["gold", "silver", "platinum", "diamond"])
target_gender = st.selectbox("Target Gender", ["f", "m"])
main_color = st.selectbox("Main Color", ["red", "blue", "green", "black", "white"])
main_gem = st.selectbox("Main Gem", ["diamond", "emerald", "sapphire", "ruby", "none"])
brand_id = st.number_input("Brand ID", min_value=1, max_value=1000, step=1)
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# **Preprocessing Function**
def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    """Preprocess input data before making a prediction."""
    
    # Create a DataFrame with user inputs
    input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                            columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem", "Brand_ID", "Year", "Month"])
    
    # Factorization (Replace categorical values with encoded mappings)
    for col in ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]:
        if col in factorized_mappings:
            input_df[col] = input_df[col].map(factorized_mappings[col]).fillna(-1)  # Use -1 for unknown categories
    
    # Ensure numeric values are in float format
    input_df[["Brand_ID", "Year", "Month"]] = input_df[["Brand_ID", "Year", "Month"]].astype(float)

    # Apply MinMaxScaler transformation (only to known scaler features)
    try:
        input_df[scaler_features] = scaler.transform(input_df[scaler_features])
    except ValueError as e:
        st.error(f"Error scaling data: {e}")
        return None  # Return None to indicate failure

    # One-hot encoding (ensure missing columns are handled correctly)
    input_df = pd.get_dummies(input_df)

    # Align columns with trained model (ensure all necessary columns exist)
    for col in one_hot_columns:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns as zero

    # Reorder columns to match training data
    input_df = input_df[one_hot_columns]

    return input_df

# **Prediction Function**
def predict_price():
    """Run prediction when user clicks the button."""
    input_data = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)

    if input_data is None:
        st.error("Prediction failed due to preprocessing errors.")
        return

    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")

# **Run Prediction**
if st.button("Predict Price"):
    predict_price()
