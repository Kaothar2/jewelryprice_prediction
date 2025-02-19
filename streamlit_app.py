import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")  # Load one-hot columns instead of encoder
scaler = joblib.load("scaler.pkl")

# Streamlit App UI
st.title("Jewelry Price Prediction App")
st.markdown("Enter the jewelry details below to get a price prediction.")

# User Inputs
category = st.selectbox("Category", ["jewelry.earring", "jewelry.pendant", "jewelry.necklace", "jewelry.ring", 
                                     "jewelry.brooch", "jewelry.bracelet", "jewelry.souvenir", "jewelry.stud"])
main_metal = st.selectbox("Main Metal", ["gold", "silver", "platinum", "diamond"])
target_gender = st.selectbox("Target Gender", ["f", "m"])
main_color = st.selectbox("Main Color", ["red", "blue", "green", "black", "white"])
main_gem = st.selectbox("Main Gem", ["diamond", "ruby", "emerald", "sapphire", "none"])
brand_id = st.number_input("Brand ID", min_value=1, max_value=100, step=1)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# Function to preprocess input
def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    # Create a DataFrame from user inputs
    input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                            columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", 
                                     "Main_Gem", "Brand_ID", "Year", "Month"])

    # Ensure numeric columns are in float format
    input_df[["Year", "Month"]] = input_df[["Year", "Month"]].astype(float)

    # Check if MinMaxScaler has been fitted
    if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        st.error("The MinMaxScaler has not been fitted. Please ensure it is trained before use.")
        return None

    # Apply MinMaxScaler transformation
    try:
        input_df[["Year", "Month"]] = scaler.transform(input_df[["Year", "Month"]])
    except ValueError as e:
        st.error(f"Error scaling data: {e}")
        return None

    # Apply factorization (if mappings exist)
    categorical_columns = ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]
    for col in categorical_columns:
        if col in factorized_mappings:
            input_df[col] = input_df[col].map(factorized_mappings[col]).fillna(-1).astype(int)

    # One-hot encode missing columns to match training data
    input_df = pd.get_dummies(input_df)
    missing_cols = set(one_hot_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing one-hot encoded columns with 0 value

    input_df = input_df[one_hot_columns]  # Ensure column order matches training
    return input_df

# Predict button
if st.button("Predict Price"):
    processed_input = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)
    
    if processed_input is not None:
        # Convert to NumPy array and make a prediction
        input_array = processed_input.to_numpy()
        prediction = model.predict(input_array)[0]
        
        st.success(f"Predicted Jewelry Price: ${prediction:,.2f}")
