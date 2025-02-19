import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")  # One-hot encoded feature names
scaler = joblib.load("scaler.pkl")

# Extract feature names the scaler was trained on (excluding 'Price_USD' if it was included)
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
main_gem = st.selectbox("Main Gem", ["ruby", "sapphire", "emerald", "diamond", "none"])
brand_id = st.number_input("Brand ID", min_value=0, max_value=100, step=1)
year = st.number_input("Year", min_value=2000, max_value=2030, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# Preprocessing function
def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    # Create a DataFrame from user inputs
    input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                            columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem", "Brand_ID", "Year", "Month"])

    # Convert numerical columns to float
    input_df[["Year", "Month", "Brand_ID"]] = input_df[["Year", "Month", "Brand_ID"]].astype(float)

    # Apply factorization mapping
    for col in ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]:
        if col in factorized_mappings:
            input_df[col] = input_df[col].map(factorized_mappings[col]).fillna(-1)  # Handle unseen categories

    # Apply one-hot encoding (ensure all necessary columns exist)
    input_df = pd.get_dummies(input_df)
    for col in one_hot_columns:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns with 0s

    # Ensure column order matches training data
    input_df = input_df[one_hot_columns]

    # Scale only the relevant numeric features
    try:
        input_df[scaler_features] = scaler.transform(input_df[scaler_features])
    except ValueError as e:
        st.error(f"Error scaling data: {str(e)}")
        return None

    return input_df

# Prediction button
if st.button("Predict Price"):
    processed_data = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)
    
    if processed_data is not None:
        prediction = model.predict(processed_data)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")
    else:
        st.error("Prediction failed due to preprocessing errors.")
