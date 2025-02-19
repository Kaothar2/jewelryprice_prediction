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
main_metal = st.selectbox("Main Metal", ["gold", "silver", "platinum"])
target_gender = st.selectbox("Target Gender", ["Unknown", "f", "m"])
main_color = st.selectbox("Main Color", ["red", "white", "yellow", "Unknown-color"])
main_gem = st.selectbox("Main Gem", ['diamond', 'sapphire', 'amethyst', 'None', 'fianit', 'pearl',
       'quartz', 'topaz', 'garnet', 'quartz_smoky', 'ruby', 'agate',
       'mix', 'citrine', 'emerald', 'amber', 'chrysolite', 'chrysoprase',
       'nanocrystal', 'turquoise', 'sitall', 'corundum_synthetic',
       'coral', 'onyx', 'nacre', 'spinel', 'tourmaline',
       'emerald_geothermal', 'garnet_synthetic', 'rhodolite',
       'sapphire_geothermal'])
brand_id = st.number_input("Brand ID", min_value=-1, max_value=5.0, step=1)
year = st.number_input("Year", min_value=2017, max_value=2030, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    """Preprocess input data before making a prediction."""
    st.write("Raw Inputs:")
    st.write(f"Category: {category}, Main Metal: {main_metal}, Target Gender: {target_gender}, Main Color: {main_color}, Main Gem: {main_gem}, Brand ID: {brand_id}, Year: {year}, Month: {month}")
    # Create a DataFrame with user inputs
    input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                            columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem", "Brand_ID", "Year", "Month"])
    # Check DataFrame types before processing
    st.write("DataFrame before preprocessing:")
    st.write(input_df.dtypes)
    # Factorization (Replace categorical values with encoded mappings)
    for col in ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]:
        if col in factorized_mappings:
            input_df[col] = input_df[col].map(factorized_mappings[col]).fillna(-1)  # Use -1 for unknown categories
    # Ensure numeric values are in float format
    input_df[["Brand_ID", "Year", "Month"]] = input_df[["Brand_ID", "Year", "Month"]].astype(float)
    # Log transformed DataFrame
    st.write("DataFrame after factorization and type conversion:")
    st.write(input_df.dtypes)
    # Apply MinMaxScaler transformation (only to known scaler features)
    try:
        input_df[scaler_features] = scaler.transform(input_df[scaler_features])
    except ValueError as e:
        st.error(f"Error scaling data: {e}")
        st.write("Scaler features expected:", scaler.feature_names_in_)
        st.write("Scaler features present in input:", input_df.columns)
        return None  # Return None to indicate failure
    # One-hot encoding
    input_df = pd.get_dummies(input_df)
    # Align columns with trained model
    for col in one_hot_columns:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns as zero
    # Final log before prediction
    st.write("Final processed DataFrame:")
    st.write(input_df.dtypes)
    return input_df
