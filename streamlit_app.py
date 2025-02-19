import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")  # Loaded one-hot columns
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
brand_id = st.number_input("Brand ID", min_value=1, step=1)
year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)

# Preprocessing function
def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    """Prepares the user input for model prediction."""
    
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([[category, main_metal, target_gender, main_color, main_gem, brand_id, year, month]],
                            columns=["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem", "Brand_ID", "Year", "Month"])

    # Convert Year and Month to float (MinMaxScaler expects numerical data)
    input_df[["Year", "Month"]] = input_df[["Year", "Month"]].astype(float)

    # Apply MinMaxScaler transformation only to numerical columns (Year & Month)
    try:
        input_df[["Year", "Month"]] = scaler.transform(input_df[["Year", "Month"]])
    except ValueError as e:
        st.error(f"Error scaling data: {e}")
        return None
    
    # Factorize categorical variables using pre-saved mappings
    for col in ["Category", "Main_Metal", "Target_Gender", "Main_Color", "Main_Gem"]:
        if col in factorized_mappings:
            if input_df[col][0] in factorized_mappings[col]:
                input_df[col] = factorized_mappings[col][input_df[col][0]]
            else:
                st.warning(f"Unknown category '{input_df[col][0]}' for {col}, using default.")
                input_df[col] = -1  # Assign a default value for unseen categories

    # One-hot encode missing columns
    input_df = pd.get_dummies(input_df)
    
    # Ensure all one-hot encoded columns match the training columns
    for col in one_hot_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with 0 value

    # Reorder columns to match training set
    input_df = input_df[one_hot_columns]

    return input_df

# Predict button
if st.button("Predict Price"):
    processed_input = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)

    if processed_input is not None:
        prediction = model.predict(processed_input)
        st.success(f"Estimated Price: ${prediction[0]:,.2f}")
    else:
        st.error("Prediction failed due to preprocessing errors.")
