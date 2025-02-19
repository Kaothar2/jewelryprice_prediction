import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, factorization mappings, scaler, and one-hot columns
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
scaler = joblib.load("scaler.pkl")
one_hot_columns = joblib.load("one_hot_columns.pkl")

# Create input widgets for the app
st.title("Jewelry Price Prediction")

# Target Gender
target_gender = st.selectbox("Target Gender", options=factorized_mappings["Target_Gender"])

# Main Color
main_color = st.selectbox("Main Color", options=factorized_mappings["Main_Color"])

# Main Metal
main_metal = st.selectbox("Main Metal", options=factorized_mappings["Main_Metal"])

# Main Gem
main_gem = st.selectbox("Main Gem", options=[gem for gem in one_hot_columns if gem.startswith("Main_Gem_")])

# Category
category = st.selectbox("Category", options=factorized_mappings["Category"])

# Year
year = st.number_input("Year", min_value=2015, max_value=2025, value=2023)

# Month
month = st.number_input("Month", min_value=1, max_value=12, value=1)

# Create a button to trigger prediction
if st.button("Predict Price"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        "Target_Gender": [target_gender],
        "Main_Color": [main_color],
        "Main_Metal": [main_metal],
        "Category": [category],
        "Year": [year],
        "Month": [month]
    })

    # Add one-hot encoded columns for Main Gem
    for gem in [gem for gem in one_hot_columns if gem.startswith("Main_Gem_")]:
        input_data[gem] = 0  # Initialize to 0
    input_data[main_gem] = 1  # Set the selected gem to 1

    # Scale numerical features
    input_data[["Year", "Month"]] = scaler.transform(input_data[["Year", "Month"]])

    # Reorder columns to match training data
    input_data = input_data[one_hot_columns[:-1]]  # Exclude Price_USD

    # Predict the price
    predicted_price = model.predict(input_data)[0]

    # Display the prediction
    st.success(f"Predicted Price (USD): {predicted_price:.2f}")
