import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Function to safely load files
def load_file(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"Error: {file_name} not found!")
        return None

# Load all necessary files
model = load_file("lightgbm_model.pkl")
factorized_mappings = load_file("factorized_mappings.pkl")
one_hot_columns = load_file("one_hot_columns.pkl")
scaler = load_file("scaler.pkl")

# Ensure all files are loaded successfully
if not all([model, factorized_mappings, one_hot_columns, scaler]):
    st.stop()

# Streamlit UI
st.title("Jewelry Price Prediction")
st.write("Enter the jewelry details to predict the price.")

# User input fields
year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
month = st.selectbox("Month", list(range(1, 13)))

# Categorical inputs
main_metal = st.selectbox("Main Metal", factorized_mappings["Main_Metal"])
target_gender = st.selectbox("Target Gender", factorized_mappings["Target_Gender"])
category = st.selectbox("Category", factorized_mappings["Category"])
main_color = st.selectbox("Main Color", factorized_mappings["Main_Color"])
main_gem = st.text_input("Main Gem", "Diamond")
brand_id = st.text_input("Brand ID", "BrandX")

# Convert categorical inputs to numerical (factorized encoding)
def encode_categorical(value, mapping):
    return mapping.index(value) if value in mapping else -1

main_metal_encoded = encode_categorical(main_metal, factorized_mappings["Main_Metal"])
target_gender_encoded = encode_categorical(target_gender, factorized_mappings["Target_Gender"])
category_encoded = encode_categorical(category, factorized_mappings["Category"])
main_color_encoded = encode_categorical(main_color, factorized_mappings["Main_Color"])

# Create input dataframe
input_data = pd.DataFrame({
    "Year": [year],
    "Month": [month],
    "Main_Metal": [main_metal_encoded],
    "Target_Gender": [target_gender_encoded],
    "Category": [category_encoded],
    "Main_Color": [main_color_encoded],
    "Main_Gem": [main_gem],
    "Brand_ID": [brand_id]
})

# One-hot encode Main_Gem and Brand_ID
input_data = pd.get_dummies(input_data, columns=["Main_Gem", "Brand_ID"])

# Ensure columns match the model's expected features
missing_cols = set(one_hot_columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0  # Add missing columns with zero value

# Reorder columns
input_data = input_data[one_hot_columns]

# Scale numerical features
scaled_data = scaler.transform(input_data)

# Predict price
if st.button("Predict Price"):
    price_prediction = model.predict(scaled_data)[0]
    st.success(f"Estimated Price: ${price_prediction:.2f}")
