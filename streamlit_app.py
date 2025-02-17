import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model, factorization mappings, and scaler
model = joblib.load("lightgbm_model.pkl")
factorized_mappings = joblib.load("factorized_mappings.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App UI
st.title("Jewelry Price Prediction App")
st.markdown("Enter the jewelry details below to get a price prediction.")

# User Inputs
category = st.selectbox("Category", factorized_mappings["Category"])
main_metal = st.selectbox("Main Metal", factorized_mappings["Main_Metal"])
target_gender = st.selectbox("Target Gender", factorized_mappings["Target_Gender"])
main_color = st.selectbox("Main Color", factorized_mappings["Main_Color"])
main_gem = st.selectbox("Main Gem", ["agate", "amber", "amethyst", "chrysolite", "chrysoprase", "citrine",
                                     "coral", "corundum_synthetic"])  # Example list
brand_id = st.selectbox("Brand ID", range(1, 8))  # Example range
year = st.number_input("Year", min_value=2000, max_value=2030, step=1, value=2023)
month = st.number_input("Month", min_value=1, max_value=12, step=1, value=6)

# Preprocessing the Input Data
def preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month):
    # Apply factorization encoding
    category_encoded = factorized_mappings["Category"].index(category) if category in factorized_mappings["Category"] else -1
    main_metal_encoded = factorized_mappings["Main_Metal"].index(main_metal) if main_metal in factorized_mappings["Main_Metal"] else -1
    target_gender_encoded = factorized_mappings["Target_Gender"].index(target_gender) if target_gender in factorized_mappings["Target_Gender"] else -1
    main_color_encoded = factorized_mappings["Main_Color"].index(main_color) if main_color in factorized_mappings["Main_Color"] else -1

    # One-Hot Encode main_gem and brand_id
    input_df = pd.DataFrame([[main_gem, brand_id]], columns=["Main_Gem", "Brand_ID"])
    one_hot_encoded = one_hot_encoder.transform(input_df).toarray()

    # Scale numerical features
    scaled_features = scaler.transform([[year, month]])[0]

    # Combine all features
    final_features = np.concatenate(([category_encoded, main_metal_encoded, target_gender_encoded, main_color_encoded], one_hot_encoded, scaled_features))
    
    return final_features.reshape(1, -1)

# Prediction
if st.button("Predict Price"):
    input_data = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"Predicted Price (USD): ${predicted_price:.2f}")
