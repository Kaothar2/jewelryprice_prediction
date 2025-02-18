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
category = st.selectbox("Category", ["jewelry.earring", "jewelry.pendant", "jewelry.necklace", "jewelry.ring", "jewelry.brooch", "jewelry.bracelet", "jewelry.souvenir", "jewelry.stud"])
main_metal = st.selectbox("Main Metal", ["gold", "silver", "platinum","diamond"])
target_gender = st.selectbox("Target Gender", ["f", "m"])
main_color = st.selectbox("Main Color", ["red", "blue", "green", "black", "white"])
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

   # One-Hot Encode main_gem and brand_id (Modified)
    input_data = {
        "Category": factorized_mappings["Category"].index(category) if category in factorized_mappings["Category"] else -1,
        "Main_Metal": factorized_mappings["Main_Metal"].index(main_metal) if main_metal in factorized_mappings["Main_Metal"] else -1,
        "Target_Gender": factorized_mappings["Target_Gender"].index(target_gender) if target_gender in factorized_mappings["Target_Gender"] else -1,
        "Main_Color": factorized_mappings["Main_Color"].index(main_color) if main_color in factorized_mappings["Main_Color"] else -1,
        "Year": year,
        "Month": month,
    }

    # Dynamically create one-hot encoded features
    for gem in ["Main_Gem_" + gem_name for gem_name in ["agate", "amber", "amethyst", "chrysolite", "chrysoprase", "citrine",
                                                       "coral", "corundum_synthetic", "diamond", "emerald", "fianit",
                                                       "garnet", "iolite", "jade", "lapis_lazuli", "malachite",
                                                       "moonstone", "onyx", "opal", "pearl", "peridot", "quartz_pink",
                                                       "quartz_smoky", "ruby", "sapphire", "spinel", "tanzanite",
                                                       "topaz", "tourmaline", "turquoise", "zircon"]]:
        input_data[gem] = 1 if gem == "Main_Gem_" + main_gem else 0

    for brand in ["Brand_ID_" + str(i) for i in range(1, 8)]:
        input_data[brand] = 1 if brand == "Brand_ID_" + str(brand_id) else 0

    # Create DataFrame and ensure all columns from training are present
    input_df = pd.DataFrame([input_data])
    for col in one_hot_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with 0

    input_df = input_df[one_hot_columns]  # Reorder columns
    
    # Scale numerical features
    input_df[["Year", "Month"]] = scaler.transform(input_df[["Year", "Month"]])
    
    return input_df

# Prediction
if st.button("Predict Price"):
    input_data = preprocess_input(category, main_metal, target_gender, main_color, main_gem, brand_id, year, month)
    predicted_price = model.predict(input_data)[0]
    
    st.success(f"Predicted Price (USD): ${predicted_price:.2f}")
