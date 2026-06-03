import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.markdown("<h1 style='text-align: center;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)

# Load model
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found. Please run train_model.py first.")
    st.stop()

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=0,
        step=100,
        value=1000
    )
    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

with col2:
    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=5,
        value=2
    )
    location = st.selectbox(
        "Location Type",
        ["Downtown", "Suburban", "Urban", "Rural"]
    )

# Prediction button
if st.button("🚀 Predict Price", use_container_width=True):
    try:
        # Create dictionary with all features = 0
        input_dict = {col: 0 for col in model.feature_names_in_}
        
        # Fill numeric values
        input_dict['area'] = area
        input_dict['bedrooms'] = bedrooms
        input_dict['bathrooms'] = bathrooms
        
        # Set selected location = 1
        loc_col = f"location_{location.lower()}"
        
        if loc_col in input_dict:
            input_dict[loc_col] = 1
        
        # Convert into DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Predict price
        prediction = model.predict(input_df)
        
        # Get predicted value
        result = round(prediction[0], 2)
        
        # Convert into lakhs
        result_lakh = round(result / 100000, 2)
        
        # Display result in a nice box
        st.success(f"✅ Estimated Price: ₹ {result_lakh} Lakhs")
        
        # Show additional details
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Area", f"{area} sq ft")
        with col2:
            st.metric("Bedrooms", bedrooms)
        with col3:
            st.metric("Bathrooms", bathrooms)
        with col4:
            st.metric("Location", location)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px;'>"
    "House Price Prediction Model</p>",
    unsafe_allow_html=True
)
