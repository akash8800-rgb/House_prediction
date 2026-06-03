import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏠 House Price Predictor</h1>", unsafe_allow_html=True)

# Load model, scaler, and feature importance
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_importance = pickle.load(open("feature_importance.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"❌ Model file not found. Please run train_model.py first.\nError: {str(e)}")
    st.stop()

# Create input form
st.markdown("---")
st.markdown("### 📝 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Basic Info**")
    area = st.number_input(
        "Area (sq ft)",
        min_value=500,
        step=100,
        value=8000
    )
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2
    )
    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=5,
        value=2
    )

with col2:
    st.markdown("**Location & Access**")
    location = st.radio(
        "Location Type",
        ["Premium", "Standard"],
        horizontal=True
    )
    mainroad = st.checkbox("On Main Road?", value=True)
    
    st.markdown("**Amenities**")
    parking = st.slider(
        "Parking Spaces",
        min_value=0,
        max_value=3,
        value=2
    )
    airconditioning = st.checkbox("Air Conditioning?", value=True)

with col3:
    st.markdown("**Additional Features**")
    guestroom = st.checkbox("Guest Room?", value=False)
    basement = st.checkbox("Basement?", value=False)
    hotwaterheating = st.checkbox("Hot Water Heating?", value=False)
    
    st.markdown("**Furnishing**")
    furnishing = st.radio(
        "Status",
        ["Furnished", "Semi-Furnished", "Unfurnished"],
        horizontal=True
    )

st.markdown("---")

# Prediction button
col_btn = st.columns([1, 4, 1])[0]
predict_button = st.button("🚀 Predict Price", use_container_width=True, key="predict")

if predict_button:
    try:
        # Create input dataframe in exact order matching training
        input_df = pd.DataFrame({
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'parking': [parking],
            'mainroad': ['yes' if mainroad else 'no'],
            'guestroom': ['yes' if guestroom else 'no'],
            'basement': ['yes' if basement else 'no'],
            'hotwaterheating': ['yes' if hotwaterheating else 'no'],
            'airconditioning': ['yes' if airconditioning else 'no'],
            'location': [location],
            'furnishingstatus': [furnishing.lower()],
        })
        
        # Add engineered features in order matching training
        input_df['area_per_bedroom'] = input_df['area'] / input_df['bedrooms']
        input_df['area_per_bathroom'] = input_df['area'] / input_df['bathrooms']
        input_df['bedroom_bathroom_ratio'] = input_df['bedrooms'] / input_df['bathrooms']
        input_df['area_squared'] = input_df['area'] ** 2
        input_df['total_rooms'] = input_df['bedrooms'] + input_df['bathrooms']
        
        # Amenities score
        input_df['amenities_score'] = (
            (input_df['airconditioning'] == 'yes').astype(int) +
            (input_df['guestroom'] == 'yes').astype(int) +
            (input_df['basement'] == 'yes').astype(int) +
            (input_df['hotwaterheating'] == 'yes').astype(int)
        )
        
        # Luxury score
        input_df['luxury_score'] = (
            input_df['parking'] +
            input_df['amenities_score'] +
            input_df['stories']
        )
        
        # Location score
        input_df['location_score'] = (
            (input_df['mainroad'] == 'yes').astype(int) * 5
        )
        
        # Apply get_dummies with drop_first=True (matching training)
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # EXACT feature order from training
        feature_order = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'area_per_bedroom', 'area_per_bathroom', 'bedroom_bathroom_ratio',
            'area_squared', 'total_rooms', 'amenities_score', 'luxury_score',
            'location_score', 'mainroad_yes', 'guestroom_yes', 'basement_yes',
            'hotwaterheating_yes', 'airconditioning_yes',
            'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished',
            'location_Standard'
        ]
        
        # Ensure all features exist (add zeros if missing)
        for feat in feature_order:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        # Reorder to exact training order
        input_df = input_df[feature_order]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        result = round(prediction[0], 2)
        result_lakh = round(result / 100000, 2)
        
        # Display result with styling
        st.markdown("---")
        st.success(f"✅ **Estimated Price: ₹ {result_lakh} Lakhs** (₹ {result:,.0f})")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    <p>🤖 Powered by Random Forest Machine Learning Model</p>
    <p>📊 Model Accuracy: 60.2% | 🎯 Features: 21 | 📚 Training Data: 545 properties</p>
    <p style='font-size: 11px;'>Predictions are estimates based on historical data patterns.</p>
    </div>
""", unsafe_allow_html=True)
