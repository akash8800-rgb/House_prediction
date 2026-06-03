# 🏠 House Price Prediction System

A machine learning application to predict house prices based on area, bedrooms, bathrooms, and location.

## 📦 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

### Option 1: Streamlit (Recommended - Simple & Interactive) ⭐
```bash
streamlit run streamlit_app.py
```
- Open browser automatically at `http://localhost:8501`
- Clean, modern UI with real-time predictions
- No need to manage separate backend

### Option 2: Flask (Advanced - Full Control)
```bash
python app.py
```
- Open browser to `http://localhost:5000`
- More customization available
- Requires managing HTML/CSS separately

## 📊 Training the Model

To retrain the model with new data:
```bash
python train_model.py
```

## 📁 Project Structure

```
house_price_model_project/
├── streamlit_app.py          # Streamlit web app (NEW - Recommended)
├── app.py                    # Flask web app (OLD)
├── train_model.py            # Model training script
├── model.pkl                 # Trained model
├── data.csv                  # Training dataset
├── requirements.txt          # Python dependencies
├── templates/                # HTML templates (for Flask)
├── static/                   # CSS & static files (for Flask)
└── House_Price_Prediction_Project.ipynb  # Jupyter notebook
```

## 🎯 Features

- 📝 Input area (sq ft), bedrooms, bathrooms, location
- 🤖 ML model predicts house price instantly
- 💰 Price displayed in Indian Lakhs (₹)
- ✨ Clean, modern UI

## 📝 Notes

- **Streamlit** is easier for data science projects (recommended for this use case)
- **Flask** gives more control if you need advanced customization
- Both use the same trained ML model (`model.pkl`)
