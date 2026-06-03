import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

# Load dataset
data = pd.read_csv("data.csv")

print("🔄 Loading data...")
print(f"Original shape: {data.shape}")

# ==================== LOCATION CONVERSION ====================
print("\n📍 Converting prefarea to location categories...")

# Convert prefarea to location categories
data['location'] = 'Standard'
data.loc[data['prefarea'] == 'yes', 'location'] = 'Premium'

print("✅ Location categories created: Standard, Premium")

# Drop prefarea since we're using location now
data = data.drop(['prefarea'], axis=1)

# ==================== FEATURE ENGINEERING ====================
print("\n⚙️ Feature Engineering...")

# 1. Create interaction features
data['area_per_bedroom'] = data['area'] / data['bedrooms']
data['area_per_bathroom'] = data['area'] / data['bathrooms']
data['bedroom_bathroom_ratio'] = data['bedrooms'] / data['bathrooms']

# 2. Create polynomial features
data['area_squared'] = data['area'] ** 2

# 3. Create total rooms feature
data['total_rooms'] = data['bedrooms'] + data['bathrooms']

# 4. Create amenities score
data['amenities_score'] = (
    data['airconditioning'].map({'yes': 1, 'no': 0}) +
    data['guestroom'].map({'yes': 1, 'no': 0}) +
    data['basement'].map({'yes': 1, 'no': 0}) +
    data['hotwaterheating'].map({'yes': 1, 'no': 0})
)

# 5. Create luxury score
data['luxury_score'] = (
    data['parking'] +
    data['amenities_score'] +
    data['stories']
)

# 6. Create location score
data['location_score'] = (
    data['mainroad'].map({'yes': 5, 'no': 0}) +
    (10 if data['location'].eq('Premium').any() else 0)
)

print("✅ Feature engineering complete!")
print("   - Interaction features (area per room)")
print("   - Polynomial features (area_squared)")
print("   - Amenities & Luxury scores")
print("   - Location score (from location + mainroad)")

# ==================== ENCODE CATEGORICAL ====================
print("🔄 Encoding categorical features...")
data = pd.get_dummies(data, drop_first=True)

# ==================== PREPARE FEATURES ====================
X = data.drop("price", axis=1)
y = data["price"]

print(f"Total features: {len(X.columns)}")

# ==================== SPLIT DATA ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================== SCALE FEATURES ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== TRAIN MODEL ====================
print("\n🤖 Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ==================== EVALUATE MODEL ====================
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\n📊 Model Performance:")
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")

y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))

print(f"RMSE: ₹ {rmse:,.0f}")
print(f"MAE: ₹ {mae:,.0f}")

# ==================== FEATURE IMPORTANCE ====================
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top 15 Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ==================== SAVE FILES ====================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_importance, open("feature_importance.pkl", "wb"))

print(f"\n✅ Model trained and saved successfully!")
print(f"📈 Accuracy: {test_score*100:.2f}%")
improvement = (test_score - 0.6124) * 100
print(f"✨ Improvement from baseline: {improvement:+.2f}%")


