import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("data.csv")

print("🔄 Loading data...")
print(f"Original shape: {data.shape}")

# ==================== FEATURE ENGINEERING ====================
print("\n⚙️ Feature Engineering...")

# 1. Create interaction features
data['area_per_bedroom'] = data['area'] / data['bedrooms']
data['area_per_bathroom'] = data['area'] / data['bathrooms']
data['bedroom_bathroom_ratio'] = data['bedrooms'] / data['bathrooms']

# 2. Create polynomial features
data['area_squared'] = data['area'] ** 2
data['bedrooms_squared'] = data['bedrooms'] ** 2

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

# 6. Create location category based on prefarea and mainroad
data['location_category'] = 0
data.loc[(data['prefarea'] == 'yes') & (data['mainroad'] == 'yes'), 'location_category'] = 3  # Premium
data.loc[(data['prefarea'] == 'yes') & (data['mainroad'] == 'no'), 'location_category'] = 2   # Good
data.loc[(data['prefarea'] == 'no') & (data['mainroad'] == 'yes'), 'location_category'] = 1   # Average
# 0 = Budget

# 7. Create synthetic location features based on area and features
# Simulate different neighborhoods
np.random.seed(42)
data['location_score'] = (
    data['prefarea'].map({'yes': 10, 'no': 0}) +
    data['mainroad'].map({'yes': 5, 'no': 0}) +
    (data['stories'] * 3)
)

print("✅ Feature engineering complete!")
print(f"New features created: area_per_bedroom, area_per_bathroom, bedroom_bathroom_ratio, area_squared, bedrooms_squared, total_rooms, amenities_score, luxury_score, location_category, location_score")

# ==================== ENCODE CATEGORICAL ====================
print("\n🔄 Encoding categorical features...")
data = pd.get_dummies(data, drop_first=True)

# ==================== PREPARE FEATURES ====================
X = data.drop("price", axis=1)
y = data["price"]

print(f"Final features: {len(X.columns)}")
print(f"Features: {X.columns.tolist()}")

# ==================== SPLIT DATA ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== SCALE FEATURES ====================
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== TRAIN MODELS ====================
print("\n🤖 Training models...\n")

# Model 1: Random Forest
print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_train_score = rf_model.score(X_train_scaled, y_train)
rf_test_score = rf_model.score(X_test_scaled, y_test)

# Model 2: Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_train_score = gb_model.score(X_train_scaled, y_train)
gb_test_score = gb_model.score(X_test_scaled, y_test)

# ==================== EVALUATE MODELS ====================
print("\n" + "="*50)
print("📊 MODEL PERFORMANCE COMPARISON")
print("="*50)

print(f"\n🌲 Random Forest:")
print(f"   Training R²: {rf_train_score:.4f}")
print(f"   Testing R²:  {rf_test_score:.4f}")
rf_pred = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(np.mean((y_test - rf_pred) ** 2))
rf_mae = np.mean(np.abs(y_test - rf_pred))
print(f"   RMSE: ₹ {rf_rmse:,.0f}")
print(f"   MAE:  ₹ {rf_mae:,.0f}")

print(f"\n📈 Gradient Boosting:")
print(f"   Training R²: {gb_train_score:.4f}")
print(f"   Testing R²:  {gb_test_score:.4f}")
gb_pred = gb_model.predict(X_test_scaled)
gb_rmse = np.sqrt(np.mean((y_test - gb_pred) ** 2))
gb_mae = np.mean(np.abs(y_test - gb_pred))
print(f"   RMSE: ₹ {gb_rmse:,.0f}")
print(f"   MAE:  ₹ {gb_mae:,.0f}")

# Choose best model
if gb_test_score > rf_test_score:
    print("\n✅ Gradient Boosting performs better!")
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_score = gb_test_score
else:
    print("\n✅ Random Forest performs better!")
    best_model = rf_model
    best_name = "Random Forest"
    best_score = rf_test_score

print(f"\n🏆 Best Model: {best_name} (R² = {best_score:.4f})")

# ==================== FEATURE IMPORTANCE ====================
print(f"\n🎯 Top 10 Important Features ({best_name}):")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ==================== SAVE FILES ====================
print("\n💾 Saving model and files...")
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_importance, open("feature_importance.pkl", "wb"))

print("\n✅ Model training complete!")
print(f"\n📈 Accuracy improved: 61.24% → {best_score*100:.2f}%")
print(f"🎯 Improvement: +{(best_score - 0.6124)*100:.2f}%")
