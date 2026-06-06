import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor


# ===============================
# LOAD DATASET
# ===============================

print("🏠 Loading Dataset...")

data = pd.read_csv("data.csv")

print("Dataset Shape:", data.shape)


# ===============================
# REMOVE UNNECESSARY COLUMNS
# ===============================

for col in ["id", "Date"]:
    if col in data.columns:
        data.drop(
            col,
            axis=1,
            inplace=True
        )


print("\nAvailable Columns:")
print(data.columns)


# ===============================
# HANDLE CATEGORICAL DATA
# ===============================

data = pd.get_dummies(
    data,
    drop_first=True
)


# ===============================
# FEATURES AND TARGET
# ===============================

X = data.drop(
    "Price",
    axis=1
)


y = data["Price"]


print(
    "\nTotal Features:",
    len(X.columns)
)


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# RANDOM FOREST MODEL
# ===============================

print(
    "\n🌲 Training Random Forest Model..."
)


model = RandomForestRegressor(

    n_estimators=100,

    max_depth=15,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


# ===============================
# MODEL TESTING
# ===============================

prediction = model.predict(
    X_test
)


accuracy = r2_score(
    y_test,
    prediction
)


error = mean_absolute_error(
    y_test,
    prediction
)


print("\n📊 Model Performance")

print(
    "Accuracy:",
    round(
        accuracy*100,
        2
    ),
    "%"
)


print(
    "Average Error ₹:",
    round(error)
)


# ===============================
# FEATURE IMPORTANCE
# ===============================

feature_importance = pd.DataFrame(

    {
        "Feature": X.columns,

        "Importance": model.feature_importances_
    }

).sort_values(

    by="Importance",

    ascending=False

)


print(
    "\nTop Important Features:"
)

print(
    feature_importance.head(10)
)



# ===============================
# SAVE FILES
# ===============================

pickle.dump(
    model,
    open(
        "model.pkl",
        "wb"
    )
)


pickle.dump(
    X.columns.tolist(),
    open(
        "features.pkl",
        "wb"
    )
)


pickle.dump(
    feature_importance,
    open(
        "feature_importance.pkl",
        "wb"
    )
)



print(
    "\n✅ Model trained successfully!"
)

print(
    "🌲 Algorithm Used: Random Forest Regression"
)


print(
    "🔥 Final Accuracy:",
    round(
        accuracy*100,
        2
    ),
    "%"
)