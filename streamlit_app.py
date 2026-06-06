import streamlit as st
import pandas as pd
import pickle


# Load Model
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))


st.title("🏠 House Price Predictor")
st.write("Random Forest Regression | Accuracy: 87.43%")


# INPUTS

bedrooms = st.number_input("Bedrooms", 1, 10, 3)

bathrooms = st.number_input("Bathrooms", 1, 10, 2)

living = st.number_input(
    "Living Area (sq ft)",
    500,
    20000,
    2500
)

floors = st.number_input(
    "Floors",
    1,
    5,
    2
)

condition = st.selectbox(
    "House Condition",
    [
        "Average",
        "Good",
        "Excellent"
    ]
)

furnishing = st.selectbox(
    "Furnishing",
    [
        "Raw",
        "Semi Furnished",
        "Fully Furnished"
    ]
)

location = st.selectbox(
    "Location",
    [
        "Normal",
        "Premium",
        "Prime"
    ]
)


# VALUE MAPS

condition_map = {

    "Average":3,

    "Good":4,

    "Excellent":5
}


furnish_map = {

    "Raw":6,

    "Semi Furnished":9,

    "Fully Furnished":12
}


location_map = {

    "Normal":[47.3,-122.1],

    "Premium":[47.5,-122.2],

    "Prime":[47.7,-122.3]

}


# PREDICT

if st.button("Predict Price"):


    data = {

        f:0

        for f in features

    }


    data.update({

        "number of bedrooms":bedrooms,

        "number of bathrooms":bathrooms,

        "living area":living,

        "number of floors":floors,

        "condition of the house":
        condition_map[condition],

        "grade of the house":
        furnish_map[furnishing],

        "Built Year":2015,

        "Number of schools nearby":3,

        "Lattitude":
        location_map[location][0],

        "Longitude":
        location_map[location][1],

        "lot area":5000,

        "Area of the house(excluding basement)":living,

        "living_area_renov":living,

        "Postal Code":98000,

        "Distance from the airport":20

    })


    df = pd.DataFrame(
        [data]
    )[features]


    price = model.predict(
        df
    )[0]


    st.success(
        f"🏠 Estimated Price: ₹ {(price*83)/10000000:.2f} Crore"
    )