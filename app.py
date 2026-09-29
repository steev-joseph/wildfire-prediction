import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Wildfire Burned Area Prediction",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🔥 Wildfire Burned Area Prediction")
st.write("### Linear Regression Machine Learning Project")

st.write(
    "This application uses Linear Regression to predict "
    "the burned area of a wildfire based on weather and fire-related features."
)


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("forestfires.csv")


# ---------------------------------------------------
# SHOW DATASET
# ---------------------------------------------------

st.subheader("📊 Dataset")

st.write("Dataset Shape:", df.shape)

st.dataframe(df.head(10))


# ---------------------------------------------------
# INPUT FEATURES
# ---------------------------------------------------

features = [
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "temp",
    "RH",
    "wind",
    "rain"
]

X = df[features]
y = df["area"]


# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# LINEAR REGRESSION MODEL
# ---------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# ---------------------------------------------------
# MODEL RESULTS
# ---------------------------------------------------

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")

col2.metric("MSE", f"{mse:.2f}")

col3.metric("RMSE", f"{rmse:.2f}")

col4.metric("R² Score", f"{r2:.2f}")


# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.subheader("🔍 Actual vs Predicted Values")

results = pd.DataFrame({
    "Actual Burned Area": y_test.values,
    "Predicted Burned Area": y_pred
})

st.dataframe(results.head(10))


# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

st.subheader("📊 Actual vs Predicted Graph")

fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.set_xlabel("Actual Burned Area")

ax.set_ylabel("Predicted Burned Area")

ax.set_title("Actual vs Predicted Wildfire Burned Area")

st.pyplot(fig)


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

st.subheader("🔥 Predict Burned Area for a New Wildfire")

col1, col2 = st.columns(2)

with col1:

    ffmc = st.number_input(
        "FFMC",
        min_value=0.0,
        value=90.0
    )

    dmc = st.number_input(
        "DMC",
        min_value=0.0,
        value=100.0
    )

    dc = st.number_input(
        "DC",
        min_value=0.0,
        value=500.0
    )

    isi = st.number_input(
        "ISI",
        min_value=0.0,
        value=10.0
    )


with col2:

    temp = st.number_input(
        "Temperature",
        min_value=0.0,
        value=30.0
    )

    rh = st.number_input(
        "Relative Humidity (RH)",
        min_value=0.0,
        value=30.0
    )

    wind = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=5.0
    )

    rain = st.number_input(
        "Rain",
        min_value=0.0,
        value=0.0
    )


# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("🔥 Predict Burned Area"):

    new_fire = np.array([[
        ffmc,
        dmc,
        dc,
        isi,
        temp,
        rh,
        wind,
        rain
    ]])

    prediction = model.predict(new_fire)

    predicted_area = prediction[0]

    if predicted_area < 0:
        predicted_area = 0

    st.success(
        f"🔥 Predicted Burned Area: {predicted_area:.2f} hectares"
    )