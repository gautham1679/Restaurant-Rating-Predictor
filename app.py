import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


scaler = joblib.load("Scaler.pkl")
model = joblib.load("mlmodel.pkl")


st.set_page_config(layout="wide")
st.title("Restaurant Rating Prediction App")
st.caption("This app helps you to predict a restaurant review class")
st.markdown("---")


averagecost = st.number_input("Please enter the estimated average cost for two", min_value=50, max_value=99999, value=1000)
tablebooking = st.checkbox("Restaurant has table booking?")
onlinedelivery = st.selectbox("Restaurant has online delivery?", ["Yes", "No"])
pricerange = st.selectbox("What is the price range (1 Cheapest, 4 Most Expensive)?", [1, 2, 3, 4])

bookingstatus = 1 if tablebooking else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]

X = pd.DataFrame(values, columns=scaler.feature_names_in_)

if st.button("Predict the review!"):
    X_scaled = scaler.transform(X)  
    prediction = model.predict(X_scaled)  
    st.snow()
    st.write(f"### ⭐ Prediction: {prediction[0]}")

st.markdown("---")

