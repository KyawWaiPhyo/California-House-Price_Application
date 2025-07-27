import streamlit as st
import joblib
import numpy as np
import pickle
import os
from datetime import datetime
import pandas as pd

# Load both models
rf_model = joblib.load("tuned_rf_compressed.pkl")
gb_model = joblib.load("tuned_gb.pkl")

# Load feature names
features = joblib.load("features.pkl")
with open("feature_ranges.pkl", "rb") as f:
    feature_ranges = pickle.load(f)

st.title("🏠 California House Price Predictor (Hybrid Model)")
st.markdown("Enter housing features below:")

# Get user input
user_input = []
validation_failed = False
for feature in features:
  if feature in feature_ranges:
        min_val, max_val = feature_ranges[feature]
        value = st.number_input(f"{feature} ({min_val} to {max_val})", min_value=min_val, max_value=max_val)
        if not (min_val <= value <= max_val):
            st.warning(f"⚠️ {feature} must be between {min_val} and {max_val}.")
            validation_failed = True
  else:
    value = st.number_input(f"{feature}", step=0.1)
  user_input.append(value)

# Predict
if st.button("Predict"):
  if validation_failed:
        st.error("❌ Please fix input values before prediction.")
  else:
    input_array = np.array(user_input).reshape(1, -1)
    median_income_index = features.index("median_income")
    median_income = input_array[0][median_income_index]

    # Apply hybrid logic
    if median_income < 4.0:
        prediction = rf_model.predict(input_array)[0]
        model_used = "Tuned Random Forest"
    else:
        prediction = gb_model.predict(input_array)[0]
        model_used = "Tuned Gradient Boosting"

    st.success(f"📊 Predicted Price: **${prediction:,.2f}**")

        # Save input + prediction to log file
    data_row = user_input + [prediction, datetime.now()]
    columns = features + ['predicted_price', 'timestamp']

    log_df = pd.DataFrame([data_row], columns=columns)

    # Create the file if it doesn't exist
    log_file = "prediction_log.csv"
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, mode='w', header=True, index=False)

    st.info(f"Model Used: {model_used} and your inputs and prediction were logged for future model improvements.")
# Show download button if log exists
log_file = "prediction_log.csv"
if os.path.exists(log_file):
    with open(log_file, "rb") as f:
        st.download_button(
            label="⬇️ Download Prediction Log",
            data=f,
            file_name="prediction_log.csv",
            mime="text/csv"
        )
else:
    st.warning("📂 No prediction log found yet. Make a prediction first.")

import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
client = gspread.authorize(creds)
sheet = client.open("house_predictions").sheet1
st.title("📊 Log Prediction to Google Sheet")
# --- Append to sheet ---
if st.button("Log to Sheet"):
    row = user_input + [prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    sheet.append_row(row)
    st.success("✅ Prediction logged to Google Sheet!")
