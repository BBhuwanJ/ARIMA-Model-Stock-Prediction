import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

# Title
st.title("📈 Stock Price Forecasting with ARIMA Model")

# File Selection
st.header("📂 Select or Upload CSV File")
sample_path = "SampleData.csv"
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
use_sample = st.checkbox("Use SampleData.csv instead", value=not uploaded_file)

# Load appropriate file
if use_sample:
    if not os.path.exists(sample_path):
        st.error("SampleData.csv not found in the working directory.")
        st.stop()
    df = pd.read_csv(sample_path)
    st.success("Loaded SampleData.csv ✅")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Loaded your uploaded file ✅")
else:
    st.warning("Please upload a file or check the 'Use SampleData.csv' option.")
    st.stop()

# Prepare data
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close']]

# Show raw data
st.subheader("📄 Raw Stock Data")
st.write(df)

# Plot historical prices
st.subheader("📊 Stock Price Over Time")
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label="Historical Prices", color="blue")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(plt)

# Split into train/test
data_train = data.iloc[:-30].copy()
data_test = data.iloc[-30:].copy()

# ADF test
st.subheader("🧪 Stationarity Check (ADF Test)")
adf_test = adfuller(data_train)
st.write(f"ADF Test p-value: {adf_test[1]}")
data_train_diff = data_train.diff().dropna()
adf_test_diff = adfuller(data_train_diff)
st.write(f"ADF after differencing p-value: {adf_test_diff[1]}")

# ARIMA modeling
st.subheader("🤖 ARIMA Model Fitting (2, 0, 4)")
model = ARIMA(data_train, order=(2, 0, 4))
model_fit = model.fit()
st.write(model_fit.summary())

# Forecast
st.subheader("🔮 30-Day Forecast")
future_forecast = model_fit.forecast(30)
future_dates = pd.date_range(start=data.index[-1], periods=31, freq='D')[1:]
forecast_df = pd.DataFrame({'Date': future_dates, 'Close': future_forecast})
forecast_df.set_index('Date', inplace=True)
st.write("### 📅 Predicted Prices")
st.write(forecast_df)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label="Historical Prices", color="blue")
plt.plot(forecast_df.index, forecast_df['Close'], label="Forecasted", color="red")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(plt)

# Evaluation
st.subheader("📏 Model Evaluation")
y_true = data_test['Close']
y_pred = model_fit.get_forecast(steps=len(data_test)).predicted_mean
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
accuracy = (1 - (rmse / np.mean(y_true))) * 100
st.write(f"MSE: {mse}")
st.write(f"RMSE: {rmse}")
st.write(f"Accuracy: {accuracy:.2f}%")

# Residuals
st.subheader("📉 Residuals")
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.tight_layout()
st.pyplot(plt)

# ACF & PACF
st.subheader("📈 ACF & PACF of Residuals")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals, ax=ax1)
plot_pacf(residuals, ax=ax2)
plt.tight_layout()
st.pyplot(plt)
