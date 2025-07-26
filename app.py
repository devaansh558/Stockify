import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Streamlit UI elements
st.title("📈 Stock Price Predictor")

# Get user input for stock ticker
stock = st.text_input("Enter stock ticker", "AAPL")
start = "2015-01-01"
end = "2025-01-01"

# Download stock data using yfinance
data = yf.download(stock, start=start, end=end)
st.subheader(f"Stock Data for {stock}")
st.write(data.tail())

# Add technical indicators (RSI, MACD, etc.)
close_series = data['Close'].squeeze()
rsi_indicator = RSIIndicator(close=close_series)
data['RSI'] = rsi_indicator.rsi()

macd = MACD(close=close_series)
data['MACD'] = macd.macd_diff()

# Display stock data with technical indicators
st.subheader(f"Stock Data with Technical Indicators for {stock}")
st.write(data.tail())

# Load the trained model
model = load_model("stock_model.keras")

# Preprocessing & feature scaling
scaler = MinMaxScaler(feature_range=(0,1))

# Prepare the data for prediction (using last 100 days)
pas_100_days = data.tail(100)
data_scaled = scaler.fit_transform(pas_100_days[['Close', 'RSI', 'MACD', 'Volume']])

# Create sequences for prediction
X_test = []
for i in range(100, data_scaled.shape[0]):
    X_test.append(data_scaled[i-100:i])
X_test = np.array(X_test)

# Predict the stock price using the trained LSTM model
predicted_price = model.predict(X_test)

# Rescale the predicted values
scale = 1 / scaler.scale_[0]  # Correct scaling factor
predicted_price = predicted_price * scale

# Displaying the results
st.subheader('Stock Price Prediction vs Actual Price')
fig = plt.figure(figsize=(8,6))
plt.plot(predicted_price, 'r', label='Predicted Price')
plt.plot(data['Close'].tail(len(predicted_price)), 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
