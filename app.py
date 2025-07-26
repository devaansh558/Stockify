import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

st.title("📈 Stock Price Predictor")

stock = st.text_input("Enter stock ticker", "AAPL")
start = "2015-01-01"
end = "2025-01-01"

data = yf.download(stock, start=start, end=end)
st.write(data.tail())

# RSI, MACD etc. can be added here

# Load model
model = load_model("stock_model.keras")  # Upload this to Streamlit Cloud

# Preprocessing & Prediction code...

st.line_chart(data['Close'])
