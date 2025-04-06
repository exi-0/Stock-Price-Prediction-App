import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page title and configure page
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-box {
        background-color: #48f04e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #24a9c7;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-price {
        font-size: 32px;
        font-weight: bold;
        color: #2e7d32;
        margin: 10px 0;
    }
    .model-name {
        color: #000000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.header("⚙️ User Input")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., GOOG):", "GOOG")
    start_date = st.text_input("Enter Start Date (YYYY-MM-DD):", "2012-01-01")
    end_date = st.text_input("Enter End Date (YYYY-MM-DD):", "2022-12-21")
    
    model_option = st.selectbox(
        "Select a Model:",
        ("Long Short-Term Memory (LSTM)", "Support Vector Regression (SVR)", "Random Forest", "Linear Regression")
    )
    
    year_to_predict = st.number_input("Enter Year to Predict:", min_value=2023, value=2023)

# Download stock data
@st.cache_data
def load_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date, timeout=30)
    data.reset_index(inplace=True)
    return data

data = load_data(stock_symbol, start_date, end_date)

# Display raw data
with st.expander("📊 View Raw Data"):
    st.dataframe(data.style.background_gradient(cmap='Blues'))

# Prepare the data
data['Date_ordinal'] = pd.to_datetime(data['Date']).apply(lambda date: date.toordinal())
X = data[['Date_ordinal']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimized LSTM functions
@st.cache_data
def prepare_lstm_data(data, seq_length=60):
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Efficient sequence creation using sliding window view
    X = np.lib.stride_tricks.sliding_window_view(scaled_prices[:-1, 0], seq_length)
    y = scaled_prices[seq_length:, 0]
    
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size].reshape(-1, seq_length, 1)
    X_test = X[train_size:].reshape(-1, seq_length, 1)
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

@st.cache_resource
def train_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], 1)),  # Reduced complexity
        Dense(16),  # Reduced complexity
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=0)  # Optimized training params
    return model

# Train the selected model
if model_option == "Long Short-Term Memory (LSTM)":
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler = prepare_lstm_data(data)
    model = train_lstm_model(X_train_lstm, y_train_lstm)
    
    y_pred = model.predict(X_test_lstm)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

elif model_option == "Support Vector Regression (SVR)":
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif model_option == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif model_option == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Model Evaluation Section
st.subheader(f"📊 Model Evaluation: {model_option}")
cols = st.columns(3)

with cols[0]:
    st.markdown(f"""
    <div class="metric-box">
        <h4>RMSE</h4>
        <h3>{rmse:.2f}</h3>
        <p style="color:#666; font-size:12px;">Root Mean Squared Error</p>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="metric-box">
        <h4>MAE</h4>
        <h3>{mae:.2f}</h3>
        <p style="color:#666; font-size:12px;">Mean Absolute Error</p>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="metric-box">
        <h4>R² Score</h4>
        <h3>{r2:.2f}</h3>
        <p style="color:#666; font-size:12px;">Variance Explained</p>
    </div>
    """, unsafe_allow_html=True)

# Plot actual vs predicted prices
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Date'], data['Close'], label='Actual Price', color='#3498db', linewidth=2)
if model_option == "Long Short-Term Memory (LSTM)":
    ax.plot(data['Date'][-len(y_test):], y_pred, label='Predicted Price', color='#e74c3c', linewidth=2)
else:
    ax.scatter(data.iloc[X_test.index]['Date'], y_pred, label='Predicted Price', color='#e74c3c', s=50)
ax.set_title(f'{stock_symbol} Stock Price: Actual vs Predicted', fontsize=16)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price ($)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# Predict Future Price
st.subheader("🔮 Future Price Prediction")
user_date = f"{year_to_predict}-01-01"
user_date_ordinal = pd.to_datetime(user_date).toordinal()

predicted_price = None

if model_option == "Linear Regression":
    predicted_price = model.predict([[user_date_ordinal]])[0]
elif model_option == "Random Forest":
    predicted_price = model.predict([[user_date_ordinal]])[0]
elif model_option == "Support Vector Regression (SVR)":
    user_date_scaled = scaler.transform([[user_date_ordinal]])
    predicted_price = model.predict(user_date_scaled)[0]
elif model_option == "Long Short-Term Memory (LSTM)":
    def predict_future_price(model, scaler, last_sequence, days_ahead):
        future_predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(days_ahead):
            next_prediction = model.predict(current_sequence.reshape(1, X_train_lstm.shape[1], 1))
            future_predictions.append(next_prediction[0, 0])
            current_sequence = np.append(current_sequence[1:], next_prediction)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        return future_predictions

    last_date = data['Date'].iloc[-1]
    days_ahead = (pd.to_datetime(user_date) - last_date).days
    if days_ahead > 0:
        last_sequence = scaler.transform(data['Close'].values[-X_train_lstm.shape[1]:].reshape(-1, 1))[:, 0]
        future_predictions = predict_future_price(model, scaler, last_sequence, days_ahead)
        predicted_price = future_predictions[-1][0]
    else:
        st.warning("The date is in the past. Please enter a future year.")

# Display predicted price
if predicted_price is not None:
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Predicted Price on {user_date}</h3>
        <div class="prediction-price">${float(predicted_price):.2f}</div>
        <p>Using <span class="model-name">{model_option}</span> model</p>
    </div>
    """, unsafe_allow_html=True)