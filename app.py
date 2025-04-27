import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow import keras

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('ny_model.keras')
    return model

with st.spinner("Loading Model...."):
    model=load_model()



# model = load_model("my_model.keras")


st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')




@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



option = st.sidebar.text_input('Enter a Stock Symbol', value='GOOG')
option = option.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')




data = download_data(option, start_date, end_date)


def main():



    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict','Model Performance And accuracy'])
    if option == 'Visualize':
        tech_indicators(data)
    elif option == 'Recent Data':
        dataframe()

    elif option=='Model Performance And accuracy':
        find_accuracy()
    else:
        predict()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def find_accuracy():
    # --- Data Preparation ---
    dates = data.index  # Dates are already in the index — use directly!

    data_test = data['Close'].values.reshape(-1, 1)
    data_test_scale = scaler.fit_transform(data_test)

    x, y = [], []
    y_dates = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
        y_dates.append(dates[i])  # Pick date directly from index

    x, y = np.array(x), np.array(y)
    y_dates = np.array(y_dates)

    # --- Predictions ---
    y_inverse = scaler.inverse_transform(y.reshape(-1, 1))
    y_predict = model.predict(x)
    y_predict_inverse = scaler.inverse_transform(y_predict)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_dates, y_predict_inverse, 'r', label='Predicted Price')
    ax.plot(y_dates, y_inverse, 'g', label='Original Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Original vs Predicted Prices')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    fig.autofmt_xdate()

    st.pyplot(fig, use_container_width=True)

    # --- Metrics ---
    r2 = r2_score(y_inverse, y_predict_inverse)
    rmse = np.sqrt(mean_squared_error(y_inverse, y_predict_inverse))
    mse = mean_squared_error(y_inverse, y_predict_inverse)
    mae = mean_absolute_error(y_inverse, y_predict_inverse)

    st.subheader("📈 Model Performance")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")



def tech_indicators(data):
    st.header('📈 Technical Indicators')
    
    # Data validation and conversion
    if data.empty or 'Close' not in data.columns:
        st.error("Invalid data format - missing 'Close' prices")
        return
    
    try:
        # Ensure proper 1D array with correct index
        if isinstance(data['Close'], pd.DataFrame):
            close_prices = data['Close'].iloc[:, 0]  # Take first column if DF
        else:
            close_prices = data['Close'].copy()
        
        close_prices = pd.Series(
            data=close_prices.values.ravel(), 
            index=data.index,
            name='Close'
        )
    except Exception as e:
        st.error(f"Could not process Close prices: {str(e)}")
        return

    # UI for indicator selection
    option = st.radio(
        'Choose a Technical Indicator to Visualize',
        ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA']
    )
    
    # Window size parameter
    window = st.slider('Window Size', 5, 50, 14, key='indicator_window')
    
    # Plotting functions
    def plot_basic(series, title):
        st.write(f'📊 {title}')
        st.line_chart(series)
    
    def plot_comparison(main_series, secondary_series, title, secondary_name):
        df = pd.DataFrame({
            main_series.name: main_series,
            secondary_name: secondary_series
        })
        st.write(f'📊 {title}')
        st.line_chart(df)
    
    # Calculate and display selected indicator
    try:
        if option == 'Close':
            plot_basic(close_prices, 'Close Price')
            
        elif option == 'BB':
            bb_indicator = BollingerBands(close=close_prices, window=window, window_dev=2)
            bb_df = pd.DataFrame({
                'Close': close_prices,
                'Upper Band': bb_indicator.bollinger_hband(),
                'Lower Band': bb_indicator.bollinger_lband()
            })
            plot_basic(bb_df, 'Bollinger Bands')
            
        elif option == 'MACD':
            macd_indicator = MACD(close=close_prices)
            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            
            plot_comparison(macd_line, signal_line, 
                          'MACD (Moving Average Convergence Divergence)',
                          'Signal Line')
            
            st.area_chart(macd_indicator.macd_diff(), use_container_width=True)
            
        elif option == 'RSI':
            rsi = RSIIndicator(close=close_prices, window=window).rsi()
            rsi_df = pd.DataFrame({
                'RSI': rsi,
                'Overbought (70)': 70,
                'Oversold (30)': 30
            })
            plot_basic(rsi_df, 'RSI (Relative Strength Index)')
            
        elif option == 'SMA':
            sma = SMAIndicator(close=close_prices, window=window).sma_indicator()
            plot_comparison(close_prices, sma,
                          f'{window}-period SMA (Simple Moving Average)',
                          'SMA')
            
        elif option == 'EMA':
            ema = EMAIndicator(close=close_prices, window=window).ema_indicator()
            plot_comparison(close_prices, ema,
                          f'{window}-period EMA (Exponential Moving Average)',
                          'EMA')
            
    except Exception as e:
        st.error(f"Error generating {option} chart: {str(e)}")
        st.error("Please check your data and try again")
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(100))



def predict():
    
    
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        model_engine( num)
     


def model_engine( num):
    temp = data[-100:]['Close']
    temp_input = scaler.fit_transform(temp.values.reshape(-1, 1)).flatten().tolist()

    lst_output = []
    n_steps = 100
    i = 0

    while i < num:
        x_input = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        
        # Make sure to append scalar, not list/array
        next_value = yhat[0][0]
        temp_input.append(next_value)
        lst_output.append(next_value)
        i += 1


    

    # st.text(f'r2_score: {r2_score(y_test, preds)} \
    #         \nMAE: {mean_absolute_error(y_test, preds)}')

    day = 1
   # Convert to NumPy array and reshape for inverse_transform
    predicted_scaled = np.array(lst_output).reshape(-1, 1)

# Inverse transform to get real price values
    predicted_prices = scaler.inverse_transform(predicted_scaled)

    for i in predicted_prices:
        st.text(f'Day {day}: {i[0]}')
        day += 1


if __name__ == '__main__':
    main()

