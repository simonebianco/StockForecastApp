import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
from dateutil.relativedelta import relativedelta  
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from statsmodels.tsa.stattools import adfuller

################################################ FUNCTIONS #######################################################################

def get_hurst(ts, min_lag=2, max_lag=20, step_size=1):
    lags = range(min_lag, max_lag, step_size)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    H = reg[0]
    st.write(f"H value: {H}")
    if H < 0.5:
        st.write("The time series has a value of Hurst exponent less than 0.5 denoting an antipersistent trend of the variations")
    elif H == 0.5:
        st.write("The time series has a Hurst exponent value of 0.5 denoting a random walk trend of the variations")
    else:
        st.write("The time series has a Hurst exponent value greater than 0.5, denoting a persistent trend of the variations")

def get_adfttest(ts):
    result = adfuller(ts)
    st.write('ADF Test Statistic: {}'.format(result[0]))
    st.write('P-value: {}'.format(result[1]))
    if result[1] <= 0.05:
        st.write("Strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        st.write("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary")

@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    info = yf.Ticker(ticker).info
    data.reset_index(inplace=True)
    return data, info

def plot_raw_data(df):
    layout = go.Layout(autosize=False,width=1000,height=500,
    margin=go.layout.Margin(l=0,r=50,b=150,t=50,pad = 5))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name="Adj Close"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
def plot_candle(df):
    layout = go.Layout(autosize=False,width=1000,height=500,
    margin=go.layout.Margin(l=0,r=50,b=150,t=50,pad = 5))
    fig = go.Figure(layout=layout, data=go.Ohlc(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def prophet_forecast(df):
    df_train = df[['Date','Adj Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

##################################################################################################################################

# Range Date
START = (date.today() - relativedelta(years = 6)).strftime("%Y-%m-%d") 
TODAY = date.today().strftime("%Y-%m-%d")

# Titolo
st.title('Stock Forecast App')
st.subheader('')

# Titoli Azionari
stocks_list = ('GOOG', 'AAPL', 'MSFT', 'TSLA')
selected_stock = st.sidebar.selectbox('Select Time Series ', stocks_list)
full_name = { 'AAPL':"Apple Inc.",
              'GOOG':"Alphabet Inc.",
              'MSFT':"Microsoft Corporation (MSFT)",
              'TSLA': "Tesla, Inc." }

# Caricamento TimeSeries da Yahoo Finance
df, info = load_data(selected_stock)

# Infarmozioni Titolo
industry = info["industry"]
sector = info["sector"]
country = info["country"]

# Slider Numero Anni Forecast
n_years_predictions = st.sidebar.slider('Prediction Years', 1, 5)
periods = n_years_predictions * 365

# Info Box
st.subheader('Info')
st.write(f"Time Series: {full_name[selected_stock]} ({selected_stock})")
st.write(f"Industry: {industry}")
st.write(f"Sector: {sector}")
st.write(f"Country: {country}")
st.write(f"Range Time Series: {START} - {TODAY}")

# Grafici Prezzi
plots = ("Line Plot", "Candlestick Plot")
selected_plot = st.selectbox('Select Plot', plots)
if selected_plot == "Line Plot":
    st.subheader('Time Series Line Plot')
    plot_raw_data(df)
else:
    st.subheader('Time Series Candlestick Plot')
    plot_candle(df)

st.subheader('Time Series Raw Data')
st.write(df.tail())
# Statistiche Descrittive
st.subheader('Time Series Summary Statistics')
st.write(df.describe())
# ADF Test
st.subheader('Dickey Fuller Test')
get_adfttest(df[['Adj Close']].values)
# Hurst Exponent
st.subheader('Hurst Exponent')
get_hurst(df['Adj Close'].values)

# Calcolo Forecast
forecast, model = prophet_forecast(df)

# Predizione Singola Data
dates = forecast[['ds']].sort_values(by=['ds'], ascending=False)
selected_date = st.sidebar.selectbox('Select Date', dates)
real_currency = 'None'

if selected_date in df['Date'].values:
    real = df[df['Date'] == selected_date]["Adj Close"].values[0]
    real_currency = "${:,.2f}".format(real)

prediction = forecast[forecast["ds"] == selected_date]["yhat"].values[0]
pred_currency = "${:,.2f}".format(prediction)
st.sidebar.write(f"Stock Price Prediction: {pred_currency}")
st.sidebar.write(f"Stock Price Actual: {real_currency}")

#Grafico Forecast
st.subheader(f'Forecast Plot For {n_years_predictions} Years')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Grafico Componenti Forecast
st.subheader("Forecast Components")
fig2 = plot_components_plotly(model, forecast)
st.write(fig2)


