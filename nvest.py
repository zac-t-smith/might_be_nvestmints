# Import the required Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import numpy as np


st.set_page_config(layout="wide")

# Define the global variables
df = None

# Functions for each of the pages
def crypto():
    st.subheader('BTC Risk')
    from datetime import date
    import numpy as np
    import pandas as pd
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.graph_objects as go
    import quandl
    import yfinance as yf

    # Download historical data from Quandl | BCHAIN/MKPRU is BTC
    df = quandl.get('BCHAIN/MKPRU', api_key='FeqsMXoDZZF_pAxV4kMi').reset_index()
    tlt_data = yf.download('TLT', period='max')['Close'].resample('W').last().dropna()
    oil_data = yf.download('CL=F', period='max')['Close'].resample('W').last().dropna()
    snp_data = yf.download('^GSPC', period='max')['Close'].resample('W').last().dropna()


    # Convert dates to datetime object for easy use
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort data by date, just in case
    df.sort_values(by='Date', inplace=True)

    # Only include data points with existing price
    df = df[df['Value'] > 0]

    # Get the last price against USD
    btcdata = yf.download(tickers='BTC-USD', period='1d', interval='1m')

    # Append the latest price data to the dataframe
    df.loc[df.index[-1]+1] = [date.today(), btcdata['Close'].iloc[-1]]
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate the `Risk Metric`
    df['MA'] = df['Value'].rolling(374, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Value) - np.log(df['MA'])) * df.index**.395
    tlt_ma = tlt_data.rolling(window=52).mean()
    oil_ma = oil_data.rolling(window=52).mean()
    snp_ma = snp_data.rolling(window=52).mean()
    snp_ma.name = 'S&P_MA'
    tlt_ma.name = 'TLT_MA'
    oil_ma.name = 'OIL_MA'
    df = pd.merge(df, tlt_ma, how='left', left_on='Date', right_index=True)
    df = pd.merge(df, oil_ma, how='left', left_on='Date', right_index=True)
    df = pd.merge(df, snp_ma, how='left', left_on='Date', right_index=True)
    df['MA'] = df['Value'].rolling(374, min_periods=1).mean().dropna()
    df['TLT_MA'] = df['TLT_MA'].rolling(52, min_periods=1).mean().dropna()
    df['OIL_MA'] = df['OIL_MA'].rolling(52, min_periods=1).mean().dropna()
    df['S&P_MA'] = df['S&P_MA'].rolling(52, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Value) - np.log(df['MA'])) * df.index**.395 - df['TLT_MA'] * -0.02 - df['OIL_MA'] * -0.02 

    # Normalization to 0-1 range
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

    # Predicting the price according to risk level
    price_per_risk = {
        round(risk, 1):round(np.exp(
            (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**.395 + np.log(df['MA'].iloc[-1])
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

    # Exclude the first 1000 days from the dataframe, because it's pure chaos
    df = df[df.index > 1000]

    # Title for the plots
    AnnotationText = f"Updated: {btcdata.index[-1]} | Price: {round(df['Value'].iloc[-1])} | Risk: {round(df['avg'].iloc[-1], 2)}"

    # Plot BTC-USD and Risk on a logarithmic chart
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Add BTC-USD and Risk data to the figure
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'],   name='Risk',  line=dict(color='white')), secondary_y=True)

    # Add green (`accumulation` or `buy`) rectangles to the figure
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

    # Add red (`distribution` or `sell`) rectangles to the figure
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    # Plot BTC-USD colored according to Risk values on a logarithmic chart
    fig = px.scatter(df, x='Date', y='Value', color='avg', color_continuous_scale='jet')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    # Plot Predicting BTC price according to specific risk
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Risk', 'Price'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
                line_color='darkslategray',
                fill_color='lightcyan',
                align='left'))
    ])
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    from sklearn import linear_model
    import pandas as pd
    import numpy as np
    import quandl as quandl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import mplcursors as mplcursors


    ### Import historical bitcoin price from quandl
    df = quandl.get("BCHAIN/MKPRU", api_key="FYzyusVT61Y4w65nFESX").reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)
    df = df[df["Value"] > 0]

    ### RANSAC Regression
    def LinearReg(ind, value):
        X = np.array(np.log(ind)).reshape(-1, 1)
        y = np.array(np.log(value))
        ransac = linear_model.RANSACRegressor(residual_threshold=2.989, random_state=0)
        ransac.fit(X, y)
        LinearRegRANSAC = ransac.predict(X)
        return LinearRegRANSAC

    df["LinearRegRANSAC"] = LinearReg(df.index, df.Value)

    #### Plot
    figs = make_subplots()
    figs.add_trace(go.Scatter(x=df["Date"], y=df["Value"], name="Price", line=dict(color="gold")))
    figs.add_trace(go.Scatter(x=df["Date"], y=np.exp(df["LinearRegRANSAC"]), name="Ransac", line=dict(color="green")))
    figs.update_layout(template="plotly_dark")
    mplcursors.cursor(hover=True)
    figs.update_xaxes(title="Date")
    figs.update_yaxes(title="Price", type='log', showgrid=True)
    figs.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(figs, use_container_width=True)

def gold():
    # Download historical data for TLT from Yahoo Finance
    df = yf.download(tickers='GC=F', period='max', interval='1d', auto_adjust=True)

    # Only include data points with existing price
    df = df[df['Close'] > 0]

    # Convert index to a column named 'Date' for consistency with the original code
    df.reset_index(inplace=True)

    # Sort data by date, just in case
    df.sort_values(by='Date', inplace=True)

    # Get the last price
    last_price = df['Close'].iloc[-1]

    # Calculate the `Risk Metric`
    df['MA'] = df['Close'].rolling(100, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Close) - np.log(df['MA'])) * df.index**.05

    # Normalization to 0-1 range
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

    # Predicting the price according to risk level
    price_per_risk = {
        round(risk, 1):round(np.exp(
            (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**.05 + np.log(df['MA'].iloc[-1])
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

    # Exclude the first 1000 days from the dataframe, because it's pure chaos
    df = df[df.index > 1000]

    # Title for the plots
    AnnotationText = f"Updated: {df['Date'].iloc[-1]} | Price: {round(last_price)} | Risk: {round(df['avg'].iloc[-1], 2)}"

    # Plot TLT and Risk on a logarithmic chart
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Add TLT and Risk data to the figure
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'],   name='Risk',  line=dict(color='white')), secondary_y=True)

    # Add green (`accumulation` or `buy`) rectangles to the figure
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

    # Add red (`distribution` or `sell`) rectangles to the figure
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    # Plot TLT colored according to Risk values on a logarithmic chart
    fig = px.scatter(df, x='Date', y='Close', color='avg', color_continuous_scale='jet')

    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)


    # Plot Predicting BTC price according to specific risk
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Risk', 'Price'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
                line_color='darkslategray',
                fill_color='lightcyan',
                align='left'))
    ])
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

def spy():
     # Download historical data for TLT from Yahoo Finance
    df = yf.download(tickers='SPY', period='max', interval='1d', auto_adjust=True)

    # Only include data points with existing price
    df = df[df['Close'] > 0]

    # Convert index to a column named 'Date' for consistency with the original code
    df.reset_index(inplace=True)

    # Sort data by date, just in case
    df.sort_values(by='Date', inplace=True)

    # Get the last price
    last_price = df['Close'].iloc[-1]

    # Calculate the `Risk Metric`
    df['MA'] = df['Close'].rolling(7, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Close) - np.log(df['MA'])) * df.index**4

    # Normalization to 0-1 range
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

    # Predicting the price according to risk level
    price_per_risk = {
        round(risk, 1):round(np.exp(
            (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**4 + np.log(df['MA'].iloc[-1])
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

    # Exclude the first 1000 days from the dataframe, because it's pure chaos
    df = df[df.index > 1000]

    # Title for the plots
    AnnotationText = f"Updated: {df['Date'].iloc[-1]} | Price: {round(last_price)} | Risk: {round(df['avg'].iloc[-1], 2)}"

    # Plot TLT and Risk on a logarithmic chart
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Add TLT and Risk data to the figure
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'],   name='Risk',  line=dict(color='white')), secondary_y=True)

    # Add green (`accumulation` or `buy`) rectangles to the figure
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

    # Add red (`distribution` or `sell`) rectangles to the figure
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    # Plot TLT colored according to Risk values on a logarithmic chart
    fig = px.scatter(df, x='Date', y='Close', color='avg', color_continuous_scale='jet')

    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)


    # Plot Predicting BTC price according to specific risk
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Risk', 'Price'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
                line_color='darkslategray',
                fill_color='lightcyan',
                align='left'))
    ])
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

def bonds():
    # Download historical data for TLT from Yahoo Finance
    df = yf.download(tickers='TLT', period='max', interval='1d', auto_adjust=True)

    # Only include data points with existing price
    df = df[df['Close'] > 0]

    # Convert index to a column named 'Date' for consistency with the original code
    df.reset_index(inplace=True)

    # Sort data by date, just in case
    df.sort_values(by='Date', inplace=True)

    # Get the last price
    last_price = df['Close'].iloc[-1]

    # Calculate the `Risk Metric`
    df['MA'] = df['Close'].rolling(100, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Close) - np.log(df['MA'])) * df.index**.05

    # Normalization to 0-1 range
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

    # Predicting the price according to risk level
    price_per_risk = {
        round(risk, 1):round(np.exp(
            (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**.05 + np.log(df['MA'].iloc[-1])
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

    # Exclude the first 1000 days from the dataframe, because it's pure chaos
    df = df[df.index > 1000]

    # Title for the plots
    AnnotationText = f"Updated: {df['Date'].iloc[-1]} | Price: {round(last_price)} | Risk: {round(df['avg'].iloc[-1], 2)}"

    # Plot TLT and Risk on a logarithmic chart
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # Add TLT and Risk data to the figure
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['avg'],   name='Risk',  line=dict(color='white')), secondary_y=True)

    # Add green (`accumulation` or `buy`) rectangles to the figure
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

    # Add red (`distribution` or `sell`) rectangles to the figure
    opacity = 0.2
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    # Plot TLT colored according to Risk values on a logarithmic chart
    fig = px.scatter(df, x='Date', y='Close', color='avg', color_continuous_scale='jet')

    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)


    # Plot Predicting BTC price according to specific risk
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Risk', 'Price'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
                line_color='darkslategray',
                fill_color='lightcyan',
                align='left'))
    ])
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
    with st.container():
        st.plotly_chart(fig, use_container_width=True)


# Add a title and intro text
st.title('SmartMate')
st.text('Why did the statistician invest in the stock market? Because he wanted to make a mean profit and avoid being median!.')

# Sidebar setup
st.sidebar.title('Deviate to the Standard ;)')

#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Review the market...', ['Crypto', 'Gold', "SPY", 'Bonds'])

# Navigation options
if options == 'Crypto':
    crypto()
elif options == 'Gold':
    gold()
elif options == 'SPY':
    spy() 
elif options == 'Bonds':
    bonds()
elif options == 'Interactive Plot':
    interactive_plot()
elif options == 'BTC Risk':
    btc_risk()
elif options == 'Risk Allocation':
    risk_allo()
