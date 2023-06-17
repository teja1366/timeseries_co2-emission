import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('CO2 Emission')
st.header('User Input Parameter')
CO2 = st.slider('Select Year', 2015, 2065)
number = CO2 - 2014

# Read the CO2 dataset
data = pd.read_csv('D:\intern\loan\CO2.csv', header=0, index_col=0, parse_dates=True)
data = data.dropna()

data1 = data.copy()
data1['CO2_diff'] = data['CO2'].diff()
data1 = data1.dropna(subset=['CO2_diff'])

# Split the data into training and test sets
train_data = data['CO2']
test_data = data['CO2']
y_data = data1['CO2_diff']

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(y_data, trend="add", seasonal="add", seasonal_periods=12, damped_trend=True)
ex_model_fit = model.fit()

# Forecast future values
forecasted_values = ex_model_fit.predict(start=len(y_data), end=len(y_data) + number - 1)

# Add the differenced values back to obtain the actual predicted values
predicted_values = (forecasted_values.cumsum() + data['CO2'].iloc[-1]).values

# Retrieve the predicted value for the selected year
selected_year_prediction = predicted_values[number - 1]

# Historical and forecasted CO2 values
historical_values = pd.concat([data['CO2'], pd.Series(predicted_values)], axis=0)
st.write("The CO2 Emission for the selected year:", selected_year_prediction)
# Line chart for historical and forecasted CO2 values
if st.checkbox('Line Plot'):
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=historical_values.index, y=historical_values.values, mode='lines', name='CO2 Emission'))
    fig_line.update_layout(title='Historical and Forecasted CO2 Emission (Line Plot)', xaxis_title='Year', yaxis_title='CO2 Emission')
    st.plotly_chart(fig_line, use_container_width=True, sharing='streamlit')  # Display Plotly line chart in Streamlit

# Bar chart for historical and forecasted CO2 values
if st.checkbox('Bar Plot'):
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=historical_values.index, y=historical_values.values, name='CO2 Emission'))
    fig_bar.update_layout(title='Historical and Forecasted CO2 Emission (Bar Plot)', xaxis_title='Year', yaxis_title='CO2 Emission',yaxis_range=[0, 20]) 
    #fig_bar.update_layout(title='Historical and Forecasted CO2 Emission (Bar Plot)', xaxis_title='Year', yaxis_title='CO2 Emission')
    st.plotly_chart(fig_bar, use_container_width=True, sharing='streamlit')  # Display Plotly bar chart in Streamlit

# Scatter plot for historical and forecasted CO2 values
if st.checkbox('Scatter Plot'):
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=historical_values.index, y=historical_values.values, mode='markers', name='CO2 Emission'))
    fig_scatter.update_layout(title='Historical and Forecasted CO2 Emission (Scatter Plot)', xaxis_title='Year', yaxis_title='CO2 Emission')
    st.plotly_chart(fig_scatter, use_container_width=True, sharing='streamlit')  # Display Plotly scatter plot in Stream
