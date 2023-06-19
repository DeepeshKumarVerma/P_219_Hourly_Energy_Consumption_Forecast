# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pickle
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore") 


st.markdown('''
<style>
.stApp {
    
    background-color:#8DC8ED;
    align:center;\
    display:fill;\
    border-radius: false;\
    border-style: solid;\
    border-color:#000000;\
    border-style: false;\
    border-width: 2px;\
    color:Black;\
    font-size:15px;\
    font-family: Source Sans Pro;\
    background-color:#ffb3b3;\
    text-align:center;\
    letter-spacing:0.1px;\
    padding: 0.1em;">\
}
.sidebar {
    background-color: black;
}
.st-b7 {
    color: #8DC8ED;
}
.css-nlntq9 {
    font-family: Source Sans Pro;
}
</style>
''', unsafe_allow_html=True)
model1=pickle.load(open("final_rf_model.pkl","rb"))
daily_data_last_7=pd.read_csv("daily_data_last_7.csv", header=None)
data=pd.read_csv("dataset_daily.csv",header=0, index_col=0, parse_dates=True)

st.title("Forecast power consumption data")
st.sidebar.subheader("Select the number of days to Forecast from 2018-Aug-4")
days = st.sidebar.number_input('Days',min_value = 1,step = 1)

z=daily_data_last_7
z=np.array(z[0].tail(7))
for i in range(0,days):
    r=z[-7:]
    r=np.array([r])
    ranf_f=model1.predict(r)
    z=np.append(z,ranf_f)
    i=+1
future_pred=z[-days:]

    
    
future = pd.date_range(start='8/4/2018',periods=days,tz=None,freq = 'D')
future_df = pd.DataFrame(index=future)
future_df['Power Consumption'] = future_pred.tolist()

st.sidebar.write(f"Power consumption for {days}th day")
st.sidebar.write(future_df[-1:])
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Power Consumption Forecasted for {days} days" )
    st.write(future_df)
with col2:
    st.subheader('Forecasted Graph')
    fig, ax = plt.subplots()
    plt.figure(figsize=(8,3))
    ax.plot(future_df.index,future_df.values, label='Forecast', color="red")
    ax.tick_params(axis='x', labelrotation = 100)
    plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
    plt.ylabel('Power consumption', fontsize=15)
    plt.xlabel('Date', fontsize=15)
    st.pyplot(fig)
    
    st.subheader('Actual Vs Forecast plot')
    fig, ax = plt.subplots()
    plt.figure(figsize=(8,3))
    ax.plot(future_df.index,future_df.values, label='Forecast', color="purple")
    ax.plot(data['Energy'][-365:].index,data['Energy'][-365:].values)
    ax.tick_params(axis='x', labelrotation = 100)
    plt.legend(fontsize=5, fancybox=True, shadow=True, frameon=True)
    plt.ylabel('Power consumption', fontsize=8)
    plt.xlabel('Date', fontsize=8)
    st.pyplot(fig)
    def generate_forecast(data, days):
        daily_data_last_7 = [data[-7:]]
        last_7_days = daily_data_last_7[0].values
        forecast = np.zeros((days,))
        for i in range(days):
            if i < 7:
                forecast[i] = last_7_days[i]
            else:
                last_7_days = forecast[i-7:i]
                forecast[i] = model1.predict(last_7_days.reshape(1,-1))[0]
        return forecast

    # Set Streamlit app title
    st.title("Forecast power consumption data of uploaded dataset")

    # Add a sidebar with options
    st.sidebar.subheader("Select the number of days to Forecast from 2018-Aug-4")

    # File uploader widget
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "xlsx"])

    # Read user uploaded dataset if provided
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, header=0, index_col=0, parse_dates=True)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, header=0, index_col=0, parse_dates=True)
        else:
            st.error("File format not supported. Please upload a CSV or Excel file.")

        # Add a number input widget for selecting the number of days to forecast
        #days = st.sidebar.number_input('Days', min_value=1, step=1)

        if days:
            # Generate forecasted data
            future_pred = generate_forecast(data, days)

            # Create a date range for the forecasted data
            future = pd.date_range(start='8/4/2018', periods=days, tz=None, freq='D')
            future_df = pd.DataFrame(index=future)
            future_df['Power Consumption'] = future_pred.tolist()

            # Display the forecasted data in a table and a graph
            st.sidebar.write(f"Power consumption for {days}th day")
            st.sidebar.write(future_df[-1:])

            #col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Power consumption Forecasted for {days} days")
                st.write(future_df)
            with col2:
                st.subheader('Forecasted Graph')
                fig, ax = plt.subplots()
                plt.figure(figsize=(8,3))
                ax.plot(future_df.index,future_df.values, label='Forecast', color="orange")
                ax.tick_params(axis='x', labelrotation=100)
                plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
                plt.ylabel('Power consumption', fontsize=15)
                plt.xlabel('Date', fontsize=15)
                st.pyplot(fig)
        
                if 'data' in locals():
                    st.subheader('Actual Vs Forecast plot')
                    fig, ax = plt.subplots()
                    plt.figure(figsize=(8,3))
                    ax.plot(future_df.index,future_df.values, label='Forecast', color="orange")
                    ax.plot(data.index, data.values, label='Actual', color="blue")
                    plt.legend(fontsize=12, fancybox=True, shadow=True, frameon=True)
                    plt.ylabel('Power consumption', fontsize=15)
                    plt.xlabel('Date', fontsize=15)
                    st.pyplot(fig)
                else:
                    st.warning("Please upload a data file to compare with forecast.")

