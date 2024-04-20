import streamlit as st
import pandas as pd
from prophet import Prophet

# Function to make predictions and sum up the yhat values
def make_predictions(df, start_date, end_date):
    
    unique_products = df['Category'].unique()
    results = {}
    for category in unique_products:
        total_sum = 0
        selected_df = df[df['Category'] == category]
        selected_df['ds'] = pd.DatetimeIndex(selected_df['Order Date'])
        selected_df['y'] = selected_df['Sales']
        selected_df.drop(['Order Date', 'Sales'], axis=1, inplace=True)
        
        m = Prophet(interval_width=0.95, daily_seasonality=True)
        model = m.fit(selected_df)
        
        future = m.make_future_dataframe(periods=(end_date - start_date).days + 1, freq='D')
        forecast = m.predict(future)
        
        mask = (forecast['ds'] >= pd.Timestamp(start_date)) & (forecast['ds'] <= pd.Timestamp(end_date))
        forecast_filtered = forecast.loc[mask]
        
        total_sum += forecast_filtered['yhat'].sum()
        st.write(category +" Demand !")
        st.write(total_sum)

        # Store the result in the dictionary
        results[category] = total_sum
        
        fig=m.plot_components(forecast)
        fig1 = m.plot(forecast)


        st.pyplot(fig)
        st.pyplot(fig1)

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    top_3 = list(sorted_results.keys())[:3]
    st.write("Top 3 Categories:")
    for i, category in enumerate(top_3):
        st.write(f"{i+1}. {category}: {sorted_results[category]}")
        
        
    
st.title('Demand Prediction App')

# File uploader for the sales data
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Date inputs
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')

    # Button to trigger predictions
    if st.button('Predict'):
        make_predictions(df, start_date, end_date)
        #st.write("Total Sales Prediction:", total_sum)
