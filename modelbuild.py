import numpy as np
import pandas as pd
from pmdarima import pm
import joblib

def build_arima_model(data):
    # Convert data to pandas DataFrame if not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Assuming data has a single column representing time series
    # You may need to adjust this based on your actual data format
    # Replace 'value_column_name' with the actual name of your column
    ts_data = data['Sales']

    # Perform auto ARIMA to determine the best parameters

    model = pm.auto_arima(data.Sales, start_p=0, start_q=0, max_p=3, max_q=3, m=12,max_P=2,d=1,
    max_D=1,max_Q=2,start_P=0, seasonal=True, trace=True,
                      error_action='ignore', 
                      suppress_warnings=True, 
                      stepwise=True) 
    # Fit the ARIMA model
    model.fit(ts_data)

    return model

def save_model(model, filename):
    # Save the model using joblib
    joblib.dump(model, filename)

def main():
    # Load your time series data here
    # Example: data = pd.read_csv('your_data.csv')

    # Call function to build ARIMA model
    model = build_arima_model(data)

    # Example: Print summary of the model
    print(model.summary())

    # Example: Plot diagnostics of the model
    model.plot_diagnostics()
    plt.show()

    # Save the model
    save_model(model, 'arima_model.pkl')

if __name__ == "__main__":
    main()