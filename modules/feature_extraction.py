# feature_extraction.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

def feature_engineering(filepath):
    DF = pd.read_csv(filepath)
    for lag in range(1, 8, 2):
        DF[f"Lag{lag}"] = DF["income"].shift(lag)
    DF["RollingMean"] = DF["income"].rolling(window=3).mean()
    DF.dropna(inplace=True)

    # Convert Date to numeric features
    DF["date"] = pd.to_datetime(DF["date"])
    DF["Month"] = DF["date"].dt.month
    DF["Day"] = DF["date"].dt.day
    DF['DayOfYear'] = DF['date'].dt.dayofyear
    DF.set_index("date", inplace=True)
    DF.index = pd.to_datetime(DF.index)

    #Used in order to get timeseries decomposition analysis
    DF['income'] = np.abs(DF['income']+0.1)
    # Ensure the 'date' column is in datetime format and set as the index
    DF.index = pd.to_datetime(DF.index)
    
    # Set the frequency (e.g., daily data -> 'D')
    DF = DF.asfreq('D')
    
    # Handle missing values - Fill missing data using forward fill
    DF['income'] = DF['income'].fillna(method='ffill')
    
    # Perform seasonal decomposition
    decomposition = sm.tsa.seasonal_decompose(DF['income'], model='additive')

    # Extract trend and seasonal components
    DF['trend'] = decomposition.trend
    DF['seasonal'] = decomposition.seasonal
    
    # Fill any missing values in the trend/seasonal components
    for column in DF.columns:
        DF[column] = DF[column].fillna(method='ffill').fillna(method='bfill')  # Use forward and backward fill for any NaNs
    #DF['seasonal'] = DF['seasonal'].fillna(method='ffill').fillna(method='bfill')
    
    DF.to_csv('/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/data/featured_data.csv')