from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from functions import *
import pandas as pd

if __name__ == '__main__':

    # Definitions

    # List of models to evaluate
    models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=False),
    'LightGBM': LGBMRegressor()
    }

    DATASET_PATH = '/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/data/income_per_day.csv'
    df_raw = pd.read_csv(DATASET_PATH)
    df = feature_engineering(df_raw)
    X_train, X_test, y_train, y_test  = split_processed_data(df)
    best_model = get_best_model(models, X_train, X_test, y_train, y_test)
    best_params = get_best_params(models, X_train, y_train, best_model)
    X, y = full_featured_data(df)
    final_model = train_final_model(X, y, best_params)
    dates_to_forecast = get_n_future_days(DF=df, n_days=7)
    data1, data2 = prepare_future_df(df, dates_to_forecast)
    forecasting = get_forecasting(DF=df, future_DF=dates_to_forecast, model=final_model)
    plot_forecasting(forecasting, df)
    model = load_model('/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/models/forecasting_income_model.pkl')
    predictions = model.predict(data2)
    data1['predictions'] = predictions
    plot_forecasting(data1, df)

