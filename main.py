from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from functions import *
import pandas as pd
from conf import *

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

    income_df = load_data(RAW_DATA, INCOME_TIMESERIES_PATH)
    preprocessed_df = pd.read_csv(DATASET_PATH)
    df = feature_engineering(preprocessed_df)
    X_train, X_test, y_train, y_test  = split_processed_data(df)
    best_model = get_best_model(models, X_train, X_test, y_train, y_test, BEST_MODEL_PLOT)
    best_params = get_best_params(models, X_train, y_train, best_model)
    X, y = full_featured_data(df)
    final_model = train_final_model(X, y, best_params)
    save_model(final_model, MODELS_PATH)
    model = load_model(MODEL)
    dates_to_forecast = get_n_future_days(DF=df, n_days=15)
    forecasting = get_forecasting(DF=df, future_DF=dates_to_forecast, model=model)
    plot_forecasting(forecasting, df, FORECASTING_PLOT)

