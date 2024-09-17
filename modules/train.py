# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import os 

# List of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'CatBoost': CatBoostRegressor(verbose=False),
    'LightGBM': LGBMRegressor()
}

def split_processed_data(filepath):
    DF = pd.read_csv(filepath)
    
    # Drop the date column if it's not needed
    if 'date' in DF.columns:
        DF = DF.drop(columns=['date'])

    X_train, X_test, y_train, y_test = train_test_split(
        DF.drop(columns="income"),
        DF["income"],
        test_size=0.2,
        random_state=42,
        shuffle=False
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(model):
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    return pipeline

def train_model(X, y, model):
    pipeline = create_pipeline(model)
    pipeline.fit(X, y)
    return pipeline

def evaluate_model(trained_model, X_test, y_test):    
    predictions = trained_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    relative_error = np.abs(np.sum(y_test) - np.sum(predictions)) / np.sum(y_test)
    return mse, relative_error

def get_best_model(models, X_train, X_test, y_train, y_test):
    results = {}
    # Train and evaluate each model
    for name, model in models.items():
        trained_model = train_model(X_train, y_train, model)
        mse, relative_error = evaluate_model(trained_model, X_test, y_test)
        results[name] = (mse, relative_error)

    # Convert the result dictionary to a DataFrame for sorting
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['mse', 'relative_error'])

    # Get the best model based on the lowest relative error
    best_model_name = results_df.sort_values("relative_error", ascending=True).index[0]
    print(results_df)
    
    # Return the name and instance of the best model
    best_model_instance = models[best_model_name]
    return best_model_name, best_model_instance

def save_model(pipeline, model_name="final.pkl"):
    models_dir = '/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/models/'#os.path.join(os.path.dirname(os.getcwd()), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)
    dump(pipeline, model_path)
    
if __name__ == '__main__':
    filepath = '/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/data/featured_data.csv'
    X_train, X_test, y_train, y_test = split_processed_data(filepath)

    best_model_name, best_model = get_best_model(models, X_train, X_test, y_train, y_test)
    print(f"Best Model: {best_model_name}", best_model)
    
    model = train_model(X=pd.concat([X_train, X_test]), y=pd.concat([y_train, y_test]), model=models[best_model_name])
    
    save_model(model)