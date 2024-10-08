import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

import joblib
import os


def feature_engineering(DF):
    for lag in range(1, 5, 2):
        DF[f"Lag{lag}"] = DF["income"].shift(lag)
    DF["RollingMean"] = DF["income"].rolling(window=3).mean()
    DF.dropna(inplace=True)

    # Convert Date to numeric features
    DF["date"] = pd.to_datetime(DF["date"])
    DF["Year"] = DF["date"].dt.year
    DF["Month"] = DF["date"].dt.month
    DF["Day"] = DF["date"].dt.day
    DF.set_index("date", inplace=True)

    return DF


def split_processed_data(DF):
    X_train, X_test, y_train, y_test = train_test_split(
        DF.drop(columns="income"),
        DF[["income"]],
        test_size=0.2,
        random_state=42,
        shuffle=False,
    )
    return X_train, X_test, y_train, y_test


# Function to create pipelines
def create_pipeline(model):
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    return pipeline


# Function to train and evaluate models
def set_train_and_evaluation(model, X_train, y_train, X_test, y_test):
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


def get_best_model(models, X_train, X_test, y_train, y_test):
    results = {}
    # Train and evaluate each model
    for name, model in models.items():
        results[name] = set_train_and_evaluation(
            model, X_train, y_train, X_test, y_test
        )

    best_model_df = (
        pd.DataFrame(results.items())
        .rename(columns={0: "model", 1: "mse"})
        .sort_values("mse", ascending=True)
        .head(1)
    )
    print(best_model_df)
    return best_model_df["model"].values[0]


def get_best_params(models, X_train, y_train, best_model):
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", models[best_model])])
    pipeline = pipeline.fit(X_train, y_train)

    # Define the parameter grid for grid search specific to LinearRegression
    param_grid = {
        "model__fit_intercept": [True, False],
        "model__copy_X": [
            True,
            False,
        ],  # Prefix with 'model__' since it's inside the pipeline
    }

    # Set up the grid search with TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error"
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best Parameters: ", best_params)
    print("Best Score: ", grid_search.best_score_)

    return best_params


def full_featured_data(DF):
    X = DF[["Lag1", "Lag3", "RollingMean", "Year", "Month", "Day"]]
    y = DF["income"]
    return X, y


def train_final_model(X, y, best_params):
    # Create and train the pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LinearRegression(
                    fit_intercept=best_params["model__fit_intercept"],
                    copy_X=best_params["model__copy_X"],
                ),
            ),
        ]
    )
    return pipeline.fit(X, y)


# Generate the next n days
def get_n_future_days(n_days, DF):
    last_date = DF.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    # Prepare the future DataFrame with lags and rolling mean
    future_df = pd.DataFrame({"date": future_dates})
    future_df["Year"] = future_df["date"].dt.year
    future_df["Month"] = future_df["date"].dt.month
    future_df["Day"] = future_df["date"].dt.day

    return future_df


def get_forecasting(DF, future_DF, model):
    # Initialize an empty list to store future predictions
    future_predictions = []
    days_in_future = len(future_DF.index)

    # Initialize future DataFrame with correct lags and rolling mean
    for i in range(days_in_future):
        if i == 0:
            future_DF.loc[i, "Lag1"] = DF.iloc[-1]["income"]
            future_DF.loc[i, "Lag3"] = DF.iloc[-3]["income"]
            future_DF.loc[i, "RollingMean"] = DF.iloc[-3:]["income"].mean()
        else:
            future_DF.loc[i, "Lag1"] = future_predictions[-1]
            future_DF.loc[i, "Lag3"] = (
                future_predictions[-3] if i > 2 else DF.iloc[-2]["income"]
            )
            future_DF.loc[i, "RollingMean"] = (
                np.mean(future_predictions[-3:])
                if i >= 3
                else DF.iloc[-3:]["income"].mean()
            )

        # Prepare features for the current prediction
        X_future = future_DF.loc[
            i, ["Lag1", "Lag3", "RollingMean", "Year", "Month", "Day"]
        ].values.reshape(1, -1)

        # Predict income and store it
        predicted_value = model.predict(X_future)[0]
        future_predictions.append(predicted_value)

    # Add predictions to future_DF
    future_DF["Predicted_Income"] = future_predictions

    return future_DF


def prepare_future_df(DF, future_DF):
    # Initialize an empty list to store future features
    future_features = []
    days_in_future = len(future_DF.index)
    
    # Initialize future DataFrame with correct lags and rolling mean
    for i in range(days_in_future):
        if i == 0:
            # For the first prediction, use the historical DF values
            future_DF.loc[i, 'Lag1'] = DF.iloc[-1]['income']
            future_DF.loc[i, 'Lag3'] = DF.iloc[-3]['income']
            future_DF.loc[i, 'RollingMean'] = DF.iloc[-3:]['income'].mean()
        else:
            # For subsequent predictions, check if 'Predicted_Income' exists
            if 'Predicted_Income' in future_DF.columns:
                future_DF.loc[i, 'Lag1'] = future_DF.loc[i - 1, 'Predicted_Income']
                future_DF.loc[i, 'Lag3'] = future_DF.loc[i - 3, 'Predicted_Income'] if i > 2 else DF.iloc[-2]['income']
                future_DF.loc[i, 'RollingMean'] = np.mean(future_DF.loc[i - 3:i - 1, 'Predicted_Income']) if i >= 3 else DF.iloc[-3:]['income'].mean()
            else:
                # Use historical DF for initial lag and rolling mean values
                future_DF.loc[i, 'Lag1'] = DF.iloc[-1]['income']
                future_DF.loc[i, 'Lag3'] = DF.iloc[-3]['income']
                future_DF.loc[i, 'RollingMean'] = DF.iloc[-3:]['income'].mean()

        # Prepare features for the current prediction
        X_future = future_DF.loc[i, ['Lag1', 'Lag3', 'RollingMean', 'Year', 'Month', 'Day']].values
        future_features.append(X_future)
    
    return future_DF, future_features


def plot_forecasting(forecasting, DF):
    # Print the predicted values
    sns.set()
    print(forecasting[["date", "Predicted_Income"]])

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(DF.index, DF["income"], label="Historical Income", color="blue")
    plt.plot(
        forecasting["date"],
        forecasting["Predicted_Income"],
        label="Predicted Income",
        color="red",
        linestyle="--",
    )
    plt.xlabel("Date")
    plt.ylabel("Income")
    plt.title(f"Income Prediction for the Next {len(forecasting.index)} Days")
    plt.legend()
    plt.savefig('/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/figures/plot.png')


def save_model(final_model):
    model_filename = "forecasting_income_model.pkl"
    models_dir = os.path.join(os.path.dirname(os.getcwd()), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)
    # Save the model
    joblib.dump(final_model, model_path)


def load_model(model_filename):
    """
    Load a model from the 'models' directory.

    Parameters:
    - model_filename (str): The filename of the saved model.

    Returns:
    - The loaded model object.
    """
    # Construct the path to the 'models' directory
    models_dir = os.path.join(os.path.dirname(os.getcwd()), "models")

    # Normalize the path
    models_dir = os.path.normpath(models_dir)

    # Define the full path to the model file
    model_path = os.path.join(models_dir, model_filename)

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"The model file '{model_filename}' was not found in '{models_dir}'."
        )

    # Load the model using joblib
    loaded_model = joblib.load(model_path)

    print(f"Model loaded from: {model_path}")
    return loaded_model
