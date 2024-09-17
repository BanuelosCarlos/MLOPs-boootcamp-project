import os

# Get the project root dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# Define paths relative to the project root
RAW_DATA = os.path.join(project_root, 'data', 'OnlineRetail.csv')
INCOME_TIMESERIES_PATH = os.path.join(project_root, 'data', 'time_series_data.csv')
DATASET_PATH = os.path.join(project_root, 'data', 'time_series_data.csv')
BEST_MODEL_PLOT = os.path.join(project_root, 'figures', 'plot_train.png')
FORECASTING_PLOT = os.path.join(project_root, 'figures', 'plot.png')
MODELS_PATH = os.path.join(project_root, 'models')
MODEL = os.path.join(MODELS_PATH, 'forecasting_income_model.pkl')