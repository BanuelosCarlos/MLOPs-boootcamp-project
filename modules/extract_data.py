# extract_data.py
import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, encoding='unicode_escape')

    df['InvoiceDate'] = (pd.to_datetime(df['InvoiceDate']).
                            dt.strftime('%Y-%m-%d'))
    (df
        .groupby('InvoiceDate')
        .sum()
        .reset_index()
        .assign(income = lambda df: df.Quantity*df.UnitPrice)
        .rename(columns={'InvoiceDate':'date'})
        [['date', 'income']]
        .to_csv('/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/data/time_series_data.csv')
    )
    
load_data('/home/aleksei/Desktop/projects/MLOPs-boootcamp-project/data/OnlineRetail.csv')