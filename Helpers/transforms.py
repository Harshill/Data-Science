import pandas as pd

def standardize(data: pd.Series, zscore=True):
    mean_diff = data - data.mean()

    return mean_diff / data.std() if zscore else mean_diff