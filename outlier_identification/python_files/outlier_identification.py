import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

# 1. Load your data
# df = pd.read_csv('sales_data.csv') 
# events = pd.read_csv('events.csv') # Contains Memorial Day, Black Friday, etc.
# promos = pd.read_csv('promos.csv') # Contains BOGO, Discounts

# For this example, let's assume 'df' is already loaded and merged with flags
# df['is_special'] = (df['is_event'] == 1) | (df['is_promo'] == 1)

def detect_time_series_outliers(df, target_col='sales', period=52):
    """
    Identifies outliers by decomposing seasonality and checking 
    residuals in two separate buckets (Normal vs Special Days).
    """
    
    # 2. Robust STL Decomposition (Removes Trend & 52-week Seasonality)
    # robust=True ensures spikes don't bleed into the trend line
    stl = STL(df[target_col], period=period, robust=True)
    result = stl.fit()
    df['resid'] = result.resid

    # 3. Define the Outlier Detection Logic (Modified Z-Score/MAD)
    def get_thresholds(series):
        # Using Median Absolute Deviation (MAD) for high robustness
        median = series.median()
        mad = (series - median).abs().median()
        # 1.4826 makes MAD comparable to Standard Deviation
        lower = median - 3 * (1.4826 * mad)
        upper = median + 3 * (1.4826 * mad)
        return lower, upper

    # 4. Apply thresholds separately to avoid bucket contamination
    df['is_outlier'] = False
    
    for is_special_day in [True, False]:
        mask = (df['is_special'] == is_special_day)
        if mask.any():
            subset_resid = df.loc[mask, 'resid']
            lower, upper = get_thresholds(subset_resid)
            
            # Flag both High-side and Low-side
            outlier_mask = (df['resid'] < lower) | (df['resid'] > upper)
            df.loc[mask & outlier_mask, 'is_outlier'] = True
            
    return df

# Example usage:
# df_cleaned = detect_time_series_outliers(df)
# print(df_cleaned[df_cleaned['is_outlier']])
