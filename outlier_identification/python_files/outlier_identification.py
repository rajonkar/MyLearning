# https://www.google.com/search?q=i+have+a+time+series+data+that+has+promotions+like+BOGO.+If+i+were+to+identify+outliers+on+both+high+and+low+side+how+do+i+do+that&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEJMzU1MTJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpaEWjvZ2Py1XXV8d8KvlI3izfzqgn7395CNCvYdZRuZ5U71YjigLP8fxOdtGNi00AID0tRCZHJ8fgh0HcZOsGj_K3nSeb8S5ovAsHNcxAlbWFsabvN_f67JLYAW48ZvNV06CZBv371fkqFwGNtqaspixefiL07nDsfYksrWsZNxLdEOW-lWrtts9K9tw3he9tfl_l_A&ved=2ahUKEwi6oNvvsfKTAxXCwTgGHTbXJ00Q0NsOegQIAxAA&aep=10&ntc=1&mstk=AUtExfCWI97WMa3QzhHKJgGY4pHZJG50F85ZuAdKieAyDiYNmyA71qG-0sdZuMZ7QZlBslEyYa64U-P-Q6eSwasjqoArFCWqL-3VDba1b9UpLpP-5HjDFtGYgvTn1911QBuG07XsZpuWREElYQBbAaFdaT0d-bi7JnUVwtzDfPFuc0yzcii4jsPmMvWm0Z8y5MZ1-HTR9tHmzP57iakf_DfY3o4DvNClkBMU_wPY78aYaozFPURsHQW5PQQhrY_JNOPtbTaJr1pc4SyxM6Kwc2-gc8QUeMRuSvNRwrj9Bos-kgppUTImflhbkTTHPKw8-lO00EJrLZCmPloNu3tx9rMWdmSu62O8N8fYao5OW6LP56BiFBS1HUBYELA8kADPj5xD_4ttNn38nj0ivCDw9yAP7RglIX6tSphkt5nLpLBBMfu8AnsKmGuYOAuP5-4fpmu0S0wgQCzv3iI&csuir=1&mtid=DNngaeffI7jA4-EPkM_8uAk&udm=50i


import pandas as pd
import numpy as np
import os
import sys
from statsmodels.tsa.seasonal import STL

import pandas as pd

print(os.getcwd())
print(sys.executable) # proves the venv is used


print(sys.prefix != sys.base_prefix)



sales_df= pd.read_csv('outlier_identification/sales.csv')
events_df= pd.read_csv('outlier_identification/events.csv')
promos_df= pd.read_csv('outlier_identification/promos.csv')

# 1. Prepare your DataFrames
# sales_df: index or column 'week_start' (Monday start)
# events_df: column 'event_date' (e.g., '2023-11-24')
# promos_df: columns 'start_date', 'end_date'

# Date Alignment: It takes single-day events (like a Friday Black Friday) and figures out which Monday-to-Sunday week bucket they fall into.
# Overlap Logic: It looks at your BOGO start/end dates. Even if a BOGO only lasted 3 days, it flags the entire week as a "Promo Week."
# Flagging: It creates the is_special (or is_promo) column that your outlier detector uses to decide whether to compare a data point against "Normal" logic or "Promo" logic.



def prepare_and_join(sales_df, events_df, promos_df):
    # Ensure all date columns are datetime objects
    sales_df['week_start'] = pd.to_datetime(sales_df['week_start'])
    events_df['event_date'] = pd.to_datetime(events_df['event_date'])
    promos_df['start_date'] = pd.to_datetime(promos_df['start_date'])
    promos_df['end_date'] = pd.to_datetime(promos_df['end_date'])

    # Define the week end for each row
    sales_df['week_end'] = sales_df['week_start'] + pd.Timedelta(days=6)

    # Initialize flags
    sales_df['is_event'] = 0
    sales_df['is_promo'] = 0

    # 2. Map Single-Day Events (Black Friday, etc.) to Weeks
    for _, row in events_df.iterrows():
        # A week is flagged if the event falls between week_start and week_end
        mask = (sales_df['week_start'] <= row['event_date']) & (sales_df['week_end'] >= row['event_date'])
        sales_df.loc[mask, 'is_event'] = 1

    # 3. Map Multi-Day Promos (BOGO, etc.) to Weeks
    for _, row in promos_df.iterrows():
        # Overlap Logic: (StartA <= EndB) and (EndA >= StartB)
        mask = (sales_df['week_start'] <= row['end_date']) & (sales_df['week_end'] >= row['start_date'])
        sales_df.loc[mask, 'is_promo'] = 1

    # 4. Final Combined Special Flag
    sales_df['is_special'] = (sales_df['is_event'] == 1) | (sales_df['is_promo'] == 1)
    
    return sales_df

# Run the join
df_final = prepare_and_join(sales_df, events_df, promos_df)

# Now you can pass df_final into the STL outlier function from the above step



#----------------part2 ----------------------




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



df_final = prepare_and_join(sales_df, events_df, promos_df)

print(df_final.head())