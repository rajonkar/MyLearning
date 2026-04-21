# https://www.google.com/search?q=i+have+a+time+series+data+that+has+promotions+like+BOGO.+If+i+were+to+identify+outliers+on+both+high+and+low+side+how+do+i+do+that&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEJMzU1MTJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpaEWjvZ2Py1XXV8d8KvlI3izfzqgn7395CNCvYdZRuZ5U71YjigLP8fxOdtGNi00AID0tRCZHJ8fgh0HcZOsGj_K3nSeb8S5ovAsHNcxAlbWFsabvN_f67JLYAW48ZvNV06CZBv371fkqFwGNtqaspixefiL07nDsfYksrWsZNxLdEOW-lWrtts9K9tw3he9tfl_l_A&ved=2ahUKEwi6oNvvsfKTAxXCwTgGHTbXJ00Q0NsOegQIAxAA&aep=10&ntc=1&mstk=AUtExfCWI97WMa3QzhHKJgGY4pHZJG50F85ZuAdKieAyDiYNmyA71qG-0sdZuMZ7QZlBslEyYa64U-P-Q6eSwasjqoArFCWqL-3VDba1b9UpLpP-5HjDFtGYgvTn1911QBuG07XsZpuWREElYQBbAaFdaT0d-bi7JnUVwtzDfPFuc0yzcii4jsPmMvWm0Z8y5MZ1-HTR9tHmzP57iakf_DfY3o4DvNClkBMU_wPY78aYaozFPURsHQW5PQQhrY_JNOPtbTaJr1pc4SyxM6Kwc2-gc8QUeMRuSvNRwrj9Bos-kgppUTImflhbkTTHPKw8-lO00EJrLZCmPloNu3tx9rMWdmSu62O8N8fYao5OW6LP56BiFBS1HUBYELA8kADPj5xD_4ttNn38nj0ivCDw9yAP7RglIX6tSphkt5nLpLBBMfu8AnsKmGuYOAuP5-4fpmu0S0wgQCzv3iI&csuir=1&mtid=DNngaeffI7jA4-EPkM_8uAk&udm=50i
# logic explained below



# seperate events and prmo will not be grouped in this code

# modify the loop to iterate over specific event types instead of just a True/False flag.
















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

# The updated prepare_and_join_granular function assigns a specific label to each week based on your events 
# and promotions files. This allows the subsequent detection function to compare BOGO weeks against other BOGO weeks, 
# and Black Friday against other Black Fridays, rather than grouping all spikes together.


def prepare_and_join_granular(sales_df, events_df, promos_df):
    # Standardize dates
    sales_df['week_start'] = pd.to_datetime(sales_df['week_start'])
    sales_df['week_end'] = sales_df['week_start'] + pd.Timedelta(days=6)
    
    # Default category is 'Normal'
    sales_df['promo_type'] = 'Normal'

    # Map Events (e.g., Black Friday, Memorial Day)
    for _, row in events_df.iterrows():
        mask = (sales_df['week_start'] <= pd.to_datetime(row['event_date'])) & \
               (sales_df['week_end'] >= pd.to_datetime(row['event_date']))
        sales_df.loc[mask, 'promo_type'] = row['event']

    # Map Promos (e.g., BOGO, Buy2Get2)
    # This overwrites 'Normal' with the specific promo name
    for _, row in promos_df.iterrows():
        mask = (sales_df['week_start'] <= pd.to_datetime(row['end_date'])) & \
               (sales_df['week_end'] >= pd.to_datetime(row['start_date']))
        sales_df.loc[mask, 'promo_type'] = row['type']

    return sales_df

# Run the join
df_final = prepare_and_join_granular(sales_df, events_df, promos_df)
df_final.to_csv('outlier_identification/df_final_v2.csv', index=False)

# Now you can pass df_final into the STL outlier function from the above step


# 2. Multi-Event Outlier Detection
# This function calculates separate thresholds for every unique label in your promo_type column.

def detect_multi_event_outliers(df, target_col='sales', period=52, multiplier=3):
    # Robust STL identifies 'Organic Baseline' (Trend + Seasonal)
    stl = STL(df[target_col], period=period, robust=True)
    res = stl.fit()
    df['resid'] = res.resid
    df['baseline'] = res.trend + res.seasonal
    
    df['is_outlier'] = False
    df['upper_limit'] = np.nan
    df['lower_limit'] = np.nan

    # Group residuals by specific promo type (e.g., compare all BOGOs to each other)
    for etype in df['promo_type'].unique():
        mask = (df['promo_type'] == etype)
        subset_resid = df.loc[mask, 'resid']
        
        # Need at least 2 occurrences of a promo type to calculate deviation (MAD)
        if len(subset_resid) < 2:
            continue
            
        median_lift = subset_resid.median()
        mad = (subset_resid - median_lift).abs().median()
        thresh = multiplier * (1.4826 * mad)
        
        # Calculate limits: Baseline + Typical Lift for this specific promo +/- Threshold
        df.loc[mask, 'upper_limit'] = df.loc[mask, 'baseline'] + median_lift + thresh
        df.loc[mask, 'lower_limit'] = df.loc[mask, 'baseline'] + median_lift - thresh
        
        # Flag outliers within this specific bucket
        outlier_cond = (df['resid'] > (median_lift + thresh)) | \
                       (df['resid'] < (median_lift - thresh))
        df.loc[mask & outlier_cond, 'is_outlier'] = True
            
    return df
