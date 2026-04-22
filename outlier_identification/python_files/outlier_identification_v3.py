# https://www.google.com/search?q=i+have+a+time+series+data+that+has+promotions+like+BOGO.+If+i+were+to+identify+outliers+on+both+high+and+low+side+how+do+i+do+that&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEJMzU1MTJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpaEWjvZ2Py1XXV8d8KvlI3izfzqgn7395CNCvYdZRuZ5U71YjigLP8fxOdtGNi00AID0tRCZHJ8fgh0HcZOsGj_K3nSeb8S5ovAsHNcxAlbWFsabvN_f67JLYAW48ZvNV06CZBv371fkqFwGNtqaspixefiL07nDsfYksrWsZNxLdEOW-lWrtts9K9tw3he9tfl_l_A&ved=2ahUKEwi6oNvvsfKTAxXCwTgGHTbXJ00Q0NsOegQIAxAA&aep=10&ntc=1&mstk=AUtExfCWI97WMa3QzhHKJgGY4pHZJG50F85ZuAdKieAyDiYNmyA71qG-0sdZuMZ7QZlBslEyYa64U-P-Q6eSwasjqoArFCWqL-3VDba1b9UpLpP-5HjDFtGYgvTn1911QBuG07XsZpuWREElYQBbAaFdaT0d-bi7JnUVwtzDfPFuc0yzcii4jsPmMvWm0Z8y5MZ1-HTR9tHmzP57iakf_DfY3o4DvNClkBMU_wPY78aYaozFPURsHQW5PQQhrY_JNOPtbTaJr1pc4SyxM6Kwc2-gc8QUeMRuSvNRwrj9Bos-kgppUTImflhbkTTHPKw8-lO00EJrLZCmPloNu3tx9rMWdmSu62O8N8fYao5OW6LP56BiFBS1HUBYELA8kADPj5xD_4ttNn38nj0ivCDw9yAP7RglIX6tSphkt5nLpLBBMfu8AnsKmGuYOAuP5-4fpmu0S0wgQCzv3iI&csuir=1&mtid=DNngaeffI7jA4-EPkM_8uAk&udm=50i


# logic explained below



# seperate events and prmo will not be grouped in this code

# modify the loop to iterate over specific event types instead of just a True/False flag.


# In STL The trend parameter must be an odd integer and should be larger than (1.5 * period) / (1 - 1.5/seasonal).

import pandas as pd
import numpy as np
import os
import sys
from statsmodels.tsa.seasonal import STL
from pathlib import Path

import pandas as pd



print(os.getcwd())
os.chdir('/Users/rajonkar/Documents/MyLearning/')
#----------------venv verification
print(sys.executable) # proves the venv is used
print(sys.prefix != sys.base_prefix,"--If Value is True venv is being used")




sales_df= pd.read_csv('outlier_identification/sales.csv')
events_df= pd.read_csv('outlier_identification/events.csv')
promos_df= pd.read_csv('outlier_identification/promos.csv')



def detect_outliers_for_catalog(df, multiplier=3):
    all_results = []
    
    for sid, group in df.groupby('series_id'):
        group = group.sort_values('week_start').copy()
        n = len(group)
        group['history_count'] = n
        
        # 1. BASELINE SELECTION & PATTERN CLASSIFICATION
        if n >= 65: 
            try:
                res = STL(group['sales'], period=52, robust=True, trend=105, seasonal=13).fit()
                group['baseline'] = res.trend + res.seasonal
                
                var_resid = res.resid.var()
                if var_resid > 0:
                    f_trend = (res.resid + res.trend).var() / var_resid
                    f_season = (res.resid + res.seasonal).var() / var_resid
                    p_trend = 1 - f.cdf(f_trend, n-1, n-1)
                    p_season = 1 - f.cdf(f_season, n-1, n-1)
                    
                    has_trend = p_trend < 0.01
                    has_season = p_season < 0.01
                    
                    if has_trend and has_season: group['class'] = 'Seasonal with Trend'
                    elif has_season: group['class'] = 'Seasonal without Trend'
                    elif has_trend: group['class'] = 'Non-Seasonal with Trend'
                    else: group['class'] = 'Non-Seasonal and No Trend'
                    
                    # Store Stats
                    group['f_ratio_trend'] = round(f_trend, 3)
                    group['f_ratio_season'] = round(f_season, 3)
                else:
                    group['class'] = 'Stable'
            except:
                group['baseline'] = group['sales'].rolling(window=12, center=True, min_periods=1).median()
                group['class'] = 'Fallback'
        else:
            group['baseline'] = group['sales'].rolling(window=8, center=True, min_periods=1).median()
            group['class'] = 'New SKU'

        # 2. FORECASTABILITY SCORE (The "Stability" Check)
        # We look at 'Normal' days only to see the base noise
        normal_resid = group[group['promo_type'] == 'Normal']['sales'] - group[group['promo_type'] == 'Normal']['baseline']
        avg_sales = group[group['promo_type'] == 'Normal']['sales'].mean()
        
        if avg_sales > 0:
            noise_cv = normal_resid.std() / avg_sales
            group['noise_cv'] = round(noise_cv, 3)
            if noise_cv < 0.2: group['forecast_score'] = 'High (Stable)'
            elif noise_cv < 0.5: group['forecast_score'] = 'Medium (Buffer)'
            else: group['forecast_score'] = 'Low (Chaotic)'
        else:
            group['noise_cv'] = np.nan
            group['forecast_score'] = 'Inactive'

        # 3. OUTLIER DETECTION (Same as before)
        group['resid'] = group['sales'] - group['baseline']
        group['is_outlier'] = False
        for etype in group['promo_type'].unique():
            mask = (group['promo_type'] == etype)
            subset = group.loc[mask, 'resid']
            if len(subset) >= 2:
                med_l = subset.median()
                mad = (subset - med_l).abs().median()
                thresh = multiplier * (1.4826 * mad)
            else:
                med_l = subset.median()
                thresh = group.loc[mask, 'baseline'].mean() * 0.5
            
            group.loc[mask, 'upper_limit'] = group.loc[mask, 'baseline'] + med_l + thresh
            group.loc[mask, 'lower_limit'] = group.loc[mask, 'baseline'] + med_l - thresh
            group.loc[mask & ((group['resid'] > med_l + thresh) | (group['resid'] < med_l - thresh)), 'is_outlier'] = True
            
        all_results.append(group)
    
    return pd.concat(all_results)


# Execution
# final_df = detect_outliers_for_catalog(sales_df)
final_df = detect_outliers_for_catalog(df_final)

print(final_df.head(5))


# Define your file path
filepath = Path("ver3/final_output.csv")
# Create the folder(s) if they don't exist
filepath.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(filepath, index=False)

print(final_df[['series_id', 'history_count', 'f_ratio_trend', 'class']].drop_duplicates())