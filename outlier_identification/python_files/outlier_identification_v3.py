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
from scipy.stats import f




print(os.getcwd())
os.chdir('/Users/rajonkar/Documents/MyLearning/')
#----------------venv verification
print(sys.executable) # proves the venv is used
print(sys.prefix != sys.base_prefix,"--If Value is True venv is being used")


# How to interpret the "Normal CV":
# CV < 0.2: Highly predictable. These are your "Cash Cow" SKUs.
# CV 0.2 – 0.5: Moderately predictable. You need a standard safety stock buffer.
# CV > 0.5: Erratic/Lumpy. These are "Chaotic" SKUs. Even without promotions, the demand is very hard to pin down.




sales_df= pd.read_csv('outlier_identification/sales.csv')
events_df= pd.read_csv('outlier_identification/events.csv')
promos_df= pd.read_csv('outlier_identification/promos.csv')



def prepare_and_join_granular(sales_df, events_df, promos_df):
    sales_df['week_start'] = pd.to_datetime(sales_df['week_start'])
    sales_df['week_end'] = sales_df['week_start'] + pd.Timedelta(days=6)
    
    # Initialize with 'Normal'
    sales_df['promo_type'] = 'Normal'

    # 1. Map Events first
    for _, row in events_df.iterrows():
        mask = (sales_df['week_start'] <= pd.to_datetime(row['event_date'])) & \
               (sales_df['week_end'] >= pd.to_datetime(row['event_date']))
        sales_df.loc[mask, 'promo_type'] = row['event']

    # 2. Map Promos and CHECK FOR OVERLAPS
    for _, row in promos_df.iterrows():
        mask = (sales_df['week_start'] <= pd.to_datetime(row['end_date'])) & \
               (sales_df['week_end'] >= pd.to_datetime(row['start_date']))
        
        # If the week is still 'Normal', just assign the promo type
        normal_mask = mask & (sales_df['promo_type'] == 'Normal')
        sales_df.loc[normal_mask, 'promo_type'] = row['type']
        
        # If the week ALREADY has an event, combine the names
        overlap_mask = mask & (sales_df['promo_type'] != 'Normal') & (sales_df['promo_type'] != row['type'])
        sales_df.loc[overlap_mask, 'promo_type'] = sales_df.loc[overlap_mask, 'promo_type'] + " + " + row['type']

    return sales_df

# Run the join
df_final = prepare_and_join_granular(sales_df, events_df, promos_df)

print(df_final.head(5))



import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import f

def detect_outliers_n_segment(df, multiplier=3):
    """
    Final Master System:
    - Adaptive History (STL vs Rolling)
    - Granular Promo Buckets (BOGO, Overlaps)
    - F-Test Pattern Classification
    - Intermittent & Regular Intermittent Detection
    - 1-Year Volume & Forecastability Analysis
    """
    all_results = []
    max_date = pd.to_datetime(df['week_start']).max()
    one_year_ago = max_date - pd.DateOffset(years=1)
    
    for sid, group in df.groupby('series_id'):
        group = group.sort_values('week_start').copy()
        n = len(group)
        
        # --- WEEK COUNTS & PERCENTAGES ---
        count_normal = (group['promo_type'] == 'Normal').sum()
        group['count_normal_weeks'] = count_normal
        group['count_promo_weeks'] = n - count_normal
        group['percent_normal_weeks'] = round((count_normal / n) * 100, 1)
        
        # 1. VOLUME & ZERO ANALYSIS
        last_year = group[group['week_start'] >= one_year_ago]
        group['last_1yr_sales_vol'] = last_year['sales'].sum()
        group['history_count'] = n
        
        zero_pct = (last_year['sales'] == 0).mean() if len(last_year) > 0 else 0
        is_intermittent = zero_pct > 0.30

        # 2. BASELINE DECOMPOSITION & PATTERN CLASSIFICATION
        if n >= 65:
            try:
                res = STL(group['sales'], period=52, robust=True, trend=105, seasonal=13).fit()
                group['trend'], group['seasonal'] = res.trend, res.seasonal
                group['baseline'] = res.trend + res.seasonal
                
                var_resid = res.resid.var()
                if var_resid > 0:
                    f_s = (res.resid + res.seasonal).var() / var_resid
                    f_t = (res.resid + res.trend).var() / var_resid
                    has_s = (1 - f.cdf(f_s, n-1, n-1)) < 0.01
                    has_t = (1 - f.cdf(f_t, n-1, n-1)) < 0.01
                    
                    if is_intermittent:
                        group['class'] = 'Regular Intermittent' if has_s else 'Intermittent'
                    else:
                        if has_t and has_s: group['class'] = 'Seasonal with Trend'
                        elif has_s: group['class'] = 'Seasonal without Trend'
                        elif has_t: group['class'] = 'Non-Seasonal with Trend'
                        else: group['class'] = 'Non-Seasonal and No Trend'
                else:
                    group['class'] = 'Stable'
            except:
                group['baseline'] = group['sales'].rolling(window=12, center=True, min_periods=1).median()
                group['class'] = 'Fallback (Rolling)'
        else:
            group['baseline'] = group['sales'].rolling(window=8, center=True, min_periods=1).median()
            group['class'] = 'New SKU (< 1.25yr)'

        # 3. FORECASTABILITY (CV on Normal Days only)
        normal_mask = (group['promo_type'] == 'Normal')
        if normal_mask.any():
            norm_data = group[normal_mask]
            norm_resid = norm_data['sales'] - (norm_data.get('baseline', norm_data['sales'].median()))
            noise_cv = norm_resid.std() / (norm_data['sales'].mean() + 1e-9)
            group['noise_cv'] = round(noise_cv, 3)
            group['forecast_score'] = 'High' if noise_cv < 0.2 else 'Medium' if noise_cv < 0.5 else 'Low'
        else:
            group['forecast_score'] = 'No Normal History'

        # 4. BUCKETED OUTLIER DETECTION
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
            
            out_mask = (group['resid'] > med_l + thresh) | (group['resid'] < med_l - thresh)
            group.loc[mask & out_mask, 'is_outlier'] = True
            
        all_results.append(group)
        
    return pd.concat(all_results)

def get_final_analysis_summary(df):
    """Summarises analysis and includes week-type counts and percentages."""
    summary_cols = [
        'series_id', 'class', 'forecast_score', 
        'last_1yr_sales_vol', 'history_count', 
        'count_normal_weeks', 'count_promo_weeks', 'percent_normal_weeks'
    ]
    summary = df[summary_cols].drop_duplicates()
    
    outlier_counts = df.groupby('series_id')['is_outlier'].sum().reset_index(name='outlier_count')
    return summary.merge(outlier_counts, on='series_id')



def get_portfolio_stratification_report(df):
    """
    Summarises the portfolio by Forecast Score and Pattern Class.
    Includes Volume per SKU to identify 'Heavy Hitters'.
    Strategic Use of "Volume per SKU":
        High Vol per SKU + Low Forecastability: These are your most dangerous items. They move a lot of money but are "chaotic." One bad forecast here results in massive lost sales or excess stock.
        Low Vol per SKU + High Outlier Count: These are your "noisy long-tail" items. They don't move much volume but they generate a lot of "false alarm" outlier alerts. You should likely ignore these or use a 5 MAD multiplier.
        High Vol per SKU + High Normal Weeks: These are your most efficient items. They are high volume, organic, and stable.
    """
    # 1. Get SKU-level metrics (removing the time dimension)
    sku_level = df.groupby('series_id').agg({
        'forecast_score': 'first',
        'class': 'first',
        'last_1yr_sales_vol': 'first',
        'is_outlier': 'sum',
        'count_normal_weeks': 'first'
    }).reset_index()

    total_portfolio_vol = sku_level['last_1yr_sales_vol'].sum()

    # 2. Group by Forecast Score and Class
    report = sku_level.groupby(['forecast_score', 'class']).agg(
        sku_count=('series_id', 'count'),
        total_vol_segment=('last_1yr_sales_vol', 'sum'),
        avg_outlier_count=('is_outlier', 'mean'),
        avg_normal_weeks=('count_normal_weeks', 'mean')
    ).reset_index()

    # 3. Add Proportion and Volume per SKU metrics
    report['vol_pct_of_total'] = round((report['total_vol_segment'] / total_portfolio_vol) * 100, 2)
    # The 'Heavy Hitter' metric
    report['volume_per_sku'] = round(report['total_vol_segment'] / report['sku_count'], 2)

    # 4. Clean up and Format
    report['avg_outlier_count'] = report['avg_outlier_count'].round(1)
    report['avg_normal_weeks'] = report['avg_normal_weeks'].round(1)
    
    # Final column order
    final_cols = [
        'forecast_score', 
        'class', 
        'vol_pct_of_total', 
        'volume_per_sku', 
        'avg_outlier_count', 
        'avg_normal_weeks',
        'sku_count'
    ]
    
    return report[final_cols].sort_values(['forecast_score', 'vol_pct_of_total'], ascending=[True, False])

# Usage
# stratification_report = get_portfolio_stratification_report(final_df)
# print(stratification_report)


# Execution
final_df = detect_outliers_n_segment(df_final)
df_report = get_final_analysis_summary(final_df)
print(final_df.head(5))


# Define your file path
filepath = Path("ver3/final_output.csv")
# Create the folder(s) if they don't exist
filepath.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(filepath, index=False)

df_report.to_csv("ver3/summary_report.csv", index=False)

stratification_report = get_portfolio_stratification_report(final_df)
stratification_report.to_csv("ver3/stratification_report.csv", index=False)

# print(final_df[['series_id', 'history_count', 'f_ratio_trend', 'class']].drop_duplicates())