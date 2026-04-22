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
os.chdir('/Users/rajonkar/Documents/MyLearning/v3')
#----------------venv verification
print(sys.executable) # proves the venv is used
print(sys.prefix != sys.base_prefix,"--If Value is True venv is being used")



# 1. Prepare your DataFrames
# sales_df: index or column 'week_start' (Monday start)
# events_df: column 'event_date' (e.g., '2023-11-24')
# promos_df: columns 'start_date', 'end_date'

# The prepare_and_join_granular function assigns a specific label to each week based on your events 
# and promotions files. This allows the subsequent detection function to compare BOGO weeks against other BOGO weeks, 
# and Black Friday against other Black Fridays, rather than grouping all spikes together.

import pandas as pd
import numpy as np

def generate_retail_sample(num_series=100):
    # Maximum potential window: 3 years (156 weeks)
    max_weeks = 156
    full_dates = pd.date_range(start='2021-01-04', periods=max_weeks, freq='W-MON')
    
    all_series = []
    
    for i in range(num_series):
        sid = f'Product_{i+1:03d}'
        
        # --- 1. Randomize History Length ---
        # Mix of: New (<65 weeks), Mid (65-104), and Full (105-156)
        history_choice = np.random.choice(['new', 'mid', 'full'], p=[0.2, 0.3, 0.5])
        if history_choice == 'new':
            weeks = np.random.randint(10, 60)
        elif history_choice == 'mid':
            weeks = np.random.randint(65, 100)
        else:
            weeks = np.random.randint(105, 156)
            
        # Select the most recent 'weeks' of data
        current_dates = full_dates[-weeks:]
        
        # --- 2. Randomize Patterns (for F-Test testing) ---
        has_trend = np.random.choice([True, False])
        has_season = np.random.choice([True, False])
        
        # Base Level
        base_level = np.random.randint(50, 200)
        
        # Trend component
        if has_trend:
            trend = np.linspace(base_level, base_level * np.random.uniform(1.2, 2.0), weeks)
        else:
            trend = np.full(weeks, base_level)
            
        # Seasonal component (Sine wave)
        if has_season:
            # Shift phase so seasonality isn't identical for every SKU
            phase = np.random.randint(0, 52)
            seasonality = 30 * np.sin(2 * np.pi * (np.arange(weeks) + phase) / 52)
        else:
            seasonality = np.zeros(weeks)
            
        noise = np.random.normal(0, 5, weeks)
        sales = trend + seasonality + noise
        
        # --- 3. Inject Events/Outliers ---
        # Note: Indexing must be handled carefully for varying lengths
        # Black Friday (Week 47ish of every year)
        for year_idx in [46, 46+52, 46+104]:
            if year_idx < weeks:
                sales[-(weeks-year_idx)] *= 3.5
        
        # High-side Outlier (Random location)
        high_idx = np.random.randint(0, weeks)
        sales[high_idx] += (sales.mean() * 5)
        
        # Low-side Outlier (Random location)
        low_idx = np.random.randint(0, weeks)
        sales[low_idx] *= 0.1
        
        temp_df = pd.DataFrame({'week_start': current_dates, 'sales': sales.clip(min=0)})
        temp_df['series_id'] = sid
        all_series.append(temp_df)
    
    sales_df = pd.concat(all_series).reset_index(drop=True)
    
    # 2. Events File (Fixed dates over 3 years)
    events_df = pd.DataFrame([
        {'event': 'Black Friday', 'event_date': '2021-11-26'},
        {'event': 'Black Friday', 'event_date': '2022-11-25'},
        {'event': 'Black Friday', 'event_date': '2023-11-24'},
        {'event': 'Memorial Day', 'event_date': '2022-05-30'},
        {'event': 'Memorial Day', 'event_date': '2023-05-29'}
    ])
    
    # 3. Promos File
    promos_df = pd.DataFrame([
        {'type': 'BOGO', 'start_date': '2022-05-15', 'end_date': '2022-05-25'},
        {'type': 'BOGO', 'start_date': '2023-05-15', 'end_date': '2023-05-25'},
        {'type': 'Disc_50', 'start_date': '2023-08-01', 'end_date': '2023-08-10'}
    ])
    
    return sales_df, events_df, promos_df

# Generate and save
sales_df, events_df, promos_df = generate_retail_sample()
print(f"Generated {sales_df['series_id'].nunique()} series with variable history lengths.")
print(sales_df.groupby('series_id').size().describe())


# Save for your notebook
sales_df.to_csv('outlier_identification/sales.csv', index=False)
events_df.to_csv('outlier_identification/events.csv', index=False)
promos_df.to_csv('outlier_identification/promos.csv', index=False)




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

def detect_outliers_for_catalog(df, multiplier=3):
    """
    Processes 100 series with history ranging from 1 to 156 weeks.
    Uses STL for items > 65 weeks and Rolling Median for newer items.
    """
    all_results = []
    
    for sid, group in df.groupby('series_id'):
        group = group.sort_values('week_start').copy()
        n = len(group)
        
        # 1. BASELINE SELECTION & PATTERN CLASSIFICATION
        if n >= 65: # Sufficient history for STL (1.25+ years)
            try:
                # trend=105 (stiff) and seasonal=13 (stable) for your 3-year max history
                res = STL(group['sales'], period=52, robust=True, trend=105, seasonal=13).fit()
                group['baseline'] = res.trend + res.seasonal
                
                # F-Test for Classification (Signal vs Noise)
                var_resid = res.resid.var()
                if var_resid > 0:
                    # Calculate F-stats: (Component + Resid) / Resid
                    f_trend = (res.resid + res.trend).var() / var_resid
                    f_season = (res.resid + res.seasonal).var() / var_resid
                    
                    # Convert F-stats to P-values
                    has_trend = (1 - f.cdf(f_trend, n-1, n-1)) < 0.01
                    has_season = (1 - f.cdf(f_season, n-1, n-1)) < 0.01
                    
                    if has_trend and has_season: group['class'] = 'Seasonal with Trend'
                    elif has_season: group['class'] = 'Seasonal without Trend'
                    elif has_trend: group['class'] = 'Non-Seasonal with Trend'
                    else: group['class'] = 'Non-Seasonal and No Trend'
                else:
                    group['class'] = 'Stable'
            except:
                group['baseline'] = group['sales'].rolling(window=12, center=True, min_periods=1).median()
                group['class'] = 'Fallback (Rolling Median)'
        else:
            # New items (< 65 weeks): Use Rolling Median
            group['baseline'] = group['sales'].rolling(window=8, center=True, min_periods=1).median()
            group['class'] = 'New SKU (< 1.25yr)'

        group['resid'] = group['sales'] - group['baseline']
        group['is_outlier'] = False

        # 2. BUCKETED OUTLIER DETECTION (By promo_type)
        for etype in group['promo_type'].unique():
            mask = (group['promo_type'] == etype)
            subset = group.loc[mask, 'resid']
            
            # Peer-group logic (MAD) if N >= 2, else 50% baseline safety net
            if len(subset) >= 2:
                med_lift = subset.median()
                mad = (subset - med_lift).abs().median()
                thresh = multiplier * (1.4826 * mad)
            else:
                med_lift = subset.median()
                thresh = group.loc[mask, 'baseline'].mean() * 0.5
            
            # Set Bounds
            group.loc[mask, 'upper_limit'] = group.loc[mask, 'baseline'] + med_lift + thresh
            group.loc[mask, 'lower_limit'] = group.loc[mask, 'baseline'] + med_lift - thresh
            
            # Final Outlier Flagging
            outlier_mask = (group['resid'] > (med_lift + thresh)) | (group['resid'] < (med_lift - thresh))
            group.loc[mask & outlier_mask, 'is_outlier'] = True
            
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