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

# The prepare_and_join_granular function assigns a specific label to each week based on your events 
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

# The below function segments by Label: It looks at your "Promo" or "Event" column first.
# It sets a "Local" Bar: It calculates the average/threshold only for those specific rows.
# It compares apples to apples: It only flags a point as an outlier if it’s weird for that specific promo.

def detect_multi_event_outliers(df, target_col='sales', period=52, multiplier=3):
    # Robust STL identifies 'Organic Baseline' (Trend + Seasonal)
    # 1. GENERATE BASELINE: Use STL to isolate 'Organic' sales (Trend + Season)
    # We use the residuals (Actual - Baseline) to measure the specific 'lift' of promos.
    stl = STL(df[target_col], period=period, robust=True)
    res = stl.fit()
    df['resid'] = res.resid
    df['baseline'] = res.trend + res.seasonal
    
    df['is_outlier'] = False
    df['upper_limit'] = np.nan
    df['lower_limit'] = np.nan


    # 2. CONTEXTUAL ANALYSIS: Group by promo_type to compare apples to apples
    # Group residuals by specific promo type (e.g., compare all BOGOs to each other)
    for etype in df['promo_type'].unique():
        mask = (df['promo_type'] == etype)
        subset_resid = df.loc[mask, 'resid']
        
        
        # 3. STATISTICAL VALIDATION: Ensure we have enough data points.
        #   # Need at least 2 occurrences of a promo type to calculate deviation (MAD)
        # If we have only 1 event, MAD is 0 and we cannot statistically define an outlier.
        # Skip outlier detection for unique/single-occurrence events
            # to avoid false positives caused by a zero-width threshold
        if len(subset_resid) < 2: # why this code- see explanation below -- find why  len(subset_resid) < 2
            continue
            
        median_lift = subset_resid.median()
        mad = (subset_resid - median_lift).abs().median()
        thresh = multiplier * (1.4826 * mad)
        #Multiplying by 1.4826 "stretches" the MAD so it behaves like a Standard Deviation, but without the weakness of being skewed by your massive promo spikes.
   
        # 4. SET THE FENCE: Organic Baseline + Typical Promo Lift +/- Tolerance
        # Calculate limits: Baseline + Typical Lift for this specific promo +/- Threshold
        df.loc[mask, 'upper_limit'] = df.loc[mask, 'baseline'] + median_lift + thresh
        df.loc[mask, 'lower_limit'] = df.loc[mask, 'baseline'] + median_lift - thresh
        
        # 5. FLAG OUTLIERS: Identify points that are extreme even for this event type
        outlier_cond = (df['resid'] > (median_lift + thresh)) | \
                       (df['resid'] < (median_lift - thresh))
        df.loc[mask & outlier_cond, 'is_outlier'] = True
            
    return df


import matplotlib.pyplot as plt

def plot_multi_event_outliers(df, series_id):
    """
    Plots sales, the organic baseline, and the dynamic thresholds 
    for each specific promo/event type.
    """
    plt.figure(figsize=(15, 8))
    
    # 1. Plot Raw Data and Organic Baseline
    plt.plot(df['week_start'], df['sales'], color='gray', alpha=0.3, label='Actual Sales')
    plt.plot(df['week_start'], df['baseline'], color='blue', linestyle='--', alpha=0.6, label='Organic Baseline (Trend+Season)')

    # 2. Plot the dynamic Outlier Boundaries (Upper and Lower)
    # Using .step() with 'mid' creates the 'shelf' effect for different promos
    plt.step(df['week_start'], df['upper_limit'], color='red', alpha=0.4, label='Upper Bound (Event Adjusted)', where='mid')
    plt.step(df['week_start'], df['lower_limit'], color='orange', alpha=0.4, label='Lower Bound (Event Adjusted)', where='mid')

    # 3. Highlight Outliers (Red X)
    outliers = df[df['is_outlier'] == True]
    plt.scatter(outliers['week_start'], outliers['sales'], color='red', marker='x', s=120, label='Detected Outliers', zorder=5)

    # 4. Color-code the background based on promo_type for clarity
    unique_promos = df[df['promo_type'] != 'Normal']['promo_type'].unique()
    colors = plt.cm.get_cmap('Set3', len(unique_promos))
    
    for i, p_type in enumerate(unique_promos):
        promo_weeks = df[df['promo_type'] == p_type]
        for start_date in promo_weeks['week_start']:
            plt.axvspan(start_date, start_date + pd.Timedelta(days=6), color=colors(i), alpha=0.1)

    plt.title(f'Multi-Event Outlier Detection: {series_id}', fontsize=16)
    plt.ylabel('Sales Volume')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Usage:





def summarize_promo_performance(df):
    """
    Groups outliers by promo type and side (High vs Low) 
    to see which marketing events are the most unstable.
    """
    # 1. Only look at detected outliers
    outliers_df = df[df['is_outlier'] == True].copy()
    
    # 2. Determine if the outlier was High or Low relative to the expected lift
    outliers_df['side'] = np.where(outliers_df['resid'] > 0, 'High-Side (Over-perf)', 'Low-Side (Under-perf)')
    
    # 3. Create the summary table
    summary = outliers_df.groupby(['promo_type', 'side']).size().unstack(fill_value=0)
    
    # 4. Calculate total instances of each promo to get an "Anomaly Rate"
    total_counts = df['promo_type'].value_counts()
    summary['Total_Occurrences'] = total_counts
    summary['Anomaly_Rate_%'] = (summary.sum(axis=1) / total_counts * 100).round(1)
    
    return summary.sort_values('Anomaly_Rate_%', ascending=False)




# First, detect outliers
df_result = detect_multi_event_outliers(df_final, target_col='sales', period=52)
# plot_multi_event_outliers(df_result, 'Product_001')

# Usage:
performance_report = summarize_promo_performance(df_result)
print(performance_report)





import matplotlib.pyplot as plt
import matplotlib.cm as cm




def plot_outlier_results(df, series_id):
    # Filter for the specific series
    df_plot = df[df['series_id'] == series_id].copy().sort_values('week_start')
    
    plt.figure(figsize=(15, 7))
    
    # 1. Plot Actual Sales and Organic Baseline
    plt.plot(df_plot['week_start'], df_plot['sales'], color='gray', alpha=0.3, label='Actual Sales', linewidth=1)
    plt.plot(df_plot['week_start'], df_plot['baseline'], color='blue', linestyle='--', alpha=0.5, label='Organic Baseline')

    # 2. Plot Dynamic Upper and Lower Limits (the "Shelves")
    # .step(where='mid') makes the boundary lines jump exactly at the week change
    plt.step(df_plot['week_start'], df_plot['upper_limit'], color='red', alpha=0.4, label='Upper Bound', where='mid')
    plt.step(df_plot['week_start'], df_plot['lower_limit'], color='orange', alpha=0.4, label='Lower Bound', where='mid')

    # 3. Highlight Detected Outliers (Red X)
    outliers = df_plot[df_plot['is_outlier'] == True]
    plt.scatter(outliers['week_start'], outliers['sales'], 
                marker='x', color='red', s=120, label='Detected Outliers', zorder=10)

    # 4. Color-code the Background by Promo Type
    # This helps visualize which "bucket" the week belongs to
    unique_promos = [p for p in df_plot['promo_type'].unique() if p != 'Normal']
    cmap = cm.get_cmap('tab10', len(unique_promos))
    
    for i, p_type in enumerate(unique_promos):
        promo_weeks = df_plot[df_plot['promo_type'] == p_type]
        for start_date in promo_weeks['week_start']:
            plt.axvspan(start_date, start_date + pd.Timedelta(days=6), 
                        color=cmap(i), alpha=0.1, label=f'Event: {p_type}' if start_date == promo_weeks['week_start'].iloc[0] else "")

    plt.title(f'Granular Outlier Detection: {series_id}', fontsize=15)
    plt.xlabel('Week Start Date')
    plt.ylabel('Sales Volume')
    
    # Place legend outside to avoid clutter
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


plot_outlier_results(df_final[df_final['series_id'] == 'Product_001'], 'Product_001')

print("print the troibleshooting")
# (df_final[df_final['series_id'] == 'Product_001']).to_csv('outlier_identification/product_001_analysis_ver.csv', index=False)



#--------------- ALL IN ONE -----------

# Here is the complete, production-ready script. 
# It loops through all 100 series, applies the granular detection logic, and outputs a summary of every outlier 
# found across your entire catalog.


# all above combined into one function for easier execution and reporting


# def process_all_series(sales_df, events_df, promos_df, multiplier=3):
#     """
#     1. Joins data granularly
#     2. Loops through every series_id
#     3. Detects outliers per promo_type
#     4. Returns a master dataframe and a summary report
#     """
    
#     # --- STEP 1: PREPARE AND JOIN ---
#     sales_df['week_start'] = pd.to_datetime(sales_df['week_start'])
#     sales_df['week_end'] = sales_df['week_start'] + pd.Timedelta(days=6)
#     sales_df['promo_type'] = 'Normal'

#     # Map Events
#     for _, row in events_df.iterrows():
#         mask = (sales_df['week_start'] <= pd.to_datetime(row['event_date'])) & \
#                (sales_df['week_end'] >= pd.to_datetime(row['event_date']))
#         sales_df.loc[mask, 'promo_type'] = row['event']

#     # Map Promos (BOGO, etc.)
#     for _, row in promos_df.iterrows():
#         mask = (sales_df['week_start'] <= pd.to_datetime(row['end_date'])) & \
#                (sales_df['week_end'] >= pd.to_datetime(row['start_date']))
#         sales_df.loc[mask, 'promo_type'] = row['type']

#     # --- STEP 2: LOOP THROUGH SERIES ---
#     all_results = []
    
#     for sid, group in sales_df.groupby('series_id'):
#         # Sort by date for STL
#         group = group.sort_values('week_start')
        
#         # Apply STL
#         stl = STL(group['sales'], period=52, robust=True)
#         res = stl.fit()
#         group['resid'] = res.resid
#         group['baseline'] = res.trend + res.seasonal
#         group['is_outlier'] = False
        
#         # Detect Outliers per Category
#         for etype in group['promo_type'].unique():
#             mask = (group['promo_type'] == etype)
#             subset = group.loc[mask, 'resid']
            
#             if len(subset) >= 2:
#                 median_lift = subset.median()
#                 mad = (subset - median_lift).abs().median()
#                 thresh = multiplier * (1.4826 * mad)
                
#                 # Flag
#                 outlier_mask = (group['resid'] > (median_lift + thresh)) | \
#                                (group['resid'] < (median_lift - thresh))
#                 group.loc[mask & outlier_mask, 'is_outlier'] = True
        
#         all_results.append(group)
    
#     final_df = pd.concat(all_results)
    
#     # --- STEP 3: GENERATE SUMMARY ---
#     outliers_only = final_df[final_df['is_outlier'] == True].copy()
#     outliers_only['side'] = np.where(outliers_only['resid'] > 0, 'High (Spike)', 'Low (Dip)')
    
#     summary = outliers_only.groupby(['series_id', 'promo_type', 'side']).size().reset_index(name='count')
    
#     return final_df, summary

# # --- EXECUTION ---


# sales_df= pd.read_csv('outlier_identification/sales.csv')
# events_df= pd.read_csv('outlier_identification/events.csv')
# promos_df= pd.read_csv('outlier_identification/promos.csv')
# final_data, outlier_summary = process_all_series(sales_df, events_df, promos_df)

# # Save to CSV
# outlier_summary.to_csv('outlier_identification/outlier_analysis_report.csv', index=False)
# # print("Processing complete. Top outliers:")
# # print(outlier_summary.head(15))




#### why len(subset_resid) < 2

# if you only have one occurrence of a promo type, the mad (Median Absolute Deviation) will be zero.
# Here is the step-by-step math for a single event:
# Subset Residuals: Let's say you have one "Special Sale" and its residual is 500.
# Median Lift: median([500]) = 500.
# MAD Calculation: median(abs(500 - 500)) = 0.
# Threshold: 3 * (1.4826 * 0) = 0.
# Outlier Condition: Is 500 > (500 + 0)?
# The answer is False.
# Because the mad becomes 0, the "fence" sits exactly on the data point itself. It has no "width." Therefore, a single event can never be "further away" from itself than 0.
# The "N=1" Problem
# As you can see in the code you shared, there is a safety check for this: