# https://www.google.com/search?q=i+have+a+time+series+data+that+has+promotions+like+BOGO.+If+i+were+to+identify+outliers+on+both+high+and+low+side+how+do+i+do+that&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEJMzU1MTJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8&fbs=ADc_l-aN0CWEZBOHjofHoaMMDiKpaEWjvZ2Py1XXV8d8KvlI3izfzqgn7395CNCvYdZRuZ5U71YjigLP8fxOdtGNi00AID0tRCZHJ8fgh0HcZOsGj_K3nSeb8S5ovAsHNcxAlbWFsabvN_f67JLYAW48ZvNV06CZBv371fkqFwGNtqaspixefiL07nDsfYksrWsZNxLdEOW-lWrtts9K9tw3he9tfl_l_A&ved=2ahUKEwi6oNvvsfKTAxXCwTgGHTbXJ00Q0NsOegQIAxAA&aep=10&ntc=1&mstk=AUtExfCWI97WMa3QzhHKJgGY4pHZJG50F85ZuAdKieAyDiYNmyA71qG-0sdZuMZ7QZlBslEyYa64U-P-Q6eSwasjqoArFCWqL-3VDba1b9UpLpP-5HjDFtGYgvTn1911QBuG07XsZpuWREElYQBbAaFdaT0d-bi7JnUVwtzDfPFuc0yzcii4jsPmMvWm0Z8y5MZ1-HTR9tHmzP57iakf_DfY3o4DvNClkBMU_wPY78aYaozFPURsHQW5PQQhrY_JNOPtbTaJr1pc4SyxM6Kwc2-gc8QUeMRuSvNRwrj9Bos-kgppUTImflhbkTTHPKw8-lO00EJrLZCmPloNu3tx9rMWdmSu62O8N8fYao5OW6LP56BiFBS1HUBYELA8kADPj5xD_4ttNn38nj0ivCDw9yAP7RglIX6tSphkt5nLpLBBMfu8AnsKmGuYOAuP5-4fpmu0S0wgQCzv3iI&csuir=1&mtid=DNngaeffI7jA4-EPkM_8uAk&udm=50i
# logic explained below



# Currently, the code groups all special days (BOGO, Buy2Get2, Black Friday) into one single "Special" bucket. This means it calculates one median lift for everything.

# Since you are using Robust STL, the baseline for a BOGO week is actually computed using the data from the surrounding "normal" weeks, effectively ignoring the BOGO spike itself.
# Here is the step-by-step logic the algorithm follows:
# 1. It "Blurs" the Spike
# When the STL algorithm looks at a BOGO week, it sees a massive value. Because you set robust=True, the algorithm identifies this as an extreme deviation and assigns it a very low weight (close to zero).
# 2. It Interpolates from Neighbors
# Instead of letting that high BOGO number pull the baseline up, the algorithm looks at the weeks before and after the promotion. It "bridges" the gap between them to estimate what the sales would have been if the promotion hadn't happened.
# The Trend: Is calculated by looking at the long-term direction of the surrounding months.
# The Seasonality: Is calculated by looking at what happened in that same week (e.g., Week 20) in previous years.
# 3. The Resulting Baseline
# The baseline (trend + seasonal) for that BOGO week ends up representing your organic demand.
# 4. How the "Promo Logic" then applies
# In the function we wrote, we don't just use that organic baseline to find outliers. We do this:
# Step A: Calculate the Residual (Actual BOGO Sales minus the Organic Baseline).
# Step B: Group all those "BOGO Residuals" together and find their Median. This Median is the "Expected Promo Lift."
# Step C: Your Final Promo Limit = Organic Baseline + Median Promo Lift ± Threshold.
# In short: The baseline is computed by ignoring the promotion, but the outlier limit is then adjusted upward by the typical lift seen in other promotions.
# Does that clarify why the baseline stays low even when sales are high?
















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
df_final.to_csv('outlier_identification/df_final.csv', index=False)

# Now you can pass df_final into the STL outlier function from the above step



#----------------part2 ----------------------




# 1. Load your data
# df = pd.read_csv('sales_data.csv') 
# events = pd.read_csv('events.csv') # Contains Memorial Day, Black Friday, etc.
# promos = pd.read_csv('promos.csv') # Contains BOGO, Discounts

# For this example, let's assume 'df' is already loaded and merged with flags
# df['is_special'] = (df['is_event'] == 1) | (df['is_promo'] == 1)

# def detect_time_series_outliers(df, target_col='sales', period=52):
#     """
#     Identifies outliers by decomposing seasonality and checking 
#     residuals in two separate buckets (Normal vs Special Days).
#     """
    
#     # 2. Robust STL Decomposition (Removes Trend & 52-week Seasonality)
#     # robust=True ensures spikes don't bleed into the trend line
#     stl = STL(df[target_col], period=period, robust=True)
#     result = stl.fit()
#     df['resid'] = result.resid

#     # 3. Define the Outlier Detection Logic (Modified Z-Score/MAD)
#     def get_thresholds(series):
#         # Using Median Absolute Deviation (MAD) for high robustness
#         median = series.median()
#         mad = (series - median).abs().median()
#         # 1.4826 makes MAD comparable to Standard Deviation
#         lower = median - 3 * (1.4826 * mad)
#         upper = median + 3 * (1.4826 * mad)
#         return lower, upper

#     # 4. Apply thresholds separately to avoid bucket contamination
#     df['is_outlier'] = False
    
#     for is_special_day in [True, False]:
#         mask = (df['is_special'] == is_special_day)
#         if mask.any():
#             subset_resid = df.loc[mask, 'resid']
#             lower, upper = get_thresholds(subset_resid)
            
#             # Flag both High-side and Low-side
#             outlier_mask = (df['resid'] < lower) | (df['resid'] > upper)
#             df.loc[mask & outlier_mask, 'is_outlier'] = True
            
#     return df



# modify the funcion above to ad lower an dupper limit and  include the multiplier of MAD as config

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

def detect_time_series_outliers(df, target_col='sales', period=52, multiplier=3):
    """
    Identifies outliers and adds the calculated upper/lower limits to the dataframe.
    """
    # 1. Decompose to get the baseline (Trend + Seasonality)
    stl = STL(df[target_col], period=period, robust=True)
    result = stl.fit()
    
    df['resid'] = result.resid
    df['baseline'] = result.trend + result.seasonal #baseline: This captures the "predictable" part of your data.
    
    # Initialize result columns
    df['is_outlier'] = False
    df['upper_limit'] = np.nan
    df['lower_limit'] = np.nan

    # 2. Process Normal and Special buckets separately
    for is_special in [True, False]:
        mask = (df['is_special'] == is_special)
        if mask.any():
            subset_resid = df.loc[mask, 'resid']
            
            # Robust Statistics
            median = subset_resid.median()
            mad = (subset_resid - median).abs().median()
            
            # Calculate Thresholds (1.4826 makes MAD comparable to Std Dev)
            thresh = multiplier * (1.4826 * mad)
            
            # 3. Set the Limits (Baseline + Median Residual + Threshold)
            # We include the median residual in case promos have a constant offset
            df.loc[mask, 'upper_limit'] = df.loc[mask, 'baseline'] + median + thresh
            df.loc[mask, 'lower_limit'] = df.loc[mask, 'baseline'] + median - thresh
            
            # 4. Flag Outliers
            outlier_mask = (df['resid'] > (median + thresh)) | (df['resid'] < (median - thresh))
            df.loc[mask & outlier_mask, 'is_outlier'] = True
            
    return df


# Example usage:
# df_cleaned = detect_time_series_outliers(df)
# print(df_cleaned[df_cleaned['is_outlier']])



df_final = prepare_and_join(sales_df, events_df, promos_df)

# Detect outliers
df_final = detect_time_series_outliers(df_final, target_col='sales', period=52)

print(df_final.head())



#This visualization snippet helps you confirm that the Robust STL logic is correctly distinguishing between legitimate promotional spikes and actual anomalies.

import matplotlib.pyplot as plt

def plot_outlier_results(df, series_id):
    plt.figure(figsize=(14, 6))
    
    # 1. Plot the actual sales line
    plt.plot(df['week_start'], df['sales'], color='#1f77b4', label='Weekly Sales', alpha=0.8, linewidth=1.5)
    
    # 2. Highlight Known Events (Green dots)
    special_days = df[df['is_special'] == True]
    plt.scatter(special_days['week_start'], special_days['sales'], 
                color='green', label='Promos/Events (BOGO, BF)', s=60, edgecolors='white', zorder=5)
    
    # 3. Highlight Detected Outliers (Red X)
    outliers = df[df['is_outlier'] == True]
    plt.scatter(outliers['week_start'], outliers['sales'], 
                marker='x', color='red', s=100, label='Detected Outliers', zorder=6)
    
    plt.title(f'Outlier Detection for {series_id}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Sales Volume')
    plt.legend(frameon=True, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Example usage for one series
# plot_outlier_results(df_final[df_final['series_id'] == 'Product_001'], 'Product_001')



# modify above to add the lower and upper limit

# def plot_outlier_results(df, series_id):
#     plt.figure(figsize=(15, 7))
    
#     # 1. Calculate the 'Expected' line (Trend + Seasonal)
#     # This is what the model thinks "Normal" looks like before the Residual
#     df['expected'] = df['sales'] - df['resid']
    
#     # 2. Calculate thresholds for both buckets to draw the limit lines
#     for is_special in [True, False]:
#         mask = (df['is_special'] == is_special)
#         if mask.any():
#             resids = df.loc[mask, 'resid']
#             median = resids.median()
#             mad = (resids - median).abs().median()
            
#             # Calculate the boundaries in units of 'Sales'
#             df.loc[mask, 'upper_limit'] = df['expected'] + (3 * 1.4826 * mad) + median
#             df.loc[mask, 'lower_limit'] = df['expected'] - (3 * 1.4826 * mad) + median

#     # 3. Plot Actuals vs Expected
#     plt.plot(df['week_start'], df['sales'], color='gray', alpha=0.3, label='Actual Sales')
#     plt.plot(df['week_start'], df['expected'], color='blue', label='Baseline (Trend+Season)', linestyle='--')

#     # 4. Plot the dynamic Outlier Boundaries
#     plt.step(df['week_start'], df['upper_limit'], color='red', alpha=0.4, label='Upper Bound', where='mid')
#     plt.step(df['week_start'], df['lower_limit'], color='orange', alpha=0.4, label='Lower Bound', where='mid')

#     # 5. Highlight Outliers
#     outliers = df[df['is_outlier'] == True]
#     plt.scatter(outliers['week_start'], outliers['sales'], color='red', marker='x', s=100, label='Outliers')

#     plt.title(f'Outlier Detection Bounds: {series_id}')
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.grid(True, alpha=0.2)
#     plt.show()





#. plot the outliers
#This summary report will help you scan all 100 series at once to see which products are the "troublemakers" (those with frequent stockouts or data errors).


def generate_outlier_summary(df):
    """
    Aggregates outlier counts across all series, 
    breaking them down by High-side vs Low-side.
    """
    summary = []
    
    for sid, group in df.groupby('series_id'):
        # Only look at rows marked as outliers
        outliers = group[group['is_outlier'] == True]
        
        # Determine if outlier was above or below the expected residual
        high_side = (outliers['resid'] > 0).sum()
        low_side = (outliers['resid'] < 0).sum()
        
        # Calculate percentage of weeks that are anomalous
        pct_impacted = (len(outliers) / len(group)) * 100
        
        summary.append({
            'series_id': sid,
            'total_outliers': len(outliers),
            'high_side_spikes': high_side,
            'low_side_dips': low_side,
            'impact_pct': f"{pct_impacted:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary).sort_values(by='total_outliers', ascending=False)
    
    print("--- Outlier Detection Summary Report ---")
    print(f"Total Series Processed: {df['series_id'].nunique()}")
    print(f"Total Outliers Found: {summary_df['total_outliers'].sum()}")
    print("-" * 40)
    return summary_df

# Usage
# summary_report = generate_outlier_summary(df_final)
# print(summary_report.head(10)) # Top 10 series with most anomalies

# print(df_final[df_final['series_id'] == 'Product_001'].head(20))

plot_outlier_results(df_final[df_final['series_id'] == 'Product_002'], 'Product_001')
# (df_final[df_final['series_id'] == 'Product_001']).to_csv('outlier_identification/product_001_ver2.csv', index=False)
# summary_report = generate_outlier_summary(df_final)
# print("summary report below:")
# print(summary_report.head(20))







# Currently, the code groups all special days (BOGO, Buy2Get2, Black Friday) into one single "Special" bucket. This means it calculates one median lift for everything.

# Since you are using Robust STL, the baseline for a BOGO week is actually computed using the data from the surrounding "normal" weeks, effectively ignoring the BOGO spike itself.
# Here is the step-by-step logic the algorithm follows:
# 1. It "Blurs" the Spike
# When the STL algorithm looks at a BOGO week, it sees a massive value. Because you set robust=True, the algorithm identifies this as an extreme deviation and assigns it a very low weight (close to zero).
# 2. It Interpolates from Neighbors
# Instead of letting that high BOGO number pull the baseline up, the algorithm looks at the weeks before and after the promotion. It "bridges" the gap between them to estimate what the sales would have been if the promotion hadn't happened.
# The Trend: Is calculated by looking at the long-term direction of the surrounding months.
# The Seasonality: Is calculated by looking at what happened in that same week (e.g., Week 20) in previous years.
# 3. The Resulting Baseline
# The baseline (trend + seasonal) for that BOGO week ends up representing your organic demand.
# 4. How the "Promo Logic" then applies
# In the function we wrote, we don't just use that organic baseline to find outliers. We do this:
# Step A: Calculate the Residual (Actual BOGO Sales minus the Organic Baseline).
# Step B: Group all those "BOGO Residuals" together and find their Median. This Median is the "Expected Promo Lift."
# Step C: Your Final Promo Limit = Organic Baseline + Median Promo Lift ± Threshold.
# In short: The baseline is computed by ignoring the promotion, but the outlier limit is then adjusted upward by the typical lift seen in other promotions.
# Does that clarify why the baseline stays low even when sales are high?




