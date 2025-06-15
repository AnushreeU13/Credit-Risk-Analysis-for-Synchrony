import pandas as pd
import numpy as np

# Step 1: Load files 
predicted_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\code\Credit-Risk-Analysis\Q4_2025_weekly_predictions.csv")
rams_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\feature_selected\rams_batch_cur_20250325.csv")

# Step 2: Calculate avg and std dev of Q4 predicted spending 
weekly_cols = predicted_df.columns[1:]
predicted_df['avg_q4_spend'] = predicted_df[weekly_cols].mean(axis=1)
predicted_df['std_q4_spend'] = predicted_df[weekly_cols].std(axis=1)

# Step 3: Ensure RAMS data has unique rows per account 
rams_clean = rams_df.drop_duplicates(subset='cu_account_nbr')

#  Step 4: Merge with cleaned RAMS features
merged_df = predicted_df.merge(
    rams_clean[['cu_account_nbr', 'cu_crd_bureau_scr', 'ca_avg_utilz_lst_6_mnths', 'ca_max_dlq_lst_6_mnths']],
    left_on='current_account_nbr',
    right_on='cu_account_nbr',
    how='left'
)

# Step 5: Define classification logic 
def classify_risk(row):
    score = row['cu_crd_bureau_scr']
    util = row['ca_avg_utilz_lst_6_mnths']
    delinquency = row['ca_max_dlq_lst_6_mnths']
    std_spend = row['std_q4_spend']

    if (score < 600) or (util > 0.60) or (delinquency >= 2) or (std_spend > 300):
        return 'High Risk'
    elif (600 <= score < 650) or (0.45 < util <= 0.60) or (200 < std_spend <= 300 and delinquency <= 1):
        return 'Watchlist'
    elif (650 <= score < 720) and (util <= 0.45) and (std_spend <= 200) and (delinquency == 0):
        return 'Low Risk'
    elif (score >= 720) and (util < 0.25) and (std_spend < 150) and (delinquency == 0):
        return 'Very Low Risk'
    else:
        return 'Watchlist'

# Step 6: Apply classifier 
merged_df['credit_risk_category'] = merged_df.apply(classify_risk, axis=1)

#Step 7: Save output 
output_columns = [
    'current_account_nbr', 'avg_q4_spend', 'std_q4_spend',
    'cu_crd_bureau_scr', 'ca_avg_utilz_lst_6_mnths', 'ca_max_dlq_lst_6_mnths',
    'credit_risk_category'
]
merged_df[output_columns].to_csv(r"C:\Users\uanus\Desktop\Datathon\code\Credit-Risk-Analysis\Q4_credit_risk_categories.csv", index=False)

print("Risk classification saved successfully to: Q4_credit_risk_categories.csv")
