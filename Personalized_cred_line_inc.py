import pandas as pd

# === Step 1: Load the risk-categorized data and RAMS data ===
risk_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\code\Credit-Risk-Analysis\Q4_credit_risk_categories.csv")
rams_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\feature_selected\rams_batch_cur_20250325.csv")

# === Step 2: Merge to get current credit line (cu_crd_line) ===
rams_clean = rams_df.drop_duplicates(subset='cu_account_nbr')
merged_df = risk_df.merge(
    rams_clean[['cu_account_nbr', 'cu_crd_line']],
    left_on='current_account_nbr',
    right_on='cu_account_nbr',
    how='left'
)

# === Step 3: Define increase recommendation logic ===
def recommend_increase(row):
    category = row['credit_risk_category']
    credit_line = row['cu_crd_line']
    avg_spend = row['avg_q4_spend']

    if pd.isna(credit_line) or credit_line == 0:
        return "Data Missing"

    # Only suggest increase for Low and Very Low Risk
    if category == "Very Low Risk" or category == "Low Risk":
        spending_ratio = avg_spend / credit_line

        if spending_ratio > 0.9:
            factor = 0.30
        elif spending_ratio > 0.7:
            factor = 0.20
        elif spending_ratio > 0.5:
            factor = 0.10
        else:
            factor = 0.05

        return round(credit_line * factor, 2)

    elif category == "Watchlist":
        return "Banker Decision"
    else:  # High Risk
        return "No Increase"

# === Step 4: Apply the logic ===
merged_df['recommended_credit_line_increase'] = merged_df.apply(recommend_increase, axis=1)

# === Step 5: Save the result ===
output_cols = [
    'current_account_nbr', 'credit_risk_category', 'avg_q4_spend', 'cu_crd_line', 
    'recommended_credit_line_increase'
]
merged_df[output_cols].to_csv(r"C:\Users\uanus\Desktop\Datathon\code\Credit-Risk-Analysis\Credit_Line_Recommendations.csv", index=False)

print("Credit line increase recommendations saved to: Credit_Line_Recommendations.csv")
