import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load input data 
account_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\feature_selected\Account_dim.csv")
trans_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\feature_selected\transaction_fact_20250325.csv", parse_dates=['transaction_date'])
wrld_df = pd.read_csv(r"C:\Users\uanus\Desktop\Datathon\feature_selected\wrld_stor_tran_fact_20250325.csv", parse_dates=['transaction_date'])

# Step 2: Combine and filter transactions 
combined_trans_df = pd.concat([trans_df, wrld_df])
mask = (combined_trans_df['transaction_date'] >= '2024-07-25') & (combined_trans_df['transaction_date'] <= '2025-03-31')
filtered_df = combined_trans_df[mask].copy()
filtered_df['month'] = filtered_df['transaction_date'].dt.to_period('M')

# Step 3: Aggregate monthly spend per account 
monthly_spend = filtered_df.groupby(['current_account_nbr', 'month'])['transaction_amt'].sum().reset_index()
pivot_df = monthly_spend.pivot(index='current_account_nbr', columns='month', values='transaction_amt').fillna(0)

# Step 4: Train Random Forest model using last known months 
if pivot_df.shape[1] >= 9:
    X = pivot_df.iloc[:, :-4]
    y = pivot_df.iloc[:, -4:]
else:
    raise ValueError("Insufficient historical months for prediction.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(
    n_estimators=2000,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)

"""
Model Evaluation Metrics (on 20% test set):
  MAE   : 875.42
  RMSE  : 2043.92
  R²    : 0.52
"""

#Step 4.5: Evaluate model performance 
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics (on 20% test set):")
print(f"  MAE   : {mae:.2f}")
print(f"  RMSE  : {rmse:.2f}")
print(f"  R²    : {r2:.2f}")

# Step 5: Predict for all accounts from Account_dim 
account_ids = account_df['current_account_nbr']
X_full = pivot_df.reindex(account_ids).fillna(0)
X_full = X_full[X_train.columns]  # match features
q4_predictions = model.predict(X_full)

# Step 6: Weekly date generation for Q4 2025 
weekly_dates = pd.date_range(start='2025-09-01', end='2025-12-31', freq='7D')

# Step 7: Map weekly dates to months 
weekly_distribution = {
    '2025-09': [d for d in weekly_dates if d.month == 9],
    '2025-10': [d for d in weekly_dates if d.month == 10],
    '2025-11': [d for d in weekly_dates if d.month == 11],
    '2025-12': [d for d in weekly_dates if d.month == 12]
}
q4_months = ['2025-09', '2025-10', '2025-11', '2025-12']

# Step 8: Build wide-format prediction DataFrame 
final_data = []
for acc_id, month_preds in zip(account_ids, q4_predictions):
    row = [acc_id]
    for month, monthly_total in zip(q4_months, month_preds):
        weeks = weekly_distribution[month]
        weekly_val = monthly_total / len(weeks) if weeks else 0
        row.extend([round(weekly_val, 2)] * len(weeks))
    final_data.append(row)

# Step 9: Assign column headers 
columns = ['current_account_nbr'] + [d.strftime('%Y-%m-%d') for d in weekly_dates]
weekly_pred_df = pd.DataFrame(final_data, columns=columns)

# Step 10: Export 
weekly_pred_df.to_csv("Q4_2025_weekly_predictions.csv", index=False)
