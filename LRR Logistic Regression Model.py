import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from scipy import stats
# 1. Data Loading and Initial Cleaning (IQR Filters)
df = pd.read_csv('single_family_loan_level_dataset_curated_2.0.csv')
df['interest_rate'] = df['interest_rate'].astype(str).str.replace(',', '.').astype(float)
median_cs = df[df['credit_score'] < 9999]['credit_score'].median()
df['credit_score'] = df['credit_score'].replace(9999, median_cs)

# --- 2. CATEGORIES DEFINITION (BASE PROFILE) ---

cat_occupancy_status = pd.CategoricalDtype(categories=['P', 'I', 'S'], ordered=True)
df['occupancy_status'] = df['occupancy_status'].astype(cat_occupancy_status)

cat_property_type = pd.CategoricalDtype(categories=['SF','CO', 'PU', 'MH', 'CP'], ordered=True)
df['property_type'] = df['property_type'].astype(cat_property_type)

cat_loan_purpose = pd.CategoricalDtype(categories=['P', 'C', 'N', 'R'], ordered=True)
df['loan_purpose'] = df['loan_purpose'].astype(cat_loan_purpose)

cat_region = pd.CategoricalDtype(categories=['South', 'Midwest', 'West', 'Northeast'], ordered=True)
df['region'] = df['region'].astype(cat_region)

df['is_30yr'] = (df['loan_term'] == 360).astype(int)

df['first_time_homebuyer'] = (df['first_time_homebuyer'] == 'Y').astype(int)


# --- 3. Create Dummies forcing INTEGER type ---
# dtype=int fixes the "Pandas data cast to numpy dtype of object" error
df_final = pd.get_dummies(df, columns=['occupancy_status'
                                       ], drop_first=True, dtype=int)

features = [
    'credit_score',
    'occupancy_status_I',
    'occupancy_status_S',
    'dti',
    'ltv',
    'interest_rate',
    'num_borrowers'
]

# Shuffle the dataset to prevent any systematic ordering bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_final[features]
X = sm.add_constant(X)
y = df_final['default']
print(df_final['default'].value_counts())
print(df_final['default'].mean())
logit_model = sm.Logit(y, X).fit(method='newton', max_iter=100)

print(logit_model.summary())

params = logit_model.params

# 5. Compute Odds Ratios (exponential of the beta coefficients)
odds_ratios = np.exp(params)

# 6. Compute confidence intervals for each coefficient
conf = logit_model.conf_int()
conf['Odds Ratio'] = odds_ratios
conf.columns = ['Lower CI', 'Upper CI', 'Odds Ratio']

print(conf)

params = logit_model.params

#Function to convert logit into probability
def logit_to_prob(logit_value):
    return 1 / (1 + np.exp(-logit_value))

# --- 7. REFINED: Exporting Reference Table for Power BI ---

# 1. Prepare lists for the DataFrame
# We initialize them with None or a default so they match the length of params.index
min_values = []
max_values = []
means_dict = X.mean().to_dict()
numeric_features = ['credit_score', 'dti', 'ltv', 'interest_rate', 'num_borrowers']

# 2. Iterate through the parameters index to ensure lengths match
for col in params.index:
    if col in numeric_features:
        # Calculate and store numeric limits
        c_min = df_final[col].min()
        c_max = df_final[col].max()
        min_values.append(c_min)
        max_values.append(c_max)
        print(f"Variable: {col:15} | Min: {c_min:8.2f} | Max: {c_max:8.2f}")
    else:
        # For 'const' or dummy variables (0/1), min/max are usually 0 and 1
        # We use None or 0/1 to keep the list length consistent
        min_values.append(df_final[col].min() if col in df_final.columns else 0)
        max_values.append(df_final[col].max() if col in df_final.columns else 1)

# 3. Create the reference DataFrame
df_model_ref = pd.DataFrame({
    'Variable': params.index,
    'Beta': params.values,
    'Mean_Value': [means_dict.get(col, 1.0) for col in params.index],
    'Min_Limit': min_values,
    'Max_Limit': max_values
})

# 4. Calculate the Base Logit and Probability for the "Average Borrower"
# sum(Beta * Mean)
avg_logit = (df_model_ref['Beta'] * df_model_ref['Mean_Value']).sum()
avg_prob = 1 / (1 + np.exp(-avg_logit))


print(f"\n--- Model Reference Generated ---")
print(f"The 'Average Borrower' has a predicted Default Probability of: {avg_prob:.2%}")

# 5. Export to CSV
df_model_ref.to_csv("model_parameters_reference.csv", index=False)
print("'model_parameters_reference.csv' saved successfully!")

#VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("--- Variance Inflation Factor (VIF) ---")
print(vif_data)

#MACHINE LEARNING

# Scale first, then split
X_features = df_final[features]  # without the constant, sklearn adds it automatically
y = df_final['default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Coefficients
# for i, name in enumerate(features):
#    print(f'{name:>25}: {model.coef_[0][i]:.4f}')

# Predictions on test set
predictions_test = model.predict(X_test_scaled)

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions_test))
print(classification_report(y_test, predictions_test))


# Predictions on full dataset
log_preds = model.predict(X_scaled)
df_final['predicted_default'] = log_preds
df_final['prob_score'] = logit_model.predict(X)
print(df_final[['loan_id', 'default', 'predicted_default', 'prob_score']].head(10))
df_final.to_csv('single_family_loan_level_dataset_curated_predictions.csv',
                index=False, encoding='utf-8-sig', sep=',', decimal='.')
print("File saved! Ready for Power BI.")
