# %% 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% 2. Data Preparation
# Objective: Load data and clean it to reflect actual park operations.

df_park = pd.read_csv("EU-park.csv")
print("Original Data Shape:", df_park.shape)

# --- DATA CLEANING & BUSINESS LOGIC ---
# Observation: The data contains constant "5 minute" wait times during the night.
# Assumption: These are default system values when the park is closed.
# Action: I am filtering the data to keep only active operating hours (08:00 to 21:00).
# This ensures my model predicts actual crowd behavior, not system defaults.
df_clean = df_park[(df_park['Hour'] >= 8) & (df_park['Hour'] <= 21)].copy()

# Handling Outliers:
# I found specific error codes (negative values) and impossible outliers (> 300 mins).
# Removing these is critical to keeping the RMSE (error metric) realistic.
df_clean = df_clean[(df_clean['WaitTime'] >= 0) & (df_clean['WaitTime'] < 300)]

# Handle Missing Values
if df_clean.isnull().sum().sum() > 0:
    print("Dropping missing values...")
    df_clean.dropna(inplace=True)

# %% 2.3 Feature Engineering
# Converting categorical data into numbers for the machine learning model.

# 'Rain' is boolean (True/False) -> Convert to 0/1
df_clean['Rain'] = df_clean['Rain'].astype(int)

# One-Hot Encoding:
# Season, DayOfWeek, and Attraction are nominal categories.
# I am converting them to dummy variables so the model can weight them individually.
df_model = pd.get_dummies(df_clean, columns=['Season', 'DayOfWeek', 'Attraction'], drop_first=False)

# Dropping 'Date' and 'Month' because 'Season' and 'DayOfWeek' capture the necessary cyclical patterns.
df_model = df_model.drop(columns=['Date', 'Month'])

print("Final Cleaned Data Shape:", df_model.shape)

# %% 3. Modeling (Prediction)

# Define features (X) and target (y)
X = df_model.drop('WaitTime', axis=1)
y = df_model['WaitTime']

# Splitting data: 80% for training the model, 20% for testing it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL SELECTION: RANDOM FOREST ---
# Decision: I chose Random Forest over Linear Regression.
# Reasoning: Wait times are non-linear. Crowds peak in the afternoon and drop in the evening.
# A linear model cannot capture this "curve" well, but decision trees handle it perfectly.
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training complete.")

# %% 4. Validation

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f} minutes (Penalty for large outliers)")
print(f"MAE:  {mae:.2f} minutes (Average error per prediction)")
print(f"Avg Wait Time in Data: {y.mean():.2f} minutes")

# Sanity Check
# If RMSE is much higher than MAE, it means we still have outliers.
# If RMSE is close to MAE and lower than standard deviation, the model is stable.
if rmse < y.std():
    print("Result: Prediction is reasonable (Error is lower than standard deviation).")
else:
    print("Result: Model needs improvement.")

# %% 5. Interpretation (Feature Importance)

# Extracting which variables drove the decisions in the Random Forest
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'].tail(10), feature_importance_df['Importance'].tail(10), color='skyblue')
plt.title('Top 10 Drivers of Wait Times')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Summary Output for the Report
print("\n--- Key Drivers (Feature Importance) ---")
top_5 = feature_importance_df.sort_values(by='Importance', ascending=False).head(5)
for i, row in top_5.iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")
    
# %% 6. Accuracy Calculation
avg_wait = y.mean()
accuracy = 100 - (mae / avg_wait * 100)

print("-" * 30)
print(f"Final Model Accuracy: {accuracy:.2f}%")
print("-" * 30)