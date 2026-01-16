#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Load the customer data
data = pd.read_csv("/Users/ranwalagekasun/Desktop/IBS/Sem 5/Data Science/Data_Task_2/EU-Park-Customers.csv")


# One-hot encoding for categorical variables (e.g., Club_Member), converted boolean into an integer value.
data_encoded = pd.get_dummies(data, columns=['Club_Member'], drop_first=True)
data_encoded['Club_Member'] = data_encoded['Club_Member'].astype(int)

# Remove empty 'Pass_Type' rows
data_encoded = data_encoded.dropna(subset=['Pass_Type'])

# Prepare features (X) and target (y)
# Predict "Pass_Type" based on other attributes, Drop Telephone_Number as it's an identifier, not a predictor
X = data_encoded.drop(['Pass_Type', 'Telephone_Number'], axis=1)
y = data_encoded['Pass_Type']

# Encode Target 'Pass_Type'
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# mapping: 0: Gold, 1: Silver (memberships)(usually alphabetical)
print("Classes mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# Split data: 67% training, 33% testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.33, random_state=42)

# Train XGBoost
# This is XGBoost’s classification model. It uses gradient-boosted decision trees
# Setting this to False tells XGBoost: “I already encoded my labels (using LabelEncoder)”
# mlogloss = multiclass log loss. Measures how confident and accurate the predictions are
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix: Predicted vs Actual Pass Type')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion Matrix:")
plt.show()

#classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feat_imp_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('XGBoost Feature Importance for Season Pass Purchase')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')