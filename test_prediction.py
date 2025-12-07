import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('best_customer_classification_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
feature_info = joblib.load('feature_info.pkl')

print("Feature columns:", feature_info['feature_cols'])
print("\nNumber of features:", len(feature_info['feature_cols']))

# Test case: Low value customer (income $1000, 1 purchase, rating 1, 1 visit)
age = 35
income = 1000
previous_purchases = 1
review_rating = 1.0
num_web_visits = 1

# Categorical
gender = 1  # Male
category = 0  # Clothing
subscription = 0  # No
discount = 0  # No
promo = 0  # No

# Engineered features
income_per_1k = income / 1000
loyalty_score = previous_purchases * review_rating
engagement_level = num_web_visits * review_rating
price_sensitive = 0

# One-hot encoded
shipping_express = 0
shipping_free = 0
shipping_nextday = 0
shipping_standard = 1
shipping_pickup = 0

payment_cash = 0
payment_credit = 1
payment_debit = 0
payment_paypal = 0
payment_venmo = 0

freq_biweekly = 1
freq_every3 = 0
freq_fortnightly = 0
freq_monthly = 0
freq_quarterly = 0
freq_weekly = 0

# Create feature array
features = np.array([[
    age, income_per_1k, previous_purchases, review_rating, num_web_visits,
    subscription, gender, category, discount, promo,
    loyalty_score, engagement_level, price_sensitive,
    shipping_express, shipping_free, shipping_nextday, shipping_standard, shipping_pickup,
    payment_cash, payment_credit, payment_debit, payment_paypal, payment_venmo,
    freq_biweekly, freq_every3, freq_fortnightly, freq_monthly, freq_quarterly, freq_weekly
]])

print("\n\nInput features:")
for i, col in enumerate(feature_info['feature_cols']):
    print(f"{col}: {features[0][i]}")

# Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0]

print(f"\n\nPrediction: {prediction}")
print(f"Probability [Low Value, High Value]: {probability}")
print(f"High Value Probability: {probability[1]*100:.1f}%")

# Now test a high value customer
print("\n" + "="*80)
print("HIGH VALUE CUSTOMER TEST")
print("="*80)

age = 35
income = 100000
previous_purchases = 50
review_rating = 5.0
num_web_visits = 20

income_per_1k = income / 1000
loyalty_score = previous_purchases * review_rating
engagement_level = num_web_visits * review_rating
price_sensitive = 0
subscription = 1  # Yes

features2 = np.array([[
    age, income_per_1k, previous_purchases, review_rating, num_web_visits,
    subscription, gender, category, discount, promo,
    loyalty_score, engagement_level, price_sensitive,
    shipping_express, shipping_free, shipping_nextday, shipping_standard, shipping_pickup,
    payment_cash, payment_credit, payment_debit, payment_paypal, payment_venmo,
    freq_biweekly, freq_every3, freq_fortnightly, freq_monthly, freq_quarterly, freq_weekly
]])

features_scaled2 = scaler.transform(features2)
prediction2 = model.predict(features_scaled2)[0]
probability2 = model.predict_proba(features_scaled2)[0]

print(f"\nPrediction: {prediction2}")
print(f"Probability [Low Value, High Value]: {probability2}")
print(f"High Value Probability: {probability2[1]*100:.1f}%")
