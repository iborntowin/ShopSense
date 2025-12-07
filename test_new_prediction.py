import pandas as pd
import numpy as np
import joblib

# Load retrained model
model = joblib.load('best_customer_classification_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Test LOW value customer: income $1000, 1 purchase, rating 1, 1 visit
age_raw = 35
income_raw = 1000
previous_purchases_raw = 1
review_rating_raw = 1.0
num_web_visits_raw = 1

# Normalize using same scales as app.py
age = (age_raw - 42) / 15
income_normalized = (income_raw - 60000) / 30000
previous_purchases = (previous_purchases_raw - 25) / 12
review_rating = (review_rating_raw - 3.75) / 0.7
num_web_visits = (num_web_visits_raw - 5) / 2.5

# Categorical
gender = 1  # Male
category = 0  # Clothing
subscription = 0  # No
discount = 0
promo = 0

# Engineered
income_per_1k = income_normalized
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

features = np.array([[
    age, income_per_1k, previous_purchases, review_rating, num_web_visits,
    subscription, gender, category, discount, promo,
    loyalty_score, engagement_level, price_sensitive,
    shipping_express, shipping_free, shipping_nextday, shipping_standard, shipping_pickup,
    payment_cash, payment_credit, payment_debit, payment_paypal, payment_venmo,
    freq_biweekly, freq_every3, freq_fortnightly, freq_monthly, freq_quarterly, freq_weekly
]])

print("="*80)
print("LOW VALUE CUSTOMER TEST")
print("="*80)
print(f"Raw inputs: Age={age_raw}, Income=${income_raw}, Purchases={previous_purchases_raw}, Rating={review_rating_raw}, Visits={num_web_visits_raw}")
print(f"\nNormalized features:")
print(f"  Age: {age:.3f}")
print(f"  Income: {income_normalized:.3f}")
print(f"  Previous Purchases: {previous_purchases:.3f}")
print(f"  Review Rating: {review_rating:.3f}")
print(f"  Web Visits: {num_web_visits:.3f}")
print(f"  Loyalty Score: {loyalty_score:.3f}")
print(f"  Engagement Level: {engagement_level:.3f}")

features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
probability = model.predict_proba(features_scaled)[0]

print(f"\nPrediction: {prediction} ({'High Value' if prediction == 1 else 'Low Value'})")
print(f"Probability: {probability[1]*100:.1f}% High Value")

# Test HIGH value customer
print("\n" + "="*80)
print("HIGH VALUE CUSTOMER TEST")
print("="*80)

age_raw = 40
income_raw = 120000
previous_purchases_raw = 50
review_rating_raw = 5.0
num_web_visits_raw = 15
subscription = 1

age = (age_raw - 42) / 15
income_normalized = (income_raw - 60000) / 30000
previous_purchases = (previous_purchases_raw - 25) / 12
review_rating = (review_rating_raw - 3.75) / 0.7
num_web_visits = (num_web_visits_raw - 5) / 2.5

income_per_1k = income_normalized
loyalty_score = previous_purchases * review_rating
engagement_level = num_web_visits * review_rating

features2 = np.array([[
    age, income_per_1k, previous_purchases, review_rating, num_web_visits,
    subscription, gender, category, discount, promo,
    loyalty_score, engagement_level, price_sensitive,
    shipping_express, shipping_free, shipping_nextday, shipping_standard, shipping_pickup,
    payment_cash, payment_credit, payment_debit, payment_paypal, payment_venmo,
    freq_biweekly, freq_every3, freq_fortnightly, freq_monthly, freq_quarterly, freq_weekly
]])

print(f"Raw inputs: Age={age_raw}, Income=${income_raw}, Purchases={previous_purchases_raw}, Rating={review_rating_raw}, Visits={num_web_visits_raw}, Subscription=Yes")
print(f"\nNormalized features:")
print(f"  Age: {age:.3f}")
print(f"  Income: {income_normalized:.3f}")
print(f"  Previous Purchases: {previous_purchases:.3f}")
print(f"  Review Rating: {review_rating:.3f}")
print(f"  Web Visits: {num_web_visits:.3f}")
print(f"  Loyalty Score: {loyalty_score:.3f}")
print(f"  Engagement Level: {engagement_level:.3f}")

features_scaled2 = scaler.transform(features2)
prediction2 = model.predict(features_scaled2)[0]
probability2 = model.predict_proba(features_scaled2)[0]

print(f"\nPrediction: {prediction2} ({'High Value' if prediction2 == 1 else 'Low Value'})")
print(f"Probability: {probability2[1]*100:.1f}% High Value")
