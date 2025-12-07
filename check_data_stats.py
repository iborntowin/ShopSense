import pandas as pd
import numpy as np
import joblib

# Load actual training data to get REAL statistics
df = pd.read_csv('../shopping_trends_cleanfinal.csv')
df = df[df['Age'] != 2]

# Since data is already normalized, let's reverse engineer the original scales
# by looking at relationships or we can just work with normalized space

print("Data is already normalized in CSV:")
print(f"Age - mean: {df['Age'].mean():.6f}, std: {df['Age'].std():.6f}")
print(f"Income - mean: {df['Income'].mean():.6f}, std: {df['Income'].std():.6f}")
print(f"Previous Purchases - mean: {df['Previous Purchases'].mean():.6f}, std: {df['Previous Purchases'].std():.6f}")
print(f"Review Rating - mean: {df['Review Rating'].mean():.6f}, std: {df['Review Rating'].std():.6f}")
print(f"NumWebVisitsMonth - mean: {df['NumWebVisitsMonth'].mean():.6f}, std: {df['NumWebVisitsMonth'].std():.6f}")

# Let's check the engineered features that were created in training
df_check = df.copy()
df_check['Income_Per_1k_check'] = df_check['Income'] / 1000  # This won't work since Income is already scaled!

print("\nThe problem: Income is scaled, so Income/1000 is wrong!")
print("We need the ORIGINAL un-normalized statistics...")
