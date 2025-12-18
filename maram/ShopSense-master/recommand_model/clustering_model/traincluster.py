from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd
import pickle

# --- Load and preprocess data ---
df = pd.read_csv("customer_segmentation.csv")
df.dropna(subset=['Income'], inplace=True)

current_year = 2014
df['Age'] = current_year - df['Year_Birth']
df.drop('Year_Birth', axis=1, inplace=True)

features_to_drop = ['ID', 'Dt_Customer', 'Education', 'Marital_Status', 'Z_CostContact', 'Z_Revenue']
clustering_features = df.drop(features_to_drop, axis=1).copy()

# --- Scale data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_features)

# --- GMM with fixed 3 clusters ---
best_k = 3
best_gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42)
best_gmm.fit(X_scaled)
labels = best_gmm.predict(X_scaled)

# --- Save scaler and GMM ---
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(best_gmm, open("gmm.pkl", "wb"))

print(f"Model saved with {best_k} clusters.")
