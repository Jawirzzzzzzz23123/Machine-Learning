import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Setup direktori output
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# 2. Logging terminal ke file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(output_dir, 'logs.txt'))
sys.stderr = sys.stdout

# 3. Load dataset
data = pd.read_csv('AmesHousing.csv')
print("Dataset shape:", data.shape)

# 4. Data Understanding
print(data.info())
print(data.describe())

# 5. Data Cleaning
data.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], errors='ignore', inplace=True)

for col in data.select_dtypes(include=np.number).columns:
    data[col].fillna(data[col].median(), inplace=True)

for col in data.select_dtypes(include='object').columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Visualisasi Distribusi sebelum dan sesudah log
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(np.expm1(np.log1p(data['SalePrice'])), bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribusi Harga Rumah (Asli)')
axes[0].set_xlabel('SalePrice')

data['SalePrice'] = np.log1p(data['SalePrice'])
sns.histplot(data['SalePrice'], bins=50, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribusi Harga Rumah (Log-Transformed)')
axes[1].set_xlabel('Log(SalePrice)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribusi_saleprice_sebelum_sesudah_log.png'))
plt.show()

# Korelasi Numerik
numeric_corr = data.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False).drop('SalePrice').head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=numeric_corr.values, y=numeric_corr.index, palette='viridis')
plt.title('Top 10 Korelasi Fitur Numerik dengan SalePrice')
plt.xlabel('Korelasi')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'korelasi_top_fitur.png'))
plt.show()

# 6. Feature Engineering
data_encoded = pd.get_dummies(data, drop_first=True)
data_encoded.to_csv(os.path.join(output_dir, 'data_cleaned_encoded.csv'), index=False)
print("Data cleaned dan encoded telah disimpan.")

# 7. Split Data
X = data_encoded.drop('SalePrice', axis=1)
y = data_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Standarisasi untuk Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Modeling
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 10. Evaluasi
def evaluate(y_true, y_pred, model_name):
    print(f"=== {model_name} ===")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²:", r2_score(y_true, y_pred))
    print()

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("Random Forest Cross-Validation R²:", np.mean(cv_scores))

# 11. Visualisasi Prediksi vs Aktual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(np.expm1(y_test), np.expm1(y_pred_lr), alpha=0.5, color='green')
axes[0].set_title('Linear Regression: Prediksi vs Aktual')
axes[0].set_xlabel('Actual Sale Price')
axes[0].set_ylabel('Predicted Sale Price')

axes[1].scatter(np.expm1(y_test), np.expm1(y_pred_rf), alpha=0.5, color='blue')
axes[1].set_title('Random Forest: Prediksi vs Aktual')
axes[1].set_xlabel('Actual Sale Price')
axes[1].set_ylabel('Predicted Sale Price')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'perbandingan_model_prediksi_vs_aktual.png'))
plt.show()

# 12. Residual Plot
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(8, 6))
sns.histplot(residuals_lr, bins=50, kde=True, color='green')
plt.title("Distribusi Residual (Linear Regression)")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_lr.png'))
plt.show()

residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(8, 6))
sns.histplot(residuals_rf, bins=50, kde=True, color='blue')
plt.title("Distribusi Residual (Random Forest)")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_rf.png'))
plt.show()

# 13. Feature Importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
plt.show()

# 14. Simpan Model
joblib.dump(rf, os.path.join(output_dir, 'random_forest_model.pkl'))
joblib.dump(lr, os.path.join(output_dir, 'linear_regression_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print("Model dan scaler telah disimpan.")
print("\n✅ Selesai! Semua hasil dan file telah disimpan di:", output_dir)