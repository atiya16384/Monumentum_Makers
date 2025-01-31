import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = 'filtered_lower_discount_data.csv'
data = pd.read_csv(file_path)
data['date_of_order'] = pd.to_datetime(data['date_of_order'])

# Feature Engineering
data['discount_squared'] = data['discount'] ** 2
data['day_of_week'] = data['date_of_order'].dt.dayofweek
data['month'] = data['date_of_order'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

# Add rolling and cumulative features
data['rolling_orders_7'] = data.groupby('product_number')['orders'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean())
data['cumulative_orders'] = data.groupby('product_number')['orders'].cumsum()

# Clustering-Based Features
product_features = data.groupby('product_number').agg({
    'orders': 'mean',
    'discount': 'mean',
    'day_of_week': 'mean',
    'is_weekend': 'mean'
}).reset_index()

kmeans = KMeans(n_clusters=5, random_state=42)
product_features['cluster'] = kmeans.fit_predict(product_features[['orders', 'discount', 'day_of_week', 'is_weekend']])
data = data.merge(product_features[['product_number', 'cluster']], on='product_number', how='left')

# Encode categorical features
encoder = OneHotEncoder()
encoded_categories = encoder.fit_transform(data[['department_desc']]).toarray()
encoded_columns = encoder.get_feature_names_out(['department_desc'])
encoded_df = pd.DataFrame(encoded_categories, columns=encoded_columns)
data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

# Define Features and Target
X = data.drop(columns=['orders', 'date_of_order', 'department_desc', 'product_number'])
y = data['orders']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=0.01, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42),
    'Bayesian Ridge': BayesianRidge()
}

# Train and Evaluate Models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)  # Constrain predictions to non-negative
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, mae, r2))
    print(f"{name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R^2 = {r2:.4f}")


# Create a directory for future predictions
output_dir = 'future_predictions'
os.makedirs(output_dir, exist_ok=True)

# Future Data for All of 2023 Predictions
future_dates = pd.date_range(start='2023-03-01', end='2023-12-31', freq='D')
unique_products = data[['product_number', 'department_desc', 'cluster']].drop_duplicates()

# Create future dataset for each product
future_data = pd.DataFrame()
for _, row in unique_products.iterrows():
    temp_data = pd.DataFrame({
        'product_number': row['product_number'],
        'department_desc': row['department_desc'],
        'cluster': row['cluster'],
        'date_of_order': future_dates
    })
    future_data = pd.concat([future_data, temp_data], axis=0)

# Add features for future predictions
future_data['discount'] = 0  # Assuming no discounts
future_data['discount_squared'] = 0 ** 2
future_data['day_of_week'] = future_data['date_of_order'].dt.dayofweek
future_data['month'] = future_data['date_of_order'].dt.month
future_data['is_weekend'] = future_data['day_of_week'].isin([5, 6]).astype(int)
rolling_mean = data.groupby('product_number')['orders'].mean()
future_data['rolling_orders_7'] = future_data['product_number'].map(rolling_mean)
future_data['cumulative_orders'] = future_data['product_number'].map(rolling_mean.cumsum())

# Encode future data
future_encoded = pd.DataFrame(
    encoder.transform(future_data[['department_desc']]).toarray(),
    columns=encoded_columns
)
future_data = pd.concat([future_data.reset_index(drop=True), future_encoded], axis=1)

# Align future features with training features
future_data = future_data[X.columns]

# Standardize features
future_data_scaled = scaler.transform(future_data)

# Generate predictions for future dates
for name, model in models.items():
    future_predictions = model.predict(future_data_scaled)
    future_predictions = np.maximum(future_predictions, 0)  # Constrain predictions to non-negative
    future_predictions = np.round(future_predictions).astype(int)  # Round to nearest integer
    future_predictions_df = pd.DataFrame({
        'product_number': unique_products['product_number'].repeat(len(future_dates)).reset_index(drop=True),
        'department_desc': unique_products['department_desc'].repeat(len(future_dates)).reset_index(drop=True),
        'date_of_order': future_dates.tolist() * len(unique_products),
        'Predicted_Orders': future_predictions
    })
    output_file = os.path.join(output_dir, f'future_predictions_{name.lower().replace(" ", "_")}.csv')
    future_predictions_df.to_csv(output_file, index=False)

# Time Series Analysis with SARIMA
sarima_data = data.groupby('date_of_order')['orders'].sum()

# Fit SARIMA model
sarima_model = SARIMAX(sarima_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_results = sarima_model.fit(disp=False)

# Evaluate SARIMA
sarima_predictions = sarima_results.get_prediction(start=0, end=len(sarima_data)-1)
sarima_rmse = np.sqrt(mean_squared_error(sarima_data, sarima_predictions.predicted_mean))
sarima_mae = mean_absolute_error(sarima_data, sarima_predictions.predicted_mean)
sarima_r2 = r2_score(sarima_data, sarima_predictions.predicted_mean)
print(f"\nSARIMA Metrics: RMSE = {sarima_rmse:.4f}, MAE = {sarima_mae:.4f}, R² = {sarima_r2:.4f}")

# Forecast for the rest of 2023
sarima_forecast = sarima_results.get_forecast(steps=len(future_dates))
sarima_forecast_df = pd.DataFrame({
    'date_of_order': future_dates,
    'Predicted_Orders': np.round(sarima_forecast.predicted_mean).astype(int),  # Round to nearest integer
    'Lower_CI': sarima_forecast.conf_int().iloc[:, 0],
    'Upper_CI': sarima_forecast.conf_int().iloc[:, 1]
})

sarima_forecast_df.to_csv(os.path.join(output_dir, 'sarima_predictions.csv'), index=False)

# Model Training and Evaluation
results = []
feature_importances = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)  # Constrain predictions to non-negative
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, mae, r2))
     # Predicted vs. Actual Plot
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Predicted vs. Actual for {name}')
    # plt.savefig(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_predicted_vs_actual.png'))
    plt.show()  # Displays the plot
    plt.close()

        # Feature Importance for Tree-Based Models
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        importance.sort_values(by='Importance', ascending=False, inplace=True)
        plt.figure()
        importance.plot(kind='bar', x='Feature', y='Importance', legend=False)
        plt.title(f'Feature Importance for {name}')
        plt.ylabel('Importance')
        # plt.savefig(os.path.join(output_dir, f'{name.lower().replace(" ", "_")}_feature_importance.png'))
        plt.show()  # Displays the plot
        plt.close()

    

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R²"])

# SARIMA Predictions Plot
plt.figure(figsize=(14, 7))
plt.plot(sarima_data.index, sarima_data, label="Actual Orders", color="blue")
plt.plot(sarima_predictions.predicted_mean, label="SARIMA Predictions", color="orange")
plt.title("SARIMA Predictions vs Actual Orders")
plt.legend()
# plt.savefig("sarima_predictions_plot.png")
plt.show()

# AIC Plot
plt.figure()
plt.plot([sarima_data], marker='o', markersize=10, label='SARIMA AIC')
plt.title('AIC Values')
plt.ylabel('AIC')
plt.legend()
# plt.savefig(os.path.join(output_dir, 'sarima_aic.png'))
plt.close()
plt.show()  # Displays the plot

# SARIMA Forecast Plot with Confidence Intervals
plt.figure(figsize=(14, 7))
plt.plot(future_dates, sarima_forecast.predicted_mean, label="SARIMA Forecast", color="orange")
plt.fill_between(
    future_dates,
    sarima_forecast.conf_int().iloc[:, 0],
    sarima_forecast.conf_int().iloc[:, 1],
    color="orange",
    alpha=0.2,
    label="Confidence Interval",
)
plt.xlabel("Date")
plt.ylabel("Predicted Orders")
plt.title("SARIMA Forecast with Confidence Intervals")
plt.legend()
# plt.savefig("sarima_forecast_plot.png")
plt.show()

# Save predictions and results
results_df.to_csv("model_performance.csv", index=False)
sarima_forecast_df.to_csv(os.path.join(output_dir, "sarima_predictions.csv"), index=False)
