import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Load CSV files into pandas DataFrames
file_paths = ['pellet0.csv', 'pellet1.csv', 'pellet2.csv', 'pellet35.csv', 'pellet66.csv', 'pellet67.csv']
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Concatenate DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Filter DataFrame for t = 0, 5, 10, 15, 20, 25 seconds
time_points = [0, 5, 10, 15, 20, 25]
df_filtered = df[df['time'].isin(time_points)]

# Prepare the features and target variable
X = df_filtered[['x', 'y', 'time']]
y = df_filtered['diameter']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict with Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)

# Train and predict with Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)

# Train and predict with Support Vector Regressor
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train, y_train)
svr_pred = svr_reg.predict(X_test)

# Train and predict with SARIMAX
sarimax_model = SARIMAX(y_train, order=(1, 0, 0), seasonal_order=(0, 1, 1, 6))
sarimax_model_fit = sarimax_model.fit()
sarimax_pred = sarimax_model_fit.get_forecast(steps=len(X_test)).predicted_mean

# Train and predict with ARIMA
arima_model = ARIMA(y_train, order=(1, 0, 0))
arima_model_fit = arima_model.fit()
arima_pred = arima_model_fit.get_forecast(steps=len(X_test)).predicted_mean

# Train and predict with Prophet
prophet_df = pd.DataFrame()
prophet_df['ds'] = pd.to_datetime(df_filtered['time'].astype(str).apply(lambda x: x.zfill(2)), format='%M')
prophet_df['y'] = y

prophet_model = Prophet()
prophet_model.fit(prophet_df)
prophet_pred = prophet_model.predict(prophet_df)['yhat'].values

# Train and predict with Logistic Regression
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, y_train)
logistic_pred = logistic_reg.predict(X_test)

# Train and predict with Gaussian Process Regression (GPR)
gpr_reg = GaussianProcessRegressor(kernel=RBF())
gpr_reg.fit(X_train, y_train)
gpr_pred, gpr_pred_std = gpr_reg.predict(X_test, return_std=True)

# Function to calculate the total number of pellets
def calculate_total_pellets(pred):
    return int(np.sum(pred))

# Evaluate the models using mean squared error
linear_mse = mean_squared_error(y_test, linear_pred)
svr_mse = mean_squared_error(y_test, svr_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
sarimax_mse = mean_squared_error(y_test, sarimax_pred)
arima_mse = mean_squared_error(y_test, arima_pred)
prophet_mse = mean_squared_error(y, prophet_pred)
logistic_mse = mean_squared_error(y_test, logistic_pred)
gpr_mse = mean_squared_error(y_test, gpr_pred)

# Calculate total pellets for each model
linear_total_pellets = calculate_total_pellets(linear_pred)
svr_total_pellets = calculate_total_pellets(svr_pred)
rf_total_pellets = calculate_total_pellets(rf_pred)
sarimax_total_pellets = calculate_total_pellets(sarimax_pred)
arima_total_pellets = calculate_total_pellets(arima_pred)
prophet_total_pellets = calculate_total_pellets(prophet_pred)
logistic_total_pellets = calculate_total_pellets(logistic_pred)
gpr_total_pellets = calculate_total_pellets(gpr_pred)

# Visualize the predicted values in separate windows
# Linear Regression
plt.figure(figsize=(8, 6))
sns.histplot(linear_pred, bins='auto', kde=True, color='blue')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Linear Regression - Predicted Size Distribution at t = 30 seconds')
plt.text(0, linear_total_pellets + 5, f'Total Pellets: {linear_total_pellets}', fontsize=10, ha='left')
plt.show()

# Random Forest Regressor
plt.figure(figsize=(8, 6))
sns.histplot(rf_pred, bins='auto', kde=True, color='green')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Random Forest Regressor - Predicted Size Distribution at t = 30 seconds')
plt.text(0, rf_total_pellets + 5, f'Total Pellets: {rf_total_pellets}', fontsize=10, ha='left')
plt.show()

# Support Vector Regressor
plt.figure(figsize=(8, 6))
sns.histplot(svr_pred, bins='auto', kde=True, color='red')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Support Vector Regressor - Predicted Size Distribution at t = 30 seconds')
plt.text(0, svr_total_pellets + 5, f'Total Pellets: {svr_total_pellets}', fontsize=10, ha='left')
plt.show()

# SARIMAX
plt.figure(figsize=(8, 6))
sns.histplot(sarimax_pred, bins='auto', kde=True, color='purple')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('SARIMAX - Predicted Size Distribution at t = 30 seconds')
plt.text(0, sarimax_total_pellets + 5, f'Total Pellets: {sarimax_total_pellets}', fontsize=10, ha='left')
plt.show()

# ARIMA
plt.figure(figsize=(8, 6))
sns.histplot(arima_pred, bins='auto', kde=True, color='orange')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('ARIMA - Predicted Size Distribution at t = 30 seconds')
plt.text(0, arima_total_pellets + 5, f'Total Pellets: {arima_total_pellets}', fontsize=10, ha='left')
plt.show()

# Prophet
plt.figure(figsize=(8, 6))
sns.histplot(prophet_pred, bins='auto', kde=True, color='brown')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Prophet - Predicted Size Distribution at t = 30 seconds')
plt.text(0, prophet_total_pellets + 5, f'Total Pellets: {prophet_total_pellets}', fontsize=10, ha='left')
plt.show()

# Logistic Regression
plt.figure(figsize=(8, 6))
sns.histplot(logistic_pred, bins='auto', kde=True, color='cyan')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Logistic Regression - Predicted Size Distribution at t = 30 seconds')
plt.text(0, logistic_total_pellets + 5, f'Total Pellets: {logistic_total_pellets}', fontsize=10, ha='left')
plt.show()

# Gaussian Process Regression (GPR)
plt.figure(figsize=(8, 6))
sns.histplot(gpr_pred, bins='auto', kde=True, color='magenta')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Gaussian Process Regression - Predicted Size Distribution at t = 30 seconds')
plt.text(0, gpr_total_pellets + 5, f'Total Pellets: {gpr_total_pellets}', fontsize=10, ha='left')
plt.show()

# Model Performance Comparison
models = ['Linear Regression', 'SVR', 'Random Forest', 'SARIMAX', 'ARIMA', 'Prophet', 'Logistic Regression', 'GPR']
mse_scores = [linear_mse, svr_mse, rf_mse, sarimax_mse, arima_mse, prophet_mse, logistic_mse, gpr_mse]

plt.figure(figsize=(10, 6))
plt.bar(models, mse_scores)
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Model Performance')
plt.show()