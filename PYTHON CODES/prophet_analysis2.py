import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from prophet import Prophet

# Load CSV files into pandas DataFrames
file_paths = ['pellet0.csv', 'pellet1.csv', 'pellet2.csv', 'pellet35.csv', 'pellet66.csv', 'pellet67.csv']
dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Concatenate DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Prepare the features and target variable
X = df[['x', 'y', 'time']]
y = df['diameter']

# Train and predict with Prophet
prophet_df = pd.DataFrame()
prophet_df['ds'] = pd.to_datetime(df['time'].astype(str).apply(lambda x: x.zfill(2)), format='%M')
prophet_df['y'] = y

prophet_model = Prophet()
prophet_model.fit(prophet_df)
prophet_pred = prophet_model.predict(prophet_df)['yhat'].values

# Function to calculate the total number of pellets
def calculate_total_pellets(pred):
    return int(np.sum(pred))

# Calculate total pellets for Prophet
prophet_total_pellets = calculate_total_pellets(prophet_pred)

# Visualize the predicted values in a histogram with gaps between bars
plt.figure(figsize=(10, 6))
plt.hist(prophet_pred, bins='auto', color='brown', edgecolor='black', align='mid')
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Prophet - Predicted Size Distribution at t = 0, 5, 10, 15, 20, 25 seconds')
plt.text(10, prophet_total_pellets + 5, f'Total Pellets: {prophet_total_pellets}', fontsize=12, ha='left')
plt.grid(True)
plt.show()

# Scatter plot of diameter vs. time with number of pellets annotated
time_points_seconds = [0, 5, 10, 15, 20, 25, 30]
colors = plt.cm.tab10(np.linspace(0, 1, len(time_points_seconds)))

for i, t in enumerate(time_points_seconds):
    c = colors[i]
    plt.scatter(df[df['time'] == t]['diameter'], [t] * df[df['time'] == t].shape[0], c=c, label=f't = {t} seconds', marker='o', s=50)

    # Annotate each dot with the number of pellets
    for index, row in df[df['time'] == t].iterrows():
        plt.annotate(f"{int(row['diameter'])}", (row['diameter'], t), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.xlabel('Diameter')
plt.ylabel('Time (seconds)')
plt.title('Pellet Diameter at Different Time Points')
plt.legend()
plt.grid(True)
plt.show()
