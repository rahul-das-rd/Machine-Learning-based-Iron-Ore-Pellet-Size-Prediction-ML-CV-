import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

# Step 1: Load the CSV files into pandas DataFrames
file_paths = ['pellet0.csv', 'pellet1.csv', 'pellet2.csv', 'pellet35.csv', 'pellet66.csv', 'pellet67.csv']
dataframes = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Step 3: Combine the data
combined_df = pd.concat(dataframes, axis=0)

# Step 4: Prepare the features and target
X = combined_df['diameter'].values.reshape(-1, 1)

# Step 5: Calculate the average number of pellets observed in previous time points
previous_counts = [df.shape[0] for df in dataframes]
avg_pellets = int(np.mean(previous_counts))

# Step 6: Adjust the number of samples based on the average number of pellets
num_samples = int(avg_pellets * 1.2)  # Increase by 20% for variability

# Step 7: Fit the kernel density estimation model
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(X)

# Step 8: Generate samples from the estimated distribution
X_pred = kde.sample(num_samples).reshape(-1)

# Step 9: Visualize the predicted size distribution
plt.figure(figsize=(8, 6))
plt.hist(X_pred, bins='auto', edgecolor='black', linewidth=1.2)
plt.xlabel('Diameter')
plt.ylabel('Number of Pellets')
plt.title('Predicted Size Distribution at t = 30 seconds')

# Add text for total pellet count in the top-left corner
total_pellets = len(X_pred)
plt.text(0.02, 0.95, f'Total Pellets: {total_pellets}', transform=plt.gca().transAxes)

# Add text labels for number of pellets above each bar
bin_counts, bin_edges, _ = plt.hist(X_pred, bins='auto', edgecolor='black', linewidth=1.2)
for count, edge in zip(bin_counts, bin_edges):
    if count > 0:
        plt.text(edge + 0.5, count + 5, str(int(count)), ha='center')

plt.show()

# Display bar graphs for each input CSV file
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    X = df['diameter'].values

    plt.figure(figsize=(8, 6))
    plt.hist(X, bins='auto', edgecolor='black', linewidth=1.2)
    plt.xlabel('Diameter')
    plt.ylabel('Number of Pellets')
    plt.title(file_path)  # Use the file path as the title

    # Add text for total pellet count in the top-left corner
    total_pellets = len(X)
    plt.text(0.02, 0.95, f'Total Pellets: {total_pellets}', transform=plt.gca().transAxes)

    # Add text labels for number of pellets above each bar
    bin_counts, bin_edges, _ = plt.hist(X, bins='auto', edgecolor='black', linewidth=1.2)
    for count, edge in zip(bin_counts, bin_edges):
        if count > 0:
            plt.text(edge + 0.5, count + 5, str(int(count)), ha='center')

    plt.show()
