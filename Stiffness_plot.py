import pandas as pd
import matplotlib.pyplot as plt

# Load the data
force_data = pd.read_csv('data/20241110/Step3_30mm/force_data.csv')
z_position_data = pd.read_csv('data/20241110/Step3_30mm/z_position_data.csv')

# Print the number of data points in each DataFrame
print(f"Number of data points in force_data: {len(force_data)}")
print(f"Number of data points in z_position_data: {len(z_position_data)}")

# Round timestamps to align data and facilitate merging
force_data['Timestamp'] = force_data['Timestamp'].round(2)
z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)

# Merge data on the rounded Timestamp column
merged_data = pd.merge(force_data, z_position_data, on="Timestamp", suffixes=('_force', '_z'), how='outer')

# Interpolate missing data points
merged_data = merged_data.sort_values(by='Timestamp').interpolate(method='linear')

# Check if the merge was successful
if merged_data.empty:
    print("Merged data is empty. Check timestamp alignment or precision.")
else:
    # Multiply Z Position by -1 to make it positive
    merged_data["Z Position"] = merged_data["Z Position"] * -1

    # Remove values before Z Position reaches 0.003
    filtered_data = merged_data[merged_data['Z Position'] >= 0.0025].copy()

    # Shift all Z Position values to the left
    z_position_shift = filtered_data['Z Position'].min()
    filtered_data['Z Position'] = filtered_data['Z Position'] - z_position_shift

    # Apply force threshold to detect the beginning of significant loading
    force_threshold = 10  # Threshold in Newtons
    above_threshold = filtered_data[filtered_data['Force Magnitude'] > force_threshold]

    if above_threshold.empty:
        print("No data points exceed the force threshold. Check the threshold value or data.")
    else:
        # Find the index where force first exceeds the threshold
        threshold_index = above_threshold.index[0]
        print(f"Threshold index: {threshold_index}")

        # Track direction changes in Z Position after reaching the force threshold
        filtered_data['Direction'] = filtered_data['Z Position'].diff().fillna(0)
        change_index = filtered_data[(filtered_data.index > threshold_index) & (filtered_data['Direction'] < 0)].index[0]
        print(f"Change index: {change_index}")

        # Split data into loading and unloading based on the threshold and direction change
        loading_data = filtered_data.loc[:change_index]
        unloading_data = filtered_data.loc[change_index + 1:]

        print(f"Loading data points: {len(loading_data)}")
        print(f"Unloading data points: {len(unloading_data)}")

        # Plotting Force Magnitude vs Z Position for loading and unloading phases
        plt.figure(figsize=(10, 6))

        # Plot loading and unloading with different colors and avoid connecting lines between them
        plt.plot(loading_data["Z Position"], loading_data["Force Magnitude"], label="Loading", color='red', marker='o', linestyle='-', markersize=2)
        plt.plot(unloading_data["Z Position"], unloading_data["Force Magnitude"], label="Unloading", color='blue', marker='o', linestyle='-', markersize=2)

        # Label and title the plot
        plt.xlabel("Z Position (m)")
        plt.ylabel("Force Magnitude (N)")
        plt.title("Stiffness on the Pressing Test (Loading and Unloading Phases)")
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        plt.savefig('data/20241110/Step3_30mm/stiffness_plot.png')
        
        # Show the plot
        plt.show()