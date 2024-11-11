import pandas as pd
import matplotlib.pyplot as plt

# Load the data
force_data = pd.read_csv('data/20241110/Step3_30mm/force_data.csv')
z_position_data = pd.read_csv('data/20241110/Step3_30mm/z_position_data.csv')

# Round timestamps to align data and facilitate merging
force_data['Timestamp'] = force_data['Timestamp'].round(2)
z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)

# Merge data on the rounded Timestamp column
merged_data = pd.merge(force_data, z_position_data, on="Timestamp", suffixes=('_force', '_z'))

# Check if the merge was successful
if merged_data.empty:
    print("Merged data is empty. Check timestamp alignment or precision.")
else:
    # Multiply Z Position by -1 to make it positive
    merged_data["Z Position"] = merged_data["Z Position"] * -1

    # Apply force threshold to detect the beginning of loading
    force_threshold = 10  # Threshold in Newtons
    above_threshold = merged_data[merged_data['Force Magnitude'] > force_threshold]

    if above_threshold.empty:
        print("No data points exceed the force threshold. Check the threshold value or data.")
    else:
        # Find the index where force first exceeds the threshold
        threshold_index = above_threshold.index[0]

        # Track direction changes in Z Position after reaching the force threshold
        merged_data['Direction'] = merged_data['Z Position'].diff().fillna(0)
        change_index = merged_data[(merged_data.index > threshold_index) & (merged_data['Direction'] < 0)].index[0]

        # Split data into loading and unloading based on the threshold and direction change
        loading_data = merged_data.loc[:change_index]
        unloading_data = merged_data.loc[change_index + 1:]

        # Plotting Force Magnitude vs Z Position for loading and unloading phases
        plt.figure(figsize=(10, 6))

        # Plot loading and unloading with different colors and avoid connecting lines between them
        plt.plot(loading_data["Z Position"], loading_data["Force Magnitude"], label="Loading", color='red', marker='o', linestyle='-')
        plt.plot(unloading_data["Z Position"], unloading_data["Force Magnitude"], label="Unloading", color='blue', marker='o', linestyle='-')

        # Label and title the plot
        plt.xlabel("Z Position (m)")
        plt.ylabel("Force Magnitude (N)")
        plt.title("Sitffness on the Pressing Test (Loading and Unloading Phases)")
        plt.legend()
        plt.grid(True)
        plt.show()
