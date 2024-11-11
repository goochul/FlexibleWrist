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

    # Detect loading and unloading parts
    # Loading phase (Z Position increases)
    loading_data = merged_data[merged_data["Z Position"].diff().fillna(0) >= 0]

    # Unloading phase (Z Position decreases)
    unloading_data = merged_data[merged_data["Z Position"].diff().fillna(0) < 0]

    # Plotting Force Magnitude vs Z Position without connecting loading and unloading
    plt.figure(figsize=(10, 6))

    # Plot loading part with scatter for individual points
    plt.scatter(loading_data["Z Position"], loading_data["Force Magnitude"], label="Loading", color='r')

    # Plot unloading part with scatter for individual points
    plt.scatter(unloading_data["Z Position"], unloading_data["Force Magnitude"], label="Unloading", color='b')

    # Label and title the plot
    plt.xlabel("Z Position (m)")
    plt.ylabel("Force Magnitude (N)")
    plt.title("Stiffness on Pressing Test (Loading and Unloading)")
    plt.legend()
    plt.grid(True)
    plt.show()
