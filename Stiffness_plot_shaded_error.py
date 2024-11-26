import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# List of directories to process
file_paths = [
    # Path('data/20241118/0.01_1'),
    # Path('data/20241118/0.01_2'),
    # Path('data/20241118/0.01_3'),
    # Path('data/20241118/0.001_2'),
    # Path('data/20241118/0.001_3')
    
    # Path('data/20241124/174523/'),
    # Path('data/20241124/174653/'),
    # Path('data/20241124/181512/'),
    # Path('data/20241124/190009/'),

    Path('data/20241124/223921/'),
    
]
# Initialize lists to store loading and unloading data from all files
loading_z_positions = []
loading_forces = []
unloading_z_positions = []
unloading_forces = []
for file_path in file_paths:
    # Load the data
    force_data = pd.read_csv(file_path / 'force_data.csv')
    z_position_data = pd.read_csv(file_path / 'z_position_data.csv')
    # Round timestamps to align data and facilitate merging
    force_data['Timestamp'] = force_data['Timestamp'].round(2)
    z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)
    # Merge data on the rounded Timestamp column
    merged_data = pd.merge(force_data, z_position_data, on="Timestamp", suffixes=('_force', '_z'), how='outer')
    # Interpolate missing data points
    merged_data = merged_data.sort_values(by='Timestamp').interpolate(method='linear')
    # Multiply Z Position by -1 to make it positive
    merged_data["Z Position"] = merged_data["Z Position"] * -1
    # Use event markers to separate loading and unloading phases
    loading_start = merged_data[merged_data['Event'] == 1].index[0]
    unloading_start = merged_data[merged_data['Event'] == 2].index[0]
    # Separate the data into loading and unloading phases
    loading_data = merged_data.loc[loading_start:unloading_start - 1]
    unloading_data = merged_data.loc[unloading_start:]
    # Append loading and unloading data
    loading_z_positions.append(loading_data["Z Position"].values)
    loading_forces.append(loading_data["Force Magnitude"].values)
    unloading_z_positions.append(unloading_data["Z Position"].values)
    unloading_forces.append(unloading_data["Force Magnitude"].values)
# Define a function to align, shift, and calculate mean and std deviation
<<<<<<< HEAD
def calculate_shaded_error(z_positions, forces, shift=0.0019):
=======
def calculate_shaded_error(z_positions, forces, shift=0.0000, shiftdown=0.4):
    """
    Calculate the shaded error with a specified shift for Z positions and downward adjustment for forces.

    Parameters:
        z_positions (list of arrays): List of Z positions for each test.
        forces (list of arrays): List of force values for each test.
        shift (float): Value to shift Z positions to align properly.
        shiftdown (float): Value to shift forces downward.

    Returns:
        mean_z (pd.Series): Mean of aligned and shifted Z positions.
        mean_force (pd.Series): Mean of aligned and shifted forces.
        std_force (pd.Series): Standard deviation of aligned and shifted forces.
    """
>>>>>>> refs/remotes/origin/main
    # Determine the smallest length to align all data
    min_length = min(len(z) for z in z_positions)

    # Align all data to the smallest length
    aligned_z_positions = [z[:min_length] for z in z_positions]
    aligned_forces = [f[:min_length] for f in forces]

    # Apply the shift to Z positions and remove negative values
    shifted_z_positions = [z - shift for z in aligned_z_positions]
    valid_indices = [z >= 0 for z in shifted_z_positions]
    filtered_z_positions = [z[v] for z, v in zip(shifted_z_positions, valid_indices)]

    # Apply the downward shift to forces
    filtered_forces = [f[v] - shiftdown for f, v in zip(aligned_forces, valid_indices)]

    # Stack and calculate mean and std deviation
    mean_z = pd.DataFrame(filtered_z_positions).mean(axis=0)
    mean_force = pd.DataFrame(filtered_forces).mean(axis=0)
    std_force = pd.DataFrame(filtered_forces).std(axis=0)

    return mean_z, mean_force, std_force

# Ensure the calculate_shaded_error function is called before attempting the conversion
mean_z_loading, mean_force_loading, std_force_loading = calculate_shaded_error(loading_z_positions, loading_forces)
mean_z_unloading, mean_force_unloading, std_force_unloading = calculate_shaded_error(unloading_z_positions, unloading_forces)

# Convert pandas Series to numpy arrays (check and handle potential undefined variables)
mean_z_loading = mean_z_loading.to_numpy() if isinstance(mean_z_loading, pd.Series) else mean_z_loading
mean_force_loading = mean_force_loading.to_numpy() if isinstance(mean_force_loading, pd.Series) else mean_force_loading
std_force_loading = std_force_loading.to_numpy() if isinstance(std_force_loading, pd.Series) else std_force_loading

mean_z_unloading = mean_z_unloading.to_numpy() if isinstance(mean_z_unloading, pd.Series) else mean_z_unloading
mean_force_unloading = mean_force_unloading.to_numpy() if isinstance(mean_force_unloading, pd.Series) else mean_force_unloading
std_force_unloading = std_force_unloading.to_numpy() if isinstance(std_force_unloading, pd.Series) else std_force_unloading
_unloading = std_force_unloading.to_numpy() if isinstance(std_force_unloading, pd.Series) else std_force_unloading

# Now plot as before
plt.figure(figsize=(10, 6))

# Plot loading phase
plt.plot(mean_z_loading, mean_force_loading, label="Loading (Mean)", color='red')
plt.fill_between(mean_z_loading, mean_force_loading - std_force_loading, mean_force_loading + std_force_loading,
                 color='red', alpha=0.3, label="Loading (Std Dev)")

# Plot unloading phase
plt.plot(mean_z_unloading, mean_force_unloading, label="Unloading (Mean)", color='blue')
plt.fill_between(mean_z_unloading, mean_force_unloading - std_force_unloading, mean_force_unloading + std_force_unloading,
                 color='blue', alpha=0.3, label="Unloading (Std Dev)")

# Plot the green dashed line
green_line_x = [0, 0.01, 0.0155]
green_line_y = [0, 11, 11]
plt.plot(green_line_x, green_line_y, color='green', linestyle='--', linewidth=2, label="Desired Traj")
# Plot the green dashed line
yellow_line_x = [0, 0.0059, 0.0155]
yellow_line_y = [0, 10.5, 10.5]
plt.plot(yellow_line_x, yellow_line_y, color='orange', linestyle='--', linewidth=2, label="Pressing Test Result Traj")

plt.xlabel("Z Position (m)")
plt.ylabel("Force Magnitude (N)")
plt.title("Shaded Error Plot of Stiffness on the Pressing Test")
plt.legend()
plt.grid(True)

# Save and show
save_path = Path('data/20241124/shaded_error_plot_shifted_BaRiFlex.png')
plt.savefig(save_path)
plt.show()

print(f"Plot saved to {save_path}")
