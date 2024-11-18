import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

# Centralized PATH variable
file_PATH = Path('data/20241117/20N_force_threshold_35mm')

# Load the data
force_data = pd.read_csv(file_PATH / 'force_data.csv')
z_position_data = pd.read_csv(file_PATH / 'z_position_data.csv')

# Print the number of data points in each DataFrame
print(f"Number of data points in force_data: {len(force_data)}")
print(f"Number of data points in z_position_data: {len(z_position_data)}")

# Round timestamps to align data and facilitate merging
force_data['Timestamp'] = force_data['Timestamp'].round(2)
z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)

# Low-pass filter function
def low_pass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
    - data: The input data to filter.
    - cutoff: The cutoff frequency of the filter (Hz).
    - fs: The sampling frequency of the data (Hz).
    - order: The order of the Butterworth filter.
    
    Returns:
    - The filtered data.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Parameters for the low-pass filter
sampling_frequency = 100  # Assume 100 Hz sampling frequency
cutoff_frequency = 10  # Low-pass filter cutoff frequency

# Apply the low-pass filter to the force data
try:
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
except KeyError as e:
    print(f"Error applying low-pass filter: {e}. Check if 'Force Magnitude' column exists.")
    print("Force data columns:", force_data.columns)
    raise

# Merge data on the rounded Timestamp column using asof for tolerance matching
force_data = force_data.sort_values('Timestamp')
z_position_data = z_position_data.sort_values('Timestamp')
merged_data = pd.merge_asof(force_data, z_position_data, on="Timestamp", direction="nearest")

# Find the maximum force and its corresponding timestamp and Z position
max_force_index = merged_data['Filtered Force Magnitude'].idxmax()
max_force = merged_data.loc[max_force_index, 'Filtered Force Magnitude']
max_force_timestamp = merged_data.loc[max_force_index, 'Timestamp']
max_force_z_position = merged_data.loc[max_force_index, 'Z Position']

print(f"Maximum Force: {max_force:.2f} N at Timestamp: {max_force_timestamp:.2f}s, Z Position: {max_force_z_position:.5f} m")

# Plotting the force and z_position data on the time domain and saving
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot filtered force data
line1, = ax1.plot(force_data["Timestamp"], force_data["Filtered Force Magnitude"], label="Force Magnitude", color='tab:blue', linewidth=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force Magnitude (N)", color='tab:blue')
ax1.set_xlim([0, max(force_data["Timestamp"]) if not force_data.empty else 1])
ax1.set_ylim([-25, 25])
ax1.grid(True)

# Create a twin axis for Z Position
ax2 = ax1.twinx()
line2, = ax2.plot(z_position_data["Timestamp"], z_position_data["Z Position"], label="Z Position", color='tab:red', linestyle="--", linewidth=2)
ax2.set_ylabel("Z Position (m)", color='tab:red')
ax2.set_ylim([-0.05, 0.05])

# Add vertical line and annotation for maximum force
ax1.axvline(x=max_force_timestamp, color='green', linestyle='--', linewidth=1.5)
ax1.text(max_force_timestamp + 4, max_force + 2, f"Buckling effect\n(Max Force: {max_force:.2f} N)", 
         color='green', fontsize=10, ha='center')
ax1.text(max_force_timestamp + 4, max_force - 7, f"Z = {max_force_z_position:.5f} m", color='green', fontsize=10, ha='center')

# Combine legends from both axes into one inside the figure
ax1.legend(handles=[line1, line2], loc="upper left", fontsize=10, frameon=True)

time_plot_path = file_PATH / "force_displacement_plot_with_LPF_Line.png"

# Adding title
plt.title("Filtered Force Magnitude and Z Position Over Time")

# Show the plot
# plt.show()

# Save the plot (optional)
plt.savefig(time_plot_path, dpi=300)
plt.close(fig)

print(f"Plot saved as '{time_plot_path}'.")
