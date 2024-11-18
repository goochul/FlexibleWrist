import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

# Centralized PATH variable
file_PATH = Path('data/20241117/10N_force_threshold')

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
cutoff_frequency = 6  # Low-pass filter cutoff frequency

# Apply the low-pass filter to the force data
try:
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
except KeyError as e:
    print(f"Error applying low-pass filter: {e}. Check if 'Force Magnitude' column exists.")
    print("Force data columns:", force_data.columns)
    raise

# Merge data on the rounded Timestamp column
merged_data = pd.merge(force_data, z_position_data, on="Timestamp", suffixes=('_force', '_z'), how='outer')

# Interpolate missing data points
merged_data = merged_data.sort_values(by='Timestamp').interpolate(method='linear')

# Multiply Z Position by -1 to make it positive
merged_data["Z Position"] = merged_data["Z Position"] * -1

# Filter data for Z Position >= 0.0025
filtered_data = merged_data[merged_data['Z Position'] >= 0.0020].copy()

# Calculate stiffness during loading phase (Force / Displacement)
filtered_data['Stiffness'] = filtered_data['Filtered Force Magnitude'] / filtered_data['Z Position']

# First Figure: Time-domain plot
time_domain_plot_path = file_PATH / "time_domain_plot_with_LPF.png"
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot filtered force data
line1, = ax1.plot(filtered_data["Timestamp"], filtered_data["Filtered Force Magnitude"], label="Force Magnitude", color='tab:blue')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force Magnitude (N)", color='tab:blue')
ax1.grid(True)

# Create a twin axis for Z position
ax2 = ax1.twinx()
line2, = ax2.plot(filtered_data["Timestamp"], filtered_data["Z Position"], label="Z Position", color='tab:red', linestyle="--")
ax2.set_ylabel("Z Position (m)", color='tab:red')

# Combine legends
ax1.legend(handles=[line1, line2], loc="upper left", fontsize=10, frameon=True)
plt.title("Filtered Force Magnitude and Z Position Over Time")

# Save and close the figure
plt.savefig(time_domain_plot_path, dpi=300)
plt.close(fig)

print(f"Time-domain plot saved as '{time_domain_plot_path}'.")

# Second Figure: Stiffness plot
stiffness_plot_path = file_PATH / "stiffness_plot_with_LPF.png"
plt.figure(figsize=(12, 6))

# Plot stiffness vs Z position
plt.plot(filtered_data["Z Position"], filtered_data["Stiffness"], label="Stiffness", color='tab:green')
plt.xlabel("Z Position (m)")
plt.ylabel("Stiffness (N/m)")
plt.title("Stiffness vs Z Position")
plt.legend()
plt.grid(True)

# Save and close the figure
plt.savefig(stiffness_plot_path, dpi=300)
plt.close()

print(f"Stiffness plot saved as '{stiffness_plot_path}'.")
