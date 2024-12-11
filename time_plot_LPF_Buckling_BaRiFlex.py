import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from pathlib import Path
from scipy.stats import linregress
import numpy as np

# Centralized PATH variable
# file_PATH = Path('data/20241127/134210/')
# file_PATH = Path('data/20241127/134523/')
# file_PATH = Path('data/20241127/134949/')
# file_PATH = Path('data/20241127/135401/')
# file_PATH = Path('data/20241127/135820/')
file_PATH = Path('data/20241210/Robotiq/172652/')

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
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Parameters for the low-pass filter
sampling_frequency = 30  # Assume 30 Hz sampling frequency
cutoff_frequency = 0.5  # Low-pass filter cutoff frequency

# Apply the low-pass filter to the force data
try:
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
except KeyError as e:
    print(f"Error applying low-pass filter: {e}. Check if 'Force Magnitude' column exists.")
    raise

# Merge data on the rounded Timestamp column using asof for tolerance matching
force_data = force_data.sort_values('Timestamp')
z_position_data = z_position_data.sort_values('Timestamp')
merged_data = pd.merge_asof(force_data, z_position_data, on="Timestamp", direction="nearest")

# Calculate the slope (derivative) over a 5-second rolling window
window_size = int(3 * sampling_frequency)  # Convert 3 seconds to data points
merged_data['Force Slope'] = 100 * merged_data['Filtered Force Magnitude'].rolling(window=window_size).apply(
    lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else 0,
    raw=False
)

# Set the parameters
initial_slope_threshold = 0.05  # Represents near-zero slope to identify starting point
slope_threshold = 1.2           # The threshold that represents a significant increase

# Initialize index values
slope_threshold_index = None
touching_point_index = None

# Step 1: Find the index where slope exceeds the slope_threshold
for idx in range(len(merged_data['Force Slope'])):
    if merged_data['Force Slope'].iloc[idx] > slope_threshold:
        slope_threshold_index = idx
        break

# Step 2: Go backwards to find the closest point where slope is at or below the initial_slope_threshold
if slope_threshold_index is not None:
    for idx in range(slope_threshold_index, -1, -1):
        if merged_data['Force Slope'].iloc[idx] <= initial_slope_threshold:
            touching_point_index = idx
            break

# Output the results
if touching_point_index is not None:
    print(f"Touching point index found closest to the slope threshold index: {touching_point_index}")
    print(f"Timestamp of touching point: {merged_data['Timestamp'].iloc[touching_point_index]}")
else:
    print("No touching point found where slope starts increasing.")

if touching_point_index is not None:
    touching_point_force = merged_data.loc[touching_point_index, 'Filtered Force Magnitude']
    touching_point_timestamp = merged_data.loc[touching_point_index, 'Timestamp']
    touching_point_z_position = merged_data.loc[touching_point_index, 'Z Position']
    merged_data['Offset Z Position'] = merged_data['Z Position'] - touching_point_z_position

    print(f"Touching Point Index: {touching_point_index}")
    print(f"Touching Point Force: {touching_point_force:.2f} N")
    print(f"Touching Point Timestamp: {touching_point_timestamp:.2f}s")
    print(f"Touching Point Z Position: {touching_point_z_position:.5f} m")
    print(f"Z-Position values offset by {touching_point_z_position:.5f} m.")
else:
    print("No valid touching point detected based on slope minima.")

# Plotting both filtered and non-filtered force data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Calculate the y-range for the filtered force data
force_magnitudes = merged_data["Filtered Force Magnitude"]
max_force_magnitude = max(abs(val) for val in force_magnitudes)
force_y_range = [-(max_force_magnitude + 5), max_force_magnitude + 5]

# Plot the filtered force data
line1, = ax1.plot(merged_data["Timestamp"], merged_data["Filtered Force Magnitude"], label="Filtered Force Magnitude", color='blue')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force Magnitude (N)", color='blue')
ax1.set_ylim(force_y_range)
ax1.grid(True)

# Add Z Position as twin axis
ax1_z = ax1.twinx()

# Calculate the y-range for Z Position
max_z_position = max(abs(val) for val in merged_data["Z Position"])
z_position_range = [-0.03, 0.03]
line2, = ax1_z.plot(
    merged_data["Timestamp"],
    merged_data["Offset Z Position"],  # Use the offset column
    label="Offset Z Position",
    color='tab:purple',
    linestyle="--",
)
ax1_z.set_ylabel("Offset Z Position (m)", color='tab:purple')
ax1_z.set_ylim(z_position_range)

# Plot the slope on the same axis
line3, = ax1.plot(
    merged_data["Timestamp"],
    merged_data["Force Slope"],
    label="Force Slope (3s Window) * 100",
    color='orange',
)

# Draw the horizontal lines representing the slope thresholds
ax1.axhline(y=initial_slope_threshold, color='grey', linestyle='-.', label=f"Initial Slope Threshold ({initial_slope_threshold})")
ax1.axhline(y=slope_threshold, color='brown', linestyle=':', label=f"Slope Threshold ({slope_threshold})")

# Mark the touching point if detected
if touching_point_index is not None:
    # Mark the vertical line on the force plot
    ax1.axvline(x=touching_point_timestamp, color='red', linestyle='--', label="Touching Point")

    # # Add text annotation near the touching point
    # ax1.text(
    #     touching_point_timestamp + 2,
    #     touching_point_force -1,  # Offset to avoid overlap with the line
    #     f"Touching Point\nTime: {touching_point_timestamp:.2f}s\nZ: {touching_point_z_position:.5f} m",
    #     color='red',
    #     fontsize=10,
    #     ha='left',
    #     va='center',
    # )

    # # Add a red dot at the Z-Position value on the twin axis
    # ax1_z.plot(
    #     touching_point_timestamp,
    #     touching_point_z_position,
    #     'o',  # Circle marker
    #     color='red',
    #     label="Touching Point (Z)",
    # )

# Add legend combining all lines
ax1.legend(handles=[line1, line2, line3], loc="upper left")
ax1.set_title("Filtered Force and Slope with Touching Point")

# Calculate the y-range for the non-filtered force data
raw_force_magnitudes = merged_data["Force Magnitude"]
max_raw_force_magnitude = max(abs(val) for val in raw_force_magnitudes)
raw_force_y_range = [-(max_raw_force_magnitude + 5), max_raw_force_magnitude + 5]

# Plot the non-filtered force data
line3, = ax2.plot(merged_data["Timestamp"], merged_data["Force Magnitude"], label="Raw Force Magnitude", color='green')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force Magnitude (N)", color='green')
ax2.set_ylim(raw_force_y_range)
ax2.grid(True)

# Add Z Position as twin axis
ax2_z = ax2.twinx()
line4, = ax2_z.plot(merged_data["Timestamp"], merged_data["Z Position"], label="Z Position", color='tab:red', linestyle="--")
ax2_z.set_ylabel("Z Position (m)", color='tab:red')
ax2_z.set_ylim(z_position_range)

# Mark the touching point on the non-filtered plot as well
if touching_point_index is not None:
    ax2.axvline(x=touching_point_timestamp, color='red', linestyle='--', label="Touching Point")
       # Add text annotation near the touching point
    ax2.text(
        touching_point_timestamp + 2,
        touching_point_force -1,  # Offset to avoid overlap with the line
        f"Touching Point\nTime: {touching_point_timestamp:.2f}s\nZ: {touching_point_z_position:.5f} m",
        color='red',
        fontsize=10,
        ha='left',
        va='center',
    )

    # Add a red dot at the Z-Position value on the twin axis
    ax2_z.plot(
        touching_point_timestamp,
        touching_point_z_position,
        'o',  # Circle marker
        color='red',
        label="Touching Point (Z)",
    )

# Add legend and title for the raw data plot
ax2.legend(handles=[line3, line4], loc="upper left")
ax2.set_title("Raw Force Magnitude with Touching Point")

# Adjust layout and show the plots
plt.tight_layout()
plt.show()

# # Define the full file path including the file name
# plot_file_path = file_PATH / "filtered_force_plot.png"  # Specify the file name

# # Save the plot
# plt.savefig(plot_file_path, dpi=300)
# plt.close(fig)

# print(f"Plot saved as '{plot_file_path}'.")