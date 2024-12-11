import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
from scipy.stats import linregress

# Centralized PATH variable
file_PATH = Path('data/20241210/Robotiq/172652/')

# Load the data
force_data = pd.read_csv(file_PATH / 'force_data.csv')
z_position_data = pd.read_csv(file_PATH / 'z_position_data.csv')

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
force_data['Filtered Force Magnitude'] = low_pass_filter(
    force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
)

# Merge data on the rounded Timestamp column using asof for tolerance matching
force_data = force_data.sort_values('Timestamp')
z_position_data = z_position_data.sort_values('Timestamp')
merged_data = pd.merge_asof(force_data, z_position_data, on="Timestamp", direction="nearest")

# Calculate the slope (derivative) over a 3-second rolling window
window_size = int(3 * sampling_frequency)  # Convert 3 seconds to data points
merged_data['Force Slope'] = 100 *merged_data['Filtered Force Magnitude'].rolling(window=window_size).apply(
    lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else 0,
    raw=False
)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Filtered Force Magnitude
plt.plot(merged_data['Timestamp'], merged_data['Filtered Force Magnitude'], label="Filtered Force Magnitude", color='blue')

# Plot Z Position
plt.plot(merged_data['Timestamp'], merged_data['Z Position'], label="Z Position", color='purple', linestyle="--")

# Plot Force Slope
plt.plot(merged_data['Timestamp'], merged_data['Force Slope'], label="Force Slope (3s Window)", color='orange')

# Customize the plot
plt.xlabel("Time (s)")
plt.ylabel("Values")
plt.title("Force, Z-Position, and 3-Second Slope")
plt.legend(loc="best")
plt.grid()
plt.tight_layout()

# Display the plot
plt.show()