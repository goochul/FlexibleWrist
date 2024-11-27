import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Centralized PATH variable
file_PATH = Path('data/20241124/190009')

# Load the data
force_data = pd.read_csv(file_PATH / 'torque_data.csv')

# Print the number of data points in each DataFrame
print(f"Number of data points in torque_data: {len(force_data)}")

# Round timestamps to align data and facilitate merging
force_data['Timestamp'] = force_data['Timestamp'].round(2)

# Find the maximum force and its corresponding timestamp
max_force_index = force_data['Torque Magnitude'].idxmax()
max_force = force_data.loc[max_force_index, 'Torque Magnitude']
max_force_timestamp = force_data.loc[max_force_index, 'Timestamp']

print(f"Maximum Torque: {max_force:.2f} N at Timestamp: {max_force_timestamp:.2f}s")

# Plotting the force data on the time domain and saving
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot force data
ax1.plot(force_data["Timestamp"], force_data["Torque Magnitude"], label="Torque Magnitude", color='tab:blue', linewidth=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Torque Magnitude (N)", color='tab:blue')
ax1.grid(True)

# Combine legends into one inside the figure
ax1.legend(loc="upper left", fontsize=10, frameon=True)

time_plot_path = file_PATH / "pure_torque.png"

# Adding title
plt.title("Torque Magnitude Over Time")

# Show the plot
# plt.show()

# Save the plot (optional)
plt.savefig(time_plot_path, dpi=300)
plt.close(fig)

print(f"Plot saved as '{time_plot_path}'.")
