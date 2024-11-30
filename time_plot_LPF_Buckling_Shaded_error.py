import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from pathlib import Path
from scipy.stats import linregress
import numpy as np
import os

# List of file paths
file_paths = [
    Path('data/20241126/155358/'),
    Path('data/20241126/164849/'),
    Path('data/20241126/165145/'),
    Path('data/20241126/165511/'),
    Path('data/20241126/165834/')
]

# Low-pass filter function
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Function to process a single dataset
def process_dataset(file_path):
    force_data = pd.read_csv(file_path / 'force_data.csv')
    z_position_data = pd.read_csv(file_path / 'z_position_data.csv')
    
    # Round timestamps for alignment
    force_data['Timestamp'] = force_data['Timestamp'].round(2)
    z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)
    
    # Apply low-pass filter
    sampling_frequency = 30
    cutoff_frequency = 0.5
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
    
    # Merge the datasets on timestamp
    merged_data = pd.merge_asof(force_data.sort_values('Timestamp'), 
                                z_position_data.sort_values('Timestamp'), 
                                on="Timestamp", direction="nearest")
    # print(merged_data)
    
    # Calculate Force Slope
    window_size = int(3 * sampling_frequency)
    merged_data['Force Slope'] = 100 * merged_data['Filtered Force Magnitude'].rolling(window=window_size).apply(
        lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else 0,
        raw=False
    )
    
    # Identify the touching point
    slope_minima_indices = argrelextrema(merged_data['Force Slope'].values, np.less)[0]
    touching_point_index = next((idx for idx in slope_minima_indices 
                                 if -0.4 <= merged_data['Force Slope'].iloc[idx] <= 0.4), None)
    
    if touching_point_index is not None:
        touching_point_z_position = merged_data.loc[touching_point_index, 'Z Position']
        merged_data['Offset Z Position'] = merged_data['Z Position'] - touching_point_z_position
    
    return merged_data, touching_point_index

# # Function to save merged data
# def process_and_save_dataset(file_path, output_path):
#     merged_data, _ = process_dataset(file_path)
#     merged_data.to_csv(output_path, index=False)
#     print(f"Saved merged data for {file_path} to {output_path}")

# # Output directory for merged datasets
# output_dir = Path('merged_data')
# os.makedirs(output_dir, exist_ok=True)

# # Save merged data for each dataset
# for i, file_path in enumerate(file_paths):
#     output_file = output_dir / f'merged_data_{i+1}.csv'
#     process_and_save_dataset(file_path, output_file)

# Process all datasets for further calculations
all_datasets = [process_dataset(path) for path in file_paths]

# Unpack the datasets and touching points
processed_datasets, touching_points = zip(*all_datasets)

# # Directory to save processed datasets and touching points
# csv_output_dir = Path('csv_output')
# os.makedirs(csv_output_dir, exist_ok=True)

# # Save each processed dataset and its corresponding touching point to CSV
# for i, (dataset, touching_point) in enumerate(zip(processed_datasets, touching_points)):
#     # Save the processed dataset
#     dataset_file = csv_output_dir / f'processed_dataset_{i+1}.csv'
#     dataset.to_csv(dataset_file, index=False)
#     print(f"Saved processed dataset {i+1} to {dataset_file}")
    
#     # Save the touching point index (if it exists) to a separate CSV
#     touching_point_file = csv_output_dir / f'touching_point_{i+1}.csv'
#     with open(touching_point_file, 'w') as f:
#         f.write('Touching Point Index\n')
#         f.write(f'{touching_point}\n' if touching_point is not None else 'None\n')
#     print(f"Saved touching point {i+1} to {touching_point_file}")

# Align all datasets to a common time axis
time = processed_datasets[0]['Timestamp'].sort_values().drop_duplicates()

# Save the time variable as a CSV
# time.to_csv('time_variable.csv', index=True)

# print("Time variable saved as 'time_variable.csv'")

# Reindex all datasets to the common time index
aligned_force = [
    df.set_index('Timestamp')['Filtered Force Magnitude'].reindex(time).interpolate()
    for df in processed_datasets
]

# aligned.to_csv('time_variable.csv', index=True)

aligned_z = [
    df.set_index('Timestamp')['Offset Z Position'].reindex(time).interpolate()
    for df in processed_datasets
]

# Reindex raw force and raw Z position datasets to the common time index
aligned_raw_force = [
    df.set_index('Timestamp')['Force Magnitude'].reindex(time).interpolate()
    for df in processed_datasets
]
aligned_raw_z_position = [
    df.set_index('Timestamp')['Z Position'].reindex(time).interpolate()
    for df in processed_datasets
]
# print(aligned_raw_z_position)

# Calculate mean and standard deviation
mean_force = pd.concat(aligned_force, axis=1).mean(axis=1)
print(time.size)
# print(time)
# print('mean_force')
# print(mean_force.size)
std_force = pd.concat(aligned_force, axis=1).std(axis=1)
mean_z = pd.concat(aligned_z, axis=1).mean(axis=1)
std_z = pd.concat(aligned_z, axis=1).std(axis=1)
# print(mean_z)
# print(mean_z.size)
# mean_z.to_csv('mean_z.csv', index=True)

# print(touching_points)

# Calculate the average touching time and average Z position
valid_touching_times = [mean_force.index[touching_point] for touching_point in touching_points if touching_point is not None]
valid_touching_z_positions = [mean_z.iloc[touching_point] for touching_point in touching_points if touching_point is not None]

print(valid_touching_times)
# print(valid_touching_z_positions)

# Extract only the DataFrame part from all_datasets
processed_dfs = [df for df, _ in all_datasets]  # Ignore the touching point index

# Calculate mean and standard deviation for raw force and Z position
mean_raw_force = pd.concat(aligned_raw_force, axis=1).mean(axis=1)
std_raw_force = pd.concat(aligned_raw_force, axis=1).std(axis=1)
mean_raw_z_position = pd.concat(aligned_raw_z_position, axis=1).mean(axis=1)
std_raw_z_position = pd.concat(aligned_raw_z_position, axis=1).std(axis=1)

valid_touching_raw_z_positions = [mean_raw_z_position.iloc[touching_point] for touching_point in touching_points if touching_point is not None]

# print(mean_raw_z_position)

# Create a single figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 16))  # Two rows, one column

# First Plot: Filtered Force Magnitude and Offset Z Position
if valid_touching_times and valid_touching_z_positions:
    avg_touching_time = np.mean(valid_touching_times)
    avg_touching_z_position = np.mean(valid_touching_z_positions)

    ax1 = axs[0]
    # Plot Filtered Force Magnitude
    ax1.plot(mean_force.index, mean_force, label="Filtered Force Magnitude Mean", color='blue')
    ax1.fill_between(mean_force.index, mean_force - std_force, mean_force + std_force,
                     color='blue', alpha=0.3, label="Force Magnitude Std Dev")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Magnitude (N)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([-15, 15])

    # Plot Offset Z Position
    ax2 = ax1.twinx()
    ax2.plot(mean_z.index, mean_z, label="Offset Z Position Mean", color='red')
    ax2.fill_between(mean_z.index, mean_z - std_z, mean_z + std_z,
                     color='red', alpha=0.3, label="Z Position Std Dev")
    ax2.set_ylabel("Offset Z Position (m)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([-0.03, 0.03])

    # Add Average Touching Point
    ax1.axvline(x=avg_touching_time, color='black', linestyle='--', label='Average Touching Point')
    # ax2.text(avg_touching_time, avg_touching_z_position, 
    #          f'Avg Touching\n{avg_touching_time:.2f}s\n{avg_touching_z_position:.4f}m',
    #          color='green', fontsize=12, ha='center')

    # Add title and legend
    ax1.set_title("Filtered Force Magnitude and Offset Z Position with Average Touching Point")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Second Plot: Raw Force Magnitude and Non-Offset Z Position
ax3 = axs[1]
# Plot Raw Force Magnitude
ax3.plot(mean_raw_force.index, mean_raw_force, label="Raw Force Magnitude Mean", color="green")
ax3.fill_between(mean_raw_force.index, mean_raw_force - std_raw_force, mean_raw_force + std_raw_force,
                 color="green", alpha=0.3, label="Raw Force Magnitude Std Dev")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Force Magnitude (N)", color="green")
ax3.tick_params(axis="y", labelcolor="green")
ax3.set_ylim([-15, 15])  # Adjusted for raw force range

# Plot Non-Offset Z Position
ax4 = ax3.twinx()
ax4.plot(mean_raw_z_position.index, mean_raw_z_position, label="Non-Offset Z Position Mean", color="red", linestyle="--")
ax4.fill_between(mean_raw_z_position.index, mean_raw_z_position - std_raw_z_position, mean_raw_z_position + std_raw_z_position,
                 color="red", alpha=0.3, label="Z Position Std Dev")
ax4.set_ylabel("Z Position (m)", color="red")
ax4.tick_params(axis='y', labelcolor='red')
ax4.set_ylim([-0.03, 0.03])  # Adjusted for Z position range

# Add Average Touching Point (Vertical Line)
if valid_touching_raw_z_positions and valid_touching_times:
    avg_touching_time = np.mean(valid_touching_times)
    avg_touching_raw_z_position = np.mean(valid_touching_raw_z_positions)

    # Add vertical line at avg_touching_time
    ax3.axvline(x=avg_touching_time, color="black", linestyle="--", label="Average Touching Point")

    # Plot a single red dot at avg_touching_time and avg_touching_raw_z_position
    ax4.scatter(avg_touching_time, avg_touching_raw_z_position, color="red", label="Touching Point (Raw Z)", zorder=5)

    # Add text label for the single red dot
    ax4.text(avg_touching_time + 6, avg_touching_raw_z_position - 0.003, 
             f"Touching Point\n{avg_touching_time:.2f}s\n{avg_touching_raw_z_position:.4f}m",
             color="blue", fontsize=12, ha="center")

# Add title and legend
ax3.set_title("Raw Force Magnitude and Non-Offset Z Position with Average Touching Point")
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper left")

# Add grid and layout
for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()