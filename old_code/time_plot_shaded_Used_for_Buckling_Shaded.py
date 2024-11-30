import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from pathlib import Path
from scipy.stats import linregress
import numpy as np

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
    
    force_data['Timestamp'] = force_data['Timestamp'].round(2)
    z_position_data['Timestamp'] = z_position_data['Timestamp'].round(2)
    
    sampling_frequency = 30
    cutoff_frequency = 0.5
    
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
    
    merged_data = pd.merge_asof(force_data.sort_values('Timestamp'), 
                                z_position_data.sort_values('Timestamp'), 
                                on="Timestamp", direction="nearest")
    
    window_size = int(3 * sampling_frequency)
    merged_data['Force Slope'] = 100 * merged_data['Filtered Force Magnitude'].rolling(window=window_size).apply(
        lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else 0,
        raw=False
    )
    
    slope_minima_indices = argrelextrema(merged_data['Force Slope'].values, np.less)[0]
    
    touching_point_index = next((idx for idx in slope_minima_indices 
                                 if -0.6 <= merged_data['Force Slope'].iloc[idx] <= 0.6), None)
    
    if touching_point_index is not None:
        touching_point_z_position = merged_data.loc[touching_point_index, 'Z Position']
        merged_data['Offset Z Position'] = merged_data['Z Position'] - touching_point_z_position
    
    return merged_data, touching_point_index

# Process all datasets and find touching points
all_datasets = [process_dataset(path) for path in file_paths]

# Unpack the datasets and touching points
processed_datasets, touching_points = zip(*all_datasets)

# Calculate mean and standard deviation
mean_force = pd.concat([df['Filtered Force Magnitude'] for df in processed_datasets], axis=1).mean(axis=1)
std_force = pd.concat([df['Filtered Force Magnitude'] for df in processed_datasets], axis=1).std(axis=1)
mean_z = pd.concat([df['Offset Z Position'] for df in processed_datasets], axis=1).mean(axis=1)
std_z = pd.concat([df['Offset Z Position'] for df in processed_datasets], axis=1).std(axis=1)

print(mean_force)

# Calculate the average touching time and average Z position
valid_touching_times = [mean_force.index[touching_point] for touching_point in touching_points if touching_point is not None]
valid_touching_z_positions = [mean_z.iloc[touching_point] for touching_point in touching_points if touching_point is not None]

print(touching_points)

if valid_touching_times and valid_touching_z_positions:
    avg_touching_time = np.mean(valid_touching_times)
    avg_touching_z_position = np.mean(valid_touching_z_positions)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot Force Magnitude
    ax1 = plt.gca()
    ax1.plot(mean_force.index, mean_force, label="Filtered Force Magnitude Mean", color='blue')
    ax1.fill_between(mean_force.index, mean_force - std_force, mean_force + std_force,
                     color='blue', alpha=0.3, label="Force Magnitude Std Dev")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Magnitude (N)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([-15, 15])

    # Create a twin axis for Z Position
    ax2 = ax1.twinx()
    ax2.plot(mean_z.index, mean_z, label="Offset Z Position Mean", color='red')
    ax2.fill_between(mean_z.index, mean_z - std_z, mean_z + std_z,
                     color='red', alpha=0.3, label="Z Position Std Dev")
    ax2.set_ylabel("Offset Z Position (m)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([-0.03, 0.03])

    # Plot a single vertical line at the average touching time and add text with the average Z position
    ax1.axvline(x=avg_touching_time, color='black', linestyle='--', label='Average Touching Point')
    ax2.text(avg_touching_time, avg_touching_z_position, 
             f'Avg Touching\n{avg_touching_time:.2f}s\n{avg_touching_z_position:.4f}m',
             color='green', fontsize=12, ha='center')

    # Set title and add legend
    plt.title("Filtered Force Magnitude and Offset Z Position over Time with Average Touching Point")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid touching points found.")