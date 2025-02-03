import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from pathlib import Path
from scipy.stats import linregress
import numpy as np
import os

title_name = "Filtered Force and Z Position with Buckling Point - thickness 0.9mm, angle 100 degrees, h = 11mm, Dc = 5.3"
save_fig = True  # Set to True or False as needed
# save_fig = False  # Set to True or False as needed

# List of file paths
file_paths = [

    # Path('data/20241126/155358/'),
    # Path('data/20241126/164849/'),
    # Path('data/20241126/165145/'),
    # Path('data/20241126/165511/'),
    # Path('data/20241126/165834/')

    # Path('data/20241211/181727/'),
    # Path('data/20241211/182206/'),
    # Path('data/20241211/182630/'),
    # Path('data/20241211/183050/'),


    # Path('data/20241211/192041/'),

    # Path('data/20250202/184332/'),
    # Path('data/20250202/184915/'),
    # Path('data/20250202/190623/'),

    Path('data/20250202/192449/'),
    Path('data/20250202/193015/'),
    Path('data/20250202/194504/'),

]


# Low-pass filter function
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

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
    
    # Calculate Force Slope
    window_size = int(3 * sampling_frequency)
    merged_data['Force Slope'] = 100 * merged_data['Filtered Force Magnitude'].rolling(window=window_size).apply(
        lambda x: linregress(range(len(x)), x).slope if len(x) > 1 else 0,
        raw=False
    )
    
    # Set the parameters
    initial_slope_threshold = 0.01  # Represents near-zero slope to identify starting point
    slope_threshold = 1.0           # The threshold that represents a significant increase

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

    # If touching point is found, calculate offset Z Position
    if touching_point_index is not None:
        touching_point_z_position = merged_data.loc[touching_point_index, 'Z Position']
        merged_data['Offset Z Position'] = merged_data['Z Position'] - touching_point_z_position
    else:
        # Initialize 'Offset Z Position' with NaN if no touching point is found
        merged_data['Offset Z Position'] = np.nan

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
    ax1.plot(mean_force.index.to_numpy(), mean_force.to_numpy(), label="Filtered Force Magnitude Mean", color='blue')
    ax1.fill_between(mean_force.index.to_numpy(), 
                    (mean_force - std_force).to_numpy(), 
                    (mean_force + std_force).to_numpy(), 
                    color='blue', alpha=0.3, label="Force Magnitude Std Dev")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Magnitude (N)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([-20, 20])

    # Offset Z Position plot
    ax2 = ax1.twinx()
    ax2.plot(mean_z.index.to_numpy(), mean_z.to_numpy(), label="Offset Z Position Mean", color='red')
    ax2.fill_between(mean_z.index.to_numpy(), 
                    (mean_z - std_z).to_numpy(), 
                    (mean_z + std_z).to_numpy(), 
                    color='red', alpha=0.3, label="Z Position Std Dev")
    ax2.set_ylabel("Offset Z Position (m)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([-0.04, 0.04])


    # Add Average Touching Point
    ax1.axvline(x=avg_touching_time, color='black', linestyle='--', label='Average Touching Point')

# ---------------- Buckling Point Detection ----------------
if valid_touching_times and valid_touching_z_positions:
    avg_touching_time = np.mean(valid_touching_times)
    # Find the nearest index to avg_touching_time
    touching_point_index = mean_force.index.get_indexer([avg_touching_time], method="nearest")[0]

    # Identify the peak force and its index starting from the touching point
    mean_force_from_touching = mean_force.iloc[touching_point_index:]
    peak_force_index = mean_force_from_touching.idxmax()
    peak_force = mean_force_from_touching[peak_force_index]

    # Find 80% of the peak force
    force_80_percent = 0.8 * peak_force

    # Find the index where force reaches 80% of the peak force
    idx_80_percent = mean_force_from_touching[mean_force_from_touching >= force_80_percent].index[0]
    pos_80_percent = mean_force.index.get_loc(idx_80_percent)

    # Perform linear fit for the increasing section from touching point to 80% of maximum force
    x_fit_values = mean_force.index[touching_point_index:pos_80_percent + 1].astype(float)
    y_fit_values = mean_force.iloc[touching_point_index:pos_80_percent + 1]

    linear_fit = np.polyfit(x_fit_values, y_fit_values, 1)

    # Calculate the intersection point
    intersection_x = (peak_force - linear_fit[1]) / linear_fit[0]
    intersection_y = peak_force

    # Interpolate the mean_z value at the intersection_x
    mean_z_values = mean_z.to_numpy()
    mean_z_indices = mean_z.index.astype(float).to_numpy()
    mean_z_at_buckling = np.interp(intersection_x, mean_z_indices, mean_z_values)

    mean_force_values = mean_force.to_numpy()
    mean_force_indices = mean_force.index.astype(float).to_numpy()
    mean_force_at_buckling = np.interp(intersection_x, mean_force_indices, mean_force_values)

    # Generate points for linear fit line
    x_fit_line = np.linspace(x_fit_values.min(), intersection_x, 100)
    y_fit_line = np.polyval(linear_fit, x_fit_line)

    # Plot the linear fit as a dashed line
    ax1.plot(x_fit_line, y_fit_line, '--', color='orange', label="Buckling Linear Fit")

    # Calculate the standard deviation of force at the buckling point
    std_force_at_buckling = np.interp(intersection_x, mean_force_indices, std_force.to_numpy())

    # Plot the interpolated point on the mean line
    ax1.scatter(intersection_x, mean_force_at_buckling, color="blue", label=f"Buckling Point on the line ({intersection_x:.2f}, {mean_force_at_buckling:.2f})")
    ax1.scatter(intersection_x, intersection_y, color="orange", label=f"80% interpolation Buckling Point ({intersection_x:.2f}, {intersection_y:.2f})")


    # Annotate the standard deviation at the buckling point
    ax1.text(intersection_x + 0.5, mean_force_at_buckling - std_force_at_buckling - 1,
            f"±{std_force_at_buckling:.2f}",
            color="blue", fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

    # Plot the standard deviation range as vertical lines
    ax1.vlines(intersection_x, mean_force_at_buckling - std_force_at_buckling,
            mean_force_at_buckling + std_force_at_buckling,
            colors='blue', linestyles='dotted', label=f"Std Dev Range at Buckling ({std_force_at_buckling:.2f})")

    # Draw a horizontal line from the buckling point to the maximum force value
    peak_force_pos = mean_force.index.get_loc(peak_force_index)
    ax1.hlines(intersection_y, intersection_x, mean_force.index[peak_force_pos].astype(float),
               colors='orange', linestyles='dashed')

    # Draw a vertical line at the buckling point
    ax1.axvline(x=intersection_x, color="purple", linestyle="--", label="Buckling Point Vertical Line")

    # Annotate the mean_z value on the plot
    if mean_z_at_buckling is not None:
        ax2.text(intersection_x + 5, mean_z_at_buckling,
                 f"Mean Z at Buckling:\n {mean_z_at_buckling:.4f} m",
                 color="black", fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

    # Calculate the standard deviation of Z position at the buckling point
    std_z_at_buckling = np.interp(intersection_x, mean_z_indices, std_z.to_numpy())

    # Mark the intersection point on the mean_z line
    ax2.scatter(intersection_x, mean_z_at_buckling, color="red", label=f"Z-position value at Buckling Point ({mean_z_at_buckling:.4f})")

    # Annotate the standard deviation at the buckling point for Z-position
    ax2.text(intersection_x + 0.5, mean_z_at_buckling - std_z_at_buckling - 0.002,
            f"±{std_z_at_buckling:.4f}",
            color="red", fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

    # Plot the standard deviation range as vertical lines for Z-position
    ax2.vlines(intersection_x, mean_z_at_buckling - std_z_at_buckling,
            mean_z_at_buckling + std_z_at_buckling,
            colors='red', linestyles='dotted', label=f"Std Dev Range at Buckling ({std_z_at_buckling:.4f})")

    # Calculate the ratio
    stiffness = mean_force_at_buckling / -mean_z_at_buckling

    print(f"force: {mean_force_at_buckling:.5f}")
    print(f"disp: {mean_z_at_buckling:.5f}")

    # Print using f-string formatting:
    #   - .2f means 2 decimal places
    print(f"Stiffness: {stiffness:.3f}")

# Mark Stiffness line
    ax2.text(intersection_x -5, -mean_z_at_buckling/2,
            f"Stiffness:\n {stiffness:.3f}",
            color="black", fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

    # Add title and legend
    ax1.set_title(title_name)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
# ---------------- End of Buckling Point Detection ----------------

# # ---------------- Buckling Point Detection ----------------
# if valid_touching_times and valid_touching_z_positions:
#     avg_touching_time = np.mean(valid_touching_times)
#     # Find the nearest index to avg_touching_time
#     touching_point_index = mean_force.index.get_indexer([avg_touching_time], method="nearest")[0]  # Nearest positional index

#     # Identify the peak force and its index starting from the touching point
#     mean_force_from_touching = mean_force.iloc[touching_point_index:]  # Force values after touching point
#     peak_force_index = mean_force_from_touching.idxmax()  # Index (label) of max force after touching point
#     peak_force = mean_force_from_touching[peak_force_index]  # Max force value

#     print(peak_force)

#     # Find 80% of the peak force
#     force_80_percent = 0.8 * peak_force

#     print(force_80_percent)

#     # Find the index where force reaches 80% of the peak force
#     idx_80_percent = mean_force_from_touching[mean_force_from_touching >= force_80_percent].index[0]
#     pos_80_percent = mean_force.index.get_loc(idx_80_percent)  # Positional index for slicing

#     # Perform linear fit from touching point to 80% of maximum force
#     x_fit_values_0_80 = mean_force.index[touching_point_index:pos_80_percent + 1].astype(float)  # Convert index to float
#     y_fit_values_0_80 = mean_force.iloc[touching_point_index:pos_80_percent + 1]  # Extract corresponding force values

#     print(x_fit_values_0_80)
#     print("x_fit and y_fit")
#     print(y_fit_values_0_80)

#     # Perform the linear fit
#     linear_fit_0_80 = np.polyfit(x_fit_values_0_80, y_fit_values_0_80, 1)

#     # Generate x and y points for plotting the linear fit
#     x_fit_line_0_80 = np.linspace(x_fit_values_0_80.min(), x_fit_values_0_80.max(), 100)
#     y_fit_line_0_80 = np.polyval(linear_fit_0_80, x_fit_line_0_80)


#     # ---------------- Second Linear Fit (80% to 100% of Maximum Force) ----------------
#     # Convert peak_force_index to positional index
#     peak_force_pos_index = mean_force.index.get_loc(peak_force_index)  # Convert to positional index

#     # Perform linear fit for the increasing section from 80% to 100% of maximum force
#     x_fit_values_80_100 = mean_force.index[pos_80_percent:peak_force_pos_index + 1].astype(float)  # Convert index to float (80% to peak)
#     y_fit_values_80_100 = mean_force.iloc[pos_80_percent:peak_force_pos_index + 1]  # Extract corresponding values

#     # Perform linear fit
#     linear_fit_80_100 = np.polyfit(x_fit_values_80_100, y_fit_values_80_100, 1)

#     # ---------------- Plot Linear Fits ----------------
#     # Extract coefficients
#     m0, b0 = linear_fit_0_80  # Slope and intercept for 0-80%
#     m1, b1 = linear_fit_80_100  # Slope and intercept for 80-100%

#     # Generate points for the first fit line (0-80%)
#     x_fit_line_0_80 = np.linspace(x_fit_values_0_80.min(), x_fit_values_0_80.max(), 100)
#     y_fit_line_0_80 = np.polyval(linear_fit_0_80, x_fit_line_0_80)
#     ax1.plot(x_fit_line_0_80, y_fit_line_0_80, '--', color='orange', label="Linear Fit (0–80%)")

#     # Generate points for the second fit line (80-100%)
#     x_fit_line_80_100 = np.linspace(x_fit_values_80_100.min(), x_fit_values_80_100.max(), 100)
#     y_fit_line_80_100 = np.polyval(linear_fit_80_100, x_fit_line_80_100)
#     ax1.plot(x_fit_line_80_100, y_fit_line_80_100, '--', color='green', label="Linear Fit (80-100%)")


#     # # Get the timestamp and force value at 80% of maximum force
#     # time_80_percent = mean_force.index[pos_80_percent]
#     # force_80_percent_value = force_80_percent


#     # # Plot the 80% point on the force plot
#     # ax1.scatter(time_80_percent, force_80_percent_value, color='red', label=f'80% Point ({time_80_percent:.2f}s, {force_80_percent_value:.2f}N)', zorder=5)

#     # # Annotate the 80% point
#     # ax1.text(time_80_percent + 2, force_80_percent_value, f'80%\n({force_80_percent_value:.2f} N)', 
#     #         color='red', fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.8))

#     # # Optionally, draw a vertical line to mark the 80% point
#     # ax1.axvline(x=time_80_percent, color='red', linestyle='--', alpha=0.7, label='80% Force Point')

#     # ---------------- Find Intersection of Linear Fits ----------------
#     if m0 != m1:
#         # Intersection point calculation
#         intersection_x = (b1 - b0) / (m0 - m1)
#         intersection_y = m0 * intersection_x + b0

#         # Plot intersection point
#         ax1.scatter(intersection_x, intersection_y, color='purple', label=f"Intersection ({intersection_x:.2f}, {intersection_y:.2f})")

#         # Find the point on the mean force line at the intersection_x
#         mean_line_y = np.interp(intersection_x, mean_force.index.astype(float), mean_force.to_numpy())

#         # Mark the point on the mean line
#         ax1.scatter(intersection_x, mean_line_y, color="blue", label=f"Buckling Point ({intersection_x:.2f}, {mean_line_y:.2f})")
#         ax1.scatter(intersection_x, intersection_y, color="orange", label=f"80-20 intersection ({intersection_x:.2f}, {intersection_y:.2f})")

#         # Plot the linear fit lines (orange and green)
#         x_fit_line_0_80 = np.linspace(x_fit_values_0_80.min(), intersection_x, 100)
#         y_fit_line_0_80 = np.polyval(linear_fit_0_80, x_fit_line_0_80)
#         ax1.plot(x_fit_line_0_80, y_fit_line_0_80, '--', color='orange', label="Buckling Linear Fit (0-80%)")

#         x_fit_line_80_100 = np.linspace(intersection_x, x_fit_values_80_100.max(), 100)
#         y_fit_line_80_100 = np.polyval(linear_fit_80_100, x_fit_line_80_100)
#         ax1.plot(x_fit_line_80_100, y_fit_line_80_100, '--', color='green', label="Buckling Linear Fit (80-100%)")

#         # # Extract the data from touching point to 80% of maximum force
#         # data_0_80 = mean_force.iloc[touching_point_index:pos_80_percent + 1]

#         # # Print the extracted data
#         # print("Data from 0% to 80% of Maximum Force:")
#         # print(data_0_80)

#         # Draw a vertical line at the buckling point
#         ax1.axvline(x=intersection_x, color="purple", linestyle="--", label="Buckling Point Vertical Line")

#     # ---------------- Annotate Mean Z Value ----------------
#     # Retrieve the mean_z value at the intersection point (if it exists)
#     mean_z_nearest_index = mean_z.index.get_indexer([intersection_x], method="nearest")[0]
#     mean_z_at_intersection = mean_z.iloc[mean_z_nearest_index]

#     # Check if the timestamp 60 exists in the mean_z index
#     time_60 = 60.0
#     if time_60 in mean_z.index:
#         z_position_60 = mean_z.loc[time_60]  # Get the Z position at 60 seconds
#     else:
#         # Interpolate the Z position if 60 seconds is not an exact index
#         z_position_60 = np.interp(time_60, mean_z.index.astype(float), mean_z.values)


#     if mean_z_at_intersection is not None:
#         ax2.text(intersection_x + 5, mean_z_at_intersection,
#                  f"Mean Z at Intersection:\n {mean_z_at_intersection:.4f} m",
#                  color="black", fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

#     # Mark the intersection point on the Z position plot
#     ax2.scatter(intersection_x, mean_z_at_intersection, color="red", label=f"Z-position at Intersection ({mean_z_at_intersection:.4f})")

#     ax2.scatter(time_60, z_position_60, color='purple', label=f'Point at 60s ({z_position_60:.4f}m)')

#     # Annotate the point
#     ax2.text(time_60 + 2, z_position_60, f'{z_position_60:.4f} m', color='purple', fontsize=10, ha='center')

#     # Add title and legend
#     ax1.set_title("Filtered Force Magnitude and Offset Z Position with Buckling Point and Intersection - Interpolation 2")
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
# # ---------------- End of Buckling Point Detection ----------------

# Second Plot: Raw Force Magnitude and Non-Offset Z Position
ax3 = axs[1]

# Convert data to NumPy arrays for compatibility
mean_raw_force_index = mean_raw_force.index.to_numpy()
mean_raw_force_values = mean_raw_force.to_numpy()
std_raw_force_values = std_raw_force.to_numpy()

mean_raw_z_position_index = mean_raw_z_position.index.to_numpy()
mean_raw_z_position_values = mean_raw_z_position.to_numpy()
std_raw_z_position_values = std_raw_z_position.to_numpy()

# Plot Raw Force Magnitude
ax3.plot(mean_raw_force_index, mean_raw_force_values, label="Raw Force Magnitude Mean", color="green")
ax3.fill_between(mean_raw_force_index, 
                 mean_raw_force_values - std_raw_force_values, 
                 mean_raw_force_values + std_raw_force_values,
                 color="green", alpha=0.3, label="Raw Force Magnitude Std Dev")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Force Magnitude (N)", color="green")
ax3.tick_params(axis="y", labelcolor="green")
ax3.set_ylim([-20, 20])  # Adjusted for raw force range

# Plot Non-Offset Z Position
ax4 = ax3.twinx()
ax4.plot(mean_raw_z_position_index, mean_raw_z_position_values, label="Non-Offset Z Position Mean", color="red", linestyle="--")
ax4.fill_between(mean_raw_z_position_index, 
                 mean_raw_z_position_values - std_raw_z_position_values, 
                 mean_raw_z_position_values + std_raw_z_position_values,
                 color="red", alpha=0.3, label="Z Position Std Dev")
ax4.set_ylabel("Z Position (m)", color="red")
ax4.tick_params(axis='y', labelcolor='red')
ax4.set_ylim([-0.04, 0.04])  # Adjusted for Z position range


# Add Average Touching Point (Vertical Line)
if valid_touching_raw_z_positions and valid_touching_times:
    avg_touching_time = np.mean(valid_touching_times)
    avg_touching_raw_z_position = np.mean(valid_touching_raw_z_positions)

    # Add vertical line at avg_touching_time
    ax3.axvline(x=avg_touching_time, color="black", linestyle="--", label="Average Touching Point")

    # Plot a single red dot at avg_touching_time and avg_touching_raw_z_position
    ax4.scatter(avg_touching_time, avg_touching_raw_z_position, color="red", label="Touching Point (Raw Z)", zorder=5)

    # Add text label for the single red dot
    ax4.text(avg_touching_time + 3, avg_touching_raw_z_position - 0.003, 
             f"Touching Point\n{avg_touching_time:.2f}s\n{avg_touching_raw_z_position:.4f}m",
             color="blue", fontsize=12, ha="center")

# Add title and legend
ax3.set_title("Raw Force Magnitude and Non-Offset Z Position with Average Touching Point")
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper right")

# Add grid and layout
for ax in axs:
    ax.grid(True)

# Save the figure
common = os.path.commonpath([str(p) for p in file_paths])
common_path = Path(common)  # Convert to a Path object
safe_title_name = title_name.replace(" ", "_").replace(":", "-")
save_path = common_path / f"{safe_title_name}.png"
# Save the figure if savefig is True
if save_fig:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
else:
    print("Figure not saved.")

# Show the plot
plt.tight_layout()
plt.show()

# print()