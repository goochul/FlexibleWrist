import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from pathlib import Path
import numpy as np
import os

# List of file paths
file_paths = [
    # Path('data/20250217/021118/')
    # Path('data/20250221/195232/')


    # Path('data/20250222/232016/'),
    # Path('data/20250222/232359/'),
    # Path('data/20250222/232840/'),

    # Path('data/20250223/151503/'),
    # Path('data/20250223/152241/'),
    # Path('data/20250223/153138/'),

    Path('data/20250223/204632/'),
    Path('data/20250223/205123/'),
]

# Low-pass filter function
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def process_dataset(file_path):
    # Read CSV files
    force_data = pd.read_csv(file_path / 'force_data.csv')
    torque_data = pd.read_csv(file_path / 'torque_data.csv')
    y_position_data = pd.read_csv(file_path / 'y_position_data.csv')
    
    # Round timestamps for alignment
    force_data['Timestamp'] = force_data['Timestamp'].round(2)
    torque_data['Timestamp'] = torque_data['Timestamp'].round(2)
    y_position_data['Timestamp'] = y_position_data['Timestamp'].round(2)
    
    # Sampling settings
    sampling_frequency = 30
    cutoff_frequency = 0.2

    # Apply low-pass filter to Force Magnitude and individual force components
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
    force_data['Filtered Fx'] = low_pass_filter(force_data['Fx'], cutoff_frequency, sampling_frequency)
    force_data['Filtered Fy'] = low_pass_filter(force_data['Fy'], cutoff_frequency, sampling_frequency)
    force_data['Filtered Fz'] = low_pass_filter(force_data['Fz'], cutoff_frequency, sampling_frequency)
    
    # Rotate Fx and Fy by 45 degrees about Z (i.e., 45° in the opposite direction)
    angle = np.deg2rad(45)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    force_data['Rotated Fx'] = cos_angle * force_data['Filtered Fx'] - sin_angle * force_data['Filtered Fy']
    force_data['Rotated Fy'] = sin_angle * force_data['Filtered Fx'] + cos_angle * force_data['Filtered Fy']
    
    # Process torque data: Filter torque magnitude and individual moment components
    torque_data['Filtered Torque Magnitude'] = low_pass_filter(
        torque_data['Torque Magnitude'], cutoff_frequency, sampling_frequency
    )
    torque_data['Filtered Tx'] = low_pass_filter(torque_data['Tx'], cutoff_frequency, sampling_frequency)
    torque_data['Filtered Ty'] = low_pass_filter(torque_data['Ty'], cutoff_frequency, sampling_frequency)
    torque_data['Filtered Tz'] = low_pass_filter(torque_data['Tz'], cutoff_frequency, sampling_frequency)
    
    # Rotate Tx and Ty by 45 degrees about Z
    torque_data['Rotated Tx'] = cos_angle * torque_data['Filtered Tx'] - sin_angle * torque_data['Filtered Ty']
    torque_data['Rotated Ty'] = sin_angle * torque_data['Filtered Tx'] + cos_angle * torque_data['Filtered Ty']
    
    # Merge force_data and y_position_data on Timestamp
    merged_data = pd.merge_asof(
        force_data.sort_values('Timestamp'),
        y_position_data.sort_values('Timestamp'),
        on="Timestamp", direction="nearest"
    )
    
    # Merge torque_data into merged_data (include both raw and filtered torque and moment components)
    merged_data = pd.merge_asof(
        merged_data.sort_values('Timestamp'),
        torque_data.sort_values('Timestamp')[['Timestamp', 'Torque Magnitude', 'Filtered Torque Magnitude',
                                                'Tx', 'Ty', 'Tz', 'Filtered Tx', 'Filtered Ty', 'Filtered Tz',
                                                'Rotated Tx', 'Rotated Ty']],
        on="Timestamp", direction="nearest"
    )
    
    # Use the same Y Position column (as offset)
    merged_data['Y Position'] = merged_data['Y Position']
    
    return merged_data

# Process all datasets
processed_datasets = [process_dataset(path) for path in file_paths]

# Create common time axis
time = processed_datasets[0]['Timestamp'].sort_values().drop_duplicates()

# ----- Reindex filtered data -----
aligned_force = [df.set_index('Timestamp')['Filtered Force Magnitude'].reindex(time).interpolate() for df in processed_datasets]
aligned_rotated_fx = [df.set_index('Timestamp')['Rotated Fx'].reindex(time).interpolate() for df in processed_datasets]
aligned_rotated_fy = [df.set_index('Timestamp')['Rotated Fy'].reindex(time).interpolate() for df in processed_datasets]
aligned_fz = [df.set_index('Timestamp')['Filtered Fz'].reindex(time).interpolate() for df in processed_datasets]

aligned_y = [df.set_index('Timestamp')['Y Position'].reindex(time).interpolate() for df in processed_datasets]
aligned_torque = [df.set_index('Timestamp')['Filtered Torque Magnitude'].reindex(time).interpolate() for df in processed_datasets]
aligned_mx = [df.set_index('Timestamp')['Filtered Tz'].reindex(time).interpolate() for df in processed_datasets]

# ----- Reindex rotated torque data -----
aligned_rotated_tx = [df.set_index('Timestamp')['Rotated Tx'].reindex(time).interpolate() for df in processed_datasets]
aligned_rotated_ty = [df.set_index('Timestamp')['Rotated Ty'].reindex(time).interpolate() for df in processed_datasets]

# ----- Reindex raw data -----
aligned_raw_force = [df.set_index('Timestamp')['Force Magnitude'].reindex(time).interpolate() for df in processed_datasets]
aligned_raw_y = [df.set_index('Timestamp')['Y Position'].reindex(time).interpolate() for df in processed_datasets]
aligned_raw_torque = [df.set_index('Timestamp')['Torque Magnitude'].reindex(time).interpolate() for df in processed_datasets]

# Compute means and standard deviations (filtered)
mean_force = pd.concat(aligned_force, axis=1).mean(axis=1)
std_force = pd.concat(aligned_force, axis=1).std(axis=1)
mean_rotated_fx = pd.concat(aligned_rotated_fx, axis=1).mean(axis=1)
mean_rotated_fy = pd.concat(aligned_rotated_fy, axis=1).mean(axis=1)
mean_fz = pd.concat(aligned_fz, axis=1).mean(axis=1)
mean_y = pd.concat(aligned_y, axis=1).mean(axis=1)
std_y = pd.concat(aligned_y, axis=1).std(axis=1)
mean_torque = pd.concat(aligned_torque, axis=1).mean(axis=1)
std_torque = pd.concat(aligned_torque, axis=1).std(axis=1)
mean_rotated_tx = pd.concat(aligned_rotated_tx, axis=1).mean(axis=1)
mean_rotated_ty = pd.concat(aligned_rotated_ty, axis=1).mean(axis=1)
mean_mx = pd.concat(aligned_mx, axis=1).mean(axis=1)

# Compute means and standard deviations (raw)
mean_raw_force = pd.concat(aligned_raw_force, axis=1).mean(axis=1)
std_raw_force = pd.concat(aligned_raw_force, axis=1).std(axis=1)
mean_raw_y = pd.concat(aligned_raw_y, axis=1).mean(axis=1)
std_raw_y = pd.concat(aligned_raw_y, axis=1).std(axis=1)
mean_raw_torque = pd.concat(aligned_raw_torque, axis=1).mean(axis=1)
std_raw_torque = pd.concat(aligned_raw_torque, axis=1).std(axis=1)

# =======================================================
# Figure 1: Raw Force and Raw Y Position
# =======================================================
fig1, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12))  # Two rows in Figure 1

# --- Top subplot: Filtered Force (with Rotated Fx, Rotated Fy, Fz) and Y Position ---
ax1.plot(mean_force.index.to_numpy(), mean_force.to_numpy(), label="Filtered Force Mag", color='blue')
ax1.fill_between(mean_force.index.to_numpy(), 
                 (mean_force - std_force).to_numpy(), 
                 (mean_force + std_force).to_numpy(), 
                 color='blue', alpha=0.3, label="Force Mag Std Dev")
ax1.plot(mean_rotated_fx.index.to_numpy(), mean_rotated_fx.to_numpy(), label="Rotated Fx", color='cyan', linestyle='--')
ax1.plot(mean_rotated_fy.index.to_numpy(), mean_rotated_fy.to_numpy(), label="Rotated Fy", color='magenta', linestyle='--')
ax1.plot(mean_fz.index.to_numpy(), mean_fz.to_numpy(), label="Filtered Fz", color='brown', linestyle='--')
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force (N)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim([-20, 20])
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)

ax1b = ax1.twinx()
ax1b.plot(mean_y.index.to_numpy(), mean_y.to_numpy(), label="Y Position", color='red')
ax1b.fill_between(mean_y.index.to_numpy(), 
                  (mean_y - std_y).to_numpy(), 
                  (mean_y + std_y).to_numpy(), 
                  color='red', alpha=0.3, label="Y Pos Std Dev")
ax1b.set_ylabel("Y Position (m)", color='red')
ax1b.tick_params(axis='y', labelcolor='red')
ax1b.set_ylim([-0.5, 0.5])
ax1b.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_title("Figure 1: Filtered Force (with Rotated Fx, Rotated Fy, Fz) and Y Position")
lines1, labels1 = ax1.get_legend_handles_labels()
lines1b, labels1b = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper right')

# --- Bottom subplot: Raw Force and Raw Y Position ---
ax3.plot(mean_raw_force.index.to_numpy(), mean_raw_force.to_numpy(), label="Raw Force Mag", color="green")
ax3.fill_between(mean_raw_force.index.to_numpy(), 
                 mean_raw_force.to_numpy() - std_raw_force.to_numpy(), 
                 mean_raw_force.to_numpy() + std_raw_force.to_numpy(),
                 color="green", alpha=0.3, label="Raw Force Std Dev")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Force (N)", color="green")
ax3.tick_params(axis="y", labelcolor="green")
ax3.set_ylim([-20, 20])
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)

ax3b = ax3.twinx()
ax3b.plot(mean_raw_y.index.to_numpy(), mean_raw_y.to_numpy(), label="Raw Y Position", color="red", linestyle="--")
ax3b.fill_between(mean_raw_y.index.to_numpy(), 
                  mean_raw_y.to_numpy() - std_raw_y.to_numpy(), 
                  mean_raw_y.to_numpy() + std_raw_y.to_numpy(),
                  color="red", alpha=0.3, label="Raw Y Std Dev")
ax3b.set_ylabel("Y Position (m)", color="red")
ax3b.tick_params(axis='y', labelcolor="red")
ax3b.set_ylim([-0.5, 0.5])
ax3b.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title("Figure 1: Raw Force and Raw Y Position")
lines3, labels3 = ax3.get_legend_handles_labels()
lines3b, labels3b = ax3b.get_legend_handles_labels()
ax3.legend(lines3 + lines3b, labels3 + labels3b, loc="upper right")

# =======================================================
# Figure 2: Torque and Y Position (Raw Torque Plots)
# =======================================================
fig2, (ax5, ax7) = plt.subplots(2, 1, figsize=(12, 12))  # Two rows in Figure 2

# --- Top subplot: Filtered Torque (with Rotated Tx, Rotated Ty, Tz) and Y Position ---
ax5.plot(mean_torque.index.to_numpy(), mean_torque.to_numpy(), label="Filtered Torque Mag", color="purple")
ax5.fill_between(mean_torque.index.to_numpy(),
                 (mean_torque - std_torque).to_numpy(),
                 (mean_torque + std_torque).to_numpy(),
                 color="purple", alpha=0.3, label="Torque Mag Std Dev")
# Plot rotated Tx and Ty instead of filtered Tx and Ty
ax5.plot(mean_rotated_tx.index.to_numpy(), mean_rotated_tx.to_numpy(), label="Rotated Tx", color="olive", linestyle="--")
ax5.plot(mean_rotated_ty.index.to_numpy(), mean_rotated_ty.to_numpy(), label="Rotated Ty", color="navy", linestyle="--")
ax5.plot(mean_mx.index.to_numpy(), mean_mx.to_numpy(), label="Filtered Tz", color="teal", linestyle="--")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Torque (Nm)", color="purple")
ax5.tick_params(axis='y', labelcolor="purple")
ax5.axhline(0, color='gray', linestyle='--', linewidth=1)

ax5b = ax5.twinx()
ax5b.plot(mean_y.index.to_numpy(), mean_y.to_numpy(), label="Y Position", color='red')
ax5b.fill_between(mean_y.index.to_numpy(), 
                  (mean_y - std_y).to_numpy(), 
                  (mean_y + std_y).to_numpy(), 
                  color='red', alpha=0.3, label="Y Pos Std Dev")
ax5b.set_ylabel("Y Position (m)", color='red')
ax5b.tick_params(axis='y', labelcolor='red')
ax5b.set_ylim([-0.5, 0.5])
ax5b.axhline(0, color='gray', linestyle='--', linewidth=1)
ax5.set_title("Figure 2: Filtered Torque (with Rotated Tx, Rotated Ty, Tz) and Y Position")
ax5.set_ylim([-4, 4])
lines5, labels5 = ax5.get_legend_handles_labels()
lines5b, labels5b = ax5b.get_legend_handles_labels()
ax5.legend(lines5 + lines5b, labels5 + labels5b, loc='upper right')

# --- Bottom subplot: Raw Torque and Raw Y Position ---
ax7.plot(mean_raw_torque.index.to_numpy(), mean_raw_torque.to_numpy(), label="Raw Torque Mag", color="orange")
ax7.fill_between(mean_raw_torque.index.to_numpy(),
                 mean_raw_torque.to_numpy() - std_raw_torque.to_numpy(),
                 mean_raw_torque.to_numpy() + std_raw_torque.to_numpy(),
                 color="orange", alpha=0.3, label="Raw Torque Std Dev")
ax7.set_xlabel("Time (s)")
ax7.set_ylabel("Torque (Nm)", color="orange")
ax7.tick_params(axis="y", labelcolor="orange")
ax7.axhline(0, color='gray', linestyle='--', linewidth=1)

ax7b = ax7.twinx()
ax7b.plot(mean_raw_y.index.to_numpy(), mean_raw_y.to_numpy(), label="Raw Y Position", color="red", linestyle="--")
ax7b.fill_between(mean_raw_y.index.to_numpy(),
                  mean_raw_y.to_numpy() - std_raw_y.to_numpy(),
                  mean_raw_y.to_numpy() + std_raw_y.to_numpy(),
                  color="red", alpha=0.3, label="Raw Y Std Dev")
ax7b.set_ylabel("Y Position (m)", color="red")
ax7b.tick_params(axis='y', labelcolor="red")
ax7b.set_ylim([-0.5, 0.5])
ax7b.axhline(0, color='gray', linestyle='--', linewidth=1)
ax7.set_title("Figure 2: Raw Torque and Raw Y Position")
ax7.set_ylim([-4, 4])
lines7, labels7 = ax7.get_legend_handles_labels()
lines7b, labels7b = ax7b.get_legend_handles_labels()
ax7.legend(lines7 + lines7b, labels7 + labels7b, loc="upper right")

# Show both figures (they will appear in separate windows)
plt.show()