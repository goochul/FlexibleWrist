import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
import numpy as np
import ast

# ======================================================
# Define new groups with updated dataset paths
# ======================================================
groups = {
    "Rigid - 0mm space": [
        Path('data/20250302/161651/'),
        Path('data/20250302/161811/'),
        Path('data/20250302/161919/')
    ],
    "Rigid - 5mm space": [
        Path('data/20250302/161111/'),
        Path('data/20250302/161218/'),
        Path('data/20250302/161327/')
    ],
    "Rigid - 10mm space": [
        Path('data/20250302/160440/'),
        Path('data/20250302/160706/'),
        Path('data/20250302/160814/')
    ],
    "Rigid - 14mm space": [
        Path('data/20250302/204348/'),
        Path('data/20250302/204537/'),
        Path('data/20250302/204700/')
    ],
    "Rigid - 15mm space": [
        Path('data/20250302/202941/')
    ],
    "Rigid - 20mm space": [
        Path('data/20250302/162555/')
    ]
}

# ======================================================
# Low-pass filter function
# ======================================================
def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ======================================================
# Process a dataset: Read force and torque data, filter, and merge by Timestamp.
# ======================================================
def process_dataset(file_path):
    force_data = pd.read_csv(file_path / 'force_data.csv')
    torque_data = pd.read_csv(file_path / 'torque_data.csv')
    
    # Round timestamps for alignment
    force_data['Timestamp'] = force_data['Timestamp'].round(2)
    torque_data['Timestamp'] = torque_data['Timestamp'].round(2)
    
    # Sampling settings
    sampling_frequency = 30
    cutoff_frequency = 0.8

    # Apply low-pass filter to Force and Torque Magnitudes
    force_data['Filtered Force Magnitude'] = low_pass_filter(
        force_data['Force Magnitude'], cutoff_frequency, sampling_frequency
    )
    torque_data['Filtered Torque Magnitude'] = low_pass_filter(
        torque_data['Torque Magnitude'], cutoff_frequency, sampling_frequency
    )
    
    # Merge force and torque data on Timestamp (using nearest match)
    merged = pd.merge_asof(
        force_data.sort_values('Timestamp')[['Timestamp', 'Filtered Force Magnitude']],
        torque_data.sort_values('Timestamp')[['Timestamp', 'Filtered Torque Magnitude']],
        on="Timestamp", direction="nearest"
    )
    return merged

# ======================================================
# FK functions
# ======================================================
def dh_transform(a, d, alpha, theta):
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0,           a],
        [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
        [0,                0,               0,                    1]
    ], dtype=float)
    return T

def forward_kinematics(joint_angles, dh_params):
    T_final = np.eye(4)
    for i in range(len(joint_angles)):
        a = dh_params[i, 0]
        d = dh_params[i, 1]
        alpha = dh_params[i, 2]
        theta = joint_angles[i]
        T_final = T_final @ dh_transform(a, d, alpha, theta)
    return T_final

def rotation_matrix_to_euler_angles(R):
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw

# ======================================================
# Compute Y-displacement via FK from joint_positions.csv.
# ======================================================
def compute_y_displacement(rep_path):
    joint_positions_path = rep_path / 'joint_positions.csv'
    joint_angles_list = []
    with joint_positions_path.open('r') as f:
        for line in f:
            line = line.strip().strip('"')
            if line.startswith('['):
                try:
                    joint_angles = [float(x) for x in ast.literal_eval(line)]
                    joint_angles_list.append(joint_angles)
                except Exception as e:
                    print(f"Error parsing line: {line}\n{e}")
            else:
                continue
    if not joint_angles_list:
        raise ValueError(f"No valid joint angles parsed from {joint_positions_path}")

    # Define DH parameters for 7 joints + flange
    dh_params = np.array([
        [0,       0.333,  0],       # Joint 1
        [0,       0,     -np.pi/2],  # Joint 2
        [0,       0.316,  np.pi/2],   # Joint 3
        [0.0825,  0,      np.pi/2],   # Joint 4
        [-0.0825, 0.384, -np.pi/2],   # Joint 5
        [0,       0,      np.pi/2],   # Joint 6
        [0.088,   0,      np.pi/2],   # Joint 7
        [0,       0.107,  0]          # Flange
    ])
    
    positions = []
    for joint_angles in joint_angles_list:
        T_step = forward_kinematics(joint_angles, dh_params[:7, :])
        T_step = T_step @ dh_transform(dh_params[7, 0], dh_params[7, 1], dh_params[7, 2], 0.0)
        pos = T_step[0:3, 3]
        positions.append(pos)
    positions = np.array(positions)
    # Extract Y-displacement (2nd coordinate)
    y_disp = positions[:, 1]
    return y_disp

# ======================================================
# Process each group and build results
# ======================================================
group_results = {}  # Will store: representative y_disp, mean & std for force & torque

for group_name, paths in groups.items():
    print(f"Processing group: {group_name}")
    rep_path = paths[0]  # Use the first dataset for FK
    try:
        y_disp = compute_y_displacement(rep_path)
    except Exception as e:
        print(f"Error computing FK for group {group_name}: {e}")
        continue
    
    # Process each dataset in the group
    datasets = [process_dataset(p) for p in paths]
    common_time = datasets[0]['Timestamp'].sort_values().drop_duplicates()
    
    aligned_force = [df.set_index('Timestamp')['Filtered Force Magnitude'].reindex(common_time).interpolate() for df in datasets]
    aligned_torque = [df.set_index('Timestamp')['Filtered Torque Magnitude'].reindex(common_time).interpolate() for df in datasets]
    
    mean_force = pd.concat(aligned_force, axis=1).mean(axis=1)
    std_force = pd.concat(aligned_force, axis=1).std(axis=1)
    mean_torque = pd.concat(aligned_torque, axis=1).mean(axis=1)
    std_torque = pd.concat(aligned_torque, axis=1).std(axis=1)
    
    # Interpolate y_disp to match common_time if needed.
    if len(common_time) != len(y_disp):
        y_disp_interp = np.interp(common_time, np.linspace(common_time.iloc[0], common_time.iloc[-1], len(y_disp)), y_disp)
    else:
        y_disp_interp = y_disp

    # Apply vertical flip.
    y_disp_flip = -y_disp_interp

    group_results[group_name] = {
        "y_disp": y_disp_flip,
        "mean_force": mean_force,
        "std_force": std_force,
        "mean_torque": mean_torque,
        "std_torque": std_torque
    }

# ======================================================
# Apply additional shifts:
# Shift "Rigid - 50mm space" and "Rigid - 55mm space" by -0.02 (left)
# Shift all other groups by +0.03 (right)
# ======================================================
for group_name, results in group_results.items():
    if group_name in ["Rigid - 50mm space", "Rigid - 55mm space"]:
        results["y_disp"] = results["y_disp"] - 0.02
    else:
        results["y_disp"] = results["y_disp"] + 0.03

# ======================================================
# Final x-axis shift: move all x-axis values +0.01 m to the right
# ======================================================
for group_name, results in group_results.items():
    results["y_disp"] = results["y_disp"] + 0.01

# ======================================================
# Plotting: Merged Force and Torque Magnitude vs SHIFTED Y-Displacement
# ======================================================
fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(12, 10))

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
for idx, (group_name, results) in enumerate(group_results.items()):
    # Use the shifted y_disp as the x-axis.
    x_axis = results["y_disp"]
    
    # Truncate to match the length of force/torque data if necessary.
    n = min(len(x_axis), len(results["mean_force"]))
    x_axis = x_axis[:n]
    mean_force = results["mean_force"].to_numpy()[:n]
    std_force  = results["std_force"].to_numpy()[:n]
    mean_torque = results["mean_torque"].to_numpy()[:n]
    std_torque  = results["std_torque"].to_numpy()[:n]
    
    ax_force.plot(x_axis, mean_force, label=group_name, color=colors[idx])
    ax_force.fill_between(x_axis,
                          mean_force - std_force,
                          mean_force + std_force,
                          color=colors[idx], alpha=0.3)
    
    ax_torque.plot(x_axis, mean_torque, label=group_name, color=colors[idx])
    ax_torque.fill_between(x_axis,
                           mean_torque - std_torque,
                           mean_torque + std_torque,
                           color=colors[idx], alpha=0.3)

ax_force.set_xlabel("Y-Displacement (m)")
ax_force.set_ylabel("Force Magnitude (N)")
ax_force.set_title("Merged Filtered Force Magnitude vs Y-Displacement")
ax_force.legend(loc="upper right")
ax_force.grid(True)

ax_torque.set_xlabel("Y-Displacement (m)")
ax_torque.set_ylabel("Torque Magnitude (Nm)")
ax_torque.set_title("Merged Filtered Torque Magnitude vs Y-Displacement")
ax_torque.legend(loc="upper right")
ax_torque.grid(True)

plt.tight_layout()
plt.show()