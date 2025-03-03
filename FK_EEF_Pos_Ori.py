import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import pandas as pd

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

# -------------------------------
# Parse the joint_positions.csv file
# -------------------------------
joint_positions_path = Path('data/20250302/150200/joint_positions.csv')
joint_angles_list = []

with joint_positions_path.open('r') as f:
    for line in f:
        # Remove whitespace and any wrapping quotes.
        line = line.strip().strip('"')
        # Only parse lines that start with '['
        if line.startswith('['):
            try:
                # Parse the string into a list of floats.
                joint_angles = [float(x) for x in ast.literal_eval(line)]
                joint_angles_list.append(joint_angles)
            except Exception as e:
                print(f"Error parsing line: {line}\n{e}")
        else:
            print(f"Skipping line: {line}")

if not joint_angles_list:
    raise ValueError("No valid joint angles were parsed from the CSV file.")

# Use the parsed joint positions to compute forward kinematics.
# -------------------------------
# Define DH parameters for 7 joints + flange
# -------------------------------
dh_params = np.array([
    [0,       0.333,  0],       # Joint 1
    [0,       0,     -np.pi/2],  # Joint 2
    [0,       0.316,  np.pi/2],   # Joint 3
    [0.0825,  0,      np.pi/2],   # Joint 4
    [-0.0825, 0.384, -np.pi/2],   # Joint 5
    [0,       0,      np.pi/2],   # Joint 6
    [0.088,   0,      np.pi/2],   # Joint 7
    [0,       0.107,  0]          # Flange (no joint angle)
])

# -------------------------------
# Compute forward kinematics for each set of joint angles.
# -------------------------------
positions = []    # End-effector positions
orientations = [] # Euler angles: roll, pitch, yaw

for joint_angles in joint_angles_list:
    T_step = forward_kinematics(joint_angles, dh_params[:7, :])
    T_step = T_step @ dh_transform(dh_params[7, 0], dh_params[7, 1], dh_params[7, 2], 0.0)
    pos = T_step[0:3, 3]
    positions.append(pos)
    roll, pitch, yaw = rotation_matrix_to_euler_angles(T_step[:3, :3])
    orientations.append([roll, pitch, yaw])
    
positions = np.array(positions)
orientations = np.array(orientations)

# -------------------------------
# Load the y_position_data.csv file to obtain Timestamps.
# -------------------------------
y_position_path = Path('data/20250228/200924/y_position_data.csv')
# Assuming the CSV has headers like "Timestamp", "Y Position", "Event"
y_data_df = pd.read_csv(y_position_path)
timestamps = y_data_df["Timestamp"].values

# Check if the number of timestamps matches the computed steps.
if len(timestamps) != positions.shape[0]:
    print("Warning: Number of timestamps does not match number of joint steps. Using index values instead.")
    timestamps = np.arange(positions.shape[0])

# -------------------------------
# Plot the results using Timestamp as the x-axis.
# -------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot end-effector position (X, Y, Z)
ax1.plot(timestamps, positions[:, 0], label='X')
ax1.plot(timestamps, positions[:, 1], label='Y')
ax1.plot(timestamps, positions[:, 2], label='Z')
ax1.set_title("End-Effector Position vs Timestamp")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Position (m)")
ax1.legend()
ax1.grid(True)

# Plot end-effector orientation (roll, pitch, yaw)
ax2.plot(timestamps, orientations[:, 0], label='Roll')
ax2.plot(timestamps, orientations[:, 1], label='Pitch')
ax2.plot(timestamps, orientations[:, 2], label='Yaw')
ax2.set_title("End-Effector Orientation (Euler Angles) vs Timestamp")
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("Angle (rad)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()