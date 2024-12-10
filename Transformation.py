import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

# Define the transformation matrix function using DH parameters
def dh_transform(a, d, alpha, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
        [0, 0, 0, 1]
    ])

# Forward Kinematics to compute end-effector position and orientation
def forward_kinematics(joint_angles, dh_params):
    T = np.eye(4)
    for (a, d, alpha), theta in zip(dh_params, joint_angles):
        T = T @ dh_transform(a, d, alpha, theta)
    return T

# Define DH parameters for each joint
dh_params = [
    (0, 0.333, 0),           # Joint 1
    (0, 0, -np.pi / 2),      # Joint 2
    (0, 0.316, np.pi / 2),   # Joint 3
    (0.0825, 0, np.pi / 2),  # Joint 4
    (-0.0825, 0.384, -np.pi / 2),  # Joint 5
    (0, 0, np.pi / 2),       # Joint 6
    (0.088, 0, np.pi / 2),   # Joint 7
    (0, 0.107, 0)            # Flange
]

# Initial joint values (theta) in radians
initial_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]

# [ 0.00596583,  0.81711726, -0.03810693, -2.05191146, -0.01553237,  4.41286068,  0.7810791 ]
#[0.00783327163909093,    0.5558480613472636, -0.04036106289390794, -2.163110868612272, -0.015487256765292425, 4.260755619281773, 0.7779242274028955]

# Step 1: Calculate initial end-effector position and orientation
initial_T = forward_kinematics(initial_joint_positions, dh_params)
print(f"initial T: {initial_T}")
initial_position = initial_T[:3, 3]
initial_orientation = R.from_matrix(initial_T[:3, :3]).as_euler('xyz', degrees=False)

print(f"Initial End-Effector Position: {initial_position}")
print(f"Initial End-Effector Orientation (radians): {initial_orientation}")

# Assuming initial_T is the final transformation matrix of the end-effector
rotation_matrix = initial_T[:3, :3]

# Calculate the rotation angle between the end-effector and the base frame
trace_R = np.trace(rotation_matrix)
rotation_angle = np.arccos((trace_R - 1) / 2)

print(f"Rotation Angle between End-Effector and Base Frame (radians): {rotation_angle}")
print(f"Rotation Angle between End-Effector and Base Frame (degrees): {np.degrees(rotation_angle)}")

# Step 2: Desired end-effector position (move -0.1m in z-direction)
desired_position = initial_position + np.array([-0.0, 0.00, 0.005])
desired_orientation = initial_orientation

print(f"desired position: {desired_position}")
print(f"Desired orientation: {desired_orientation}")

# Inverse Kinematics Cost Function
def ik_cost_function(joint_angles, target_position, target_orientation, dh_params, weight_position=1.0, weight_orientation=0.1):
    T = forward_kinematics(joint_angles, dh_params)
    position_error = np.linalg.norm(T[:3, 3] - target_position)
    orientation_error = np.linalg.norm(R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False) - target_orientation)
    return weight_position * position_error + weight_orientation * orientation_error

# Step 3: Use optimization to solve inverse kinematics
result = minimize(
    ik_cost_function,
    initial_joint_positions,  # Initial guess
    args=(desired_position, desired_orientation, dh_params),
    method='Nelder-Mead',  # Changed to Nelder-Mead, gradient-free method
    options={'disp': True, 'maxiter': 1000}  # Increased max iterations
)


# Extract the optimized joint angles
if result.success:
    optimized_joint_positions = result.x
    formatted_positions = [f"{angle:.4f}" for angle in optimized_joint_positions]
    print(f"Optimized Joint Positions: [{', '.join(formatted_positions)}]")
else:
    print("Inverse kinematics optimization failed.")

# Calculate the final end-effector position and orientation using the optimized joint positions
final_T = forward_kinematics(optimized_joint_positions, dh_params)
final_position = final_T[:3, 3]
final_orientation = R.from_matrix(final_T[:3, :3]).as_euler('xyz', degrees=False)

print(f"Final End-Effector Position: {final_position}")
print(f"Final End-Effector Orientation (radians): {final_orientation}")
