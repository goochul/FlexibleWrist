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
initial_joint_positions = [-0.0087831, 0.3709803, -0.0241358, -2.1980871, -0.0297141, 4.1597863, 0.7708481]

# Joint limits
joint_limits_lower = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
joint_limits_upper = [ 2.8973,  1.7628,  2.8973,  3.0718,  2.8973,  3.7525,  2.8973]
bounds = [(low, high) for low, high in zip(joint_limits_lower, joint_limits_upper)]

# Step 1: Calculate initial end-effector position and orientation
initial_T = forward_kinematics(initial_joint_positions, dh_params)
initial_position = initial_T[:3, 3]
initial_orientation = R.from_matrix(initial_T[:3, :3]).as_euler('xyz', degrees=False)

print(f"Initial End-Effector Position: {initial_position}")
print(f"Initial End-Effector Orientation (radians): {initial_orientation}")

# Desired end-effector position and orientation
desired_position = initial_position + np.array([-0.01, 0.00, 0.0])  # Move +0.01m in y-direction
desired_orientation = initial_orientation  # Maintain current orientation

position_tolerance = 0.005  # Allowable position error in meters
orientation_tolerance = 0.01  # Allowable orientation error in radians

weight_position = 1.0
weight_orientation = 0.5

def ik_cost_function(joint_angles, target_position, dh_params, weight_position=1.0):
    """
    Cost function for inverse kinematics to prioritize position over orientation.
    """
    T = forward_kinematics(joint_angles, dh_params)

    # Position error
    position_error = np.linalg.norm(T[:3, 3] - target_position)

    # Debugging output
    # print(f"Joint Angles: {joint_angles}")
    # print(f"Position Error: {position_error}")

    # Return position error only, ignoring orientation
    return weight_position * position_error

# Desired end-effector position (keep orientation flexible)
desired_position = initial_position + np.array([0.0, -0.01, 0.0])  # Move +0.01m in y-direction

# Use optimization to solve inverse kinematics
result = minimize(
    ik_cost_function,
    initial_joint_positions,
    args=(desired_position, dh_params),
    method='SLSQP',
    bounds=bounds,  # Add bounds to limit joint angles
    options={'disp': True, 'maxiter': 2000, 'ftol': 1e-6}  # Increased max iterations and tighter tolerance
)

# Extract the optimized joint angles and calculate final position
if result.success:
    optimized_joint_positions = result.x
    formatted_positions = [f"{angle:.7f}" for angle in optimized_joint_positions]
    print(f"Optimized Joint Positions: [{', '.join(formatted_positions)}]")

    final_T = forward_kinematics(optimized_joint_positions, dh_params)
    final_position = final_T[:3, 3]
    final_orientation = R.from_matrix(final_T[:3, :3]).as_euler('xyz', degrees=False)

    print(f"Final End-Effector Position: {final_position}")
    print(f"Final End-Effector Orientation (radians): {final_orientation}")
else:
    print("Inverse kinematics optimization failed.")
