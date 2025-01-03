import numpy as np
import matplotlib.pyplot as plt

def dh_transform(a, d, alpha, theta):
    """
    Calculate the individual transformation matrix from DH parameters.

    Parameters:
    -----------
    a     : float
        Link length (distance along the X-axis)
    d     : float
        Link offset (distance along the Z-axis)
    alpha : float
        Twist angle (rotation about X-axis)
    theta : float
        Joint angle (rotation about Z-axis)

    Returns:
    --------
    T : 4x4 numpy array
        The transformation matrix
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0,           a],
        [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
        [0,               0,               0,                     1]
    ], dtype=float)
    return T

def forward_kinematics(joint_angles, dh_params):
    """
    Compute the forward kinematics for a given set of joint angles using DH parameters.
    
    Parameters:
    -----------
    joint_angles : 1D array-like
        Joint angles in radians
    dh_params : 2D array-like
        DH parameters for each joint, shape = (N, 3) where columns are [a, d, alpha]

    Returns:
    --------
    T_final : 4x4 numpy array
        Overall transformation matrix from the base to the end-effector
    """
    T_final = np.eye(4)
    num_joints = len(joint_angles)
    for i in range(num_joints):
        a = dh_params[i, 0]
        d = dh_params[i, 1]
        alpha = dh_params[i, 2]
        theta = joint_angles[i]
        T_final = T_final @ dh_transform(a, d, alpha, theta)
    return T_final

def rotm2quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    Parameters:
    -----------
    R : 3x3 numpy array
        Rotation matrix

    Returns:
    --------
    q : 1D array, shape = (4,)
        Quaternion [w, x, y, z]
    """
    # To avoid numerical issues, use max() to ensure no negative values in sqrt
    trace = np.trace(R)
    w = np.sqrt(max(0, 1 + trace)) / 2.0
    x = (R[2,1] - R[1,2]) / (4.0 * w)
    y = (R[0,2] - R[2,0]) / (4.0 * w)
    z = (R[1,0] - R[0,1]) / (4.0 * w)
    return np.array([w, x, y, z], dtype=float)

def quatmultiply(q1, q2):
    """
    Multiply two quaternions.
    Quaternion format: [w, x, y, z]

    Parameters:
    -----------
    q1, q2 : 1D array-like
        Quaternions in [w, x, y, z] format

    Returns:
    --------
    q : 1D array, shape = (4,)
        Result of the quaternion multiplication
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def quatinv(q):
    """
    Inverse of a quaternion (assuming a unit quaternion).

    Parameters:
    -----------
    q : 1D array
        Quaternion [w, x, y, z]

    Returns:
    --------
    q_inv : 1D array
        Inverse of the quaternion
    """
    w, x, y, z = q
    # For a unit quaternion, inverse is [w, -x, -y, -z].
    return np.array([w, -x, -y, -z], dtype=float)

def compute_jacobian(joint_angles, dh_params):
    """
    Compute the geometric Jacobian for a manipulator given the current joint angles.
    
    Parameters:
    -----------
    joint_angles : 1D array-like
        Joint angles in radians
    dh_params : 2D array-like
        DH parameters for each joint, shape = (N, 3)

    Returns:
    --------
    J : 6xN numpy array
        Jacobian matrix
    """
    num_joints = len(joint_angles)
    # Storage for z-axes and origins for each link
    z_vectors = np.zeros((3, num_joints))
    origins = np.zeros((3, num_joints))

    T = np.eye(4)
    for i in range(num_joints):
        a = dh_params[i, 0]
        d = dh_params[i, 1]
        alpha = dh_params[i, 2]
        theta = joint_angles[i]
        T = T @ dh_transform(a, d, alpha, theta)

        z_vectors[:, i] = T[0:3, 2]
        origins[:, i] = T[0:3, 3]

    # End-effector position
    o_n = T[0:3, 3]

    # Build the Jacobian
    J = np.zeros((6, num_joints))
    for i in range(num_joints):
        # Cross product for linear part
        J[0:3, i] = np.cross(z_vectors[:, i], (o_n - origins[:, i]))
        # Angular part is just the axis
        J[3:6, i] = z_vectors[:, i]

    return J

def rotation_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to Roll-Pitch-Yaw (RPY) angles (XYZ convention).

    Parameters:
    -----------
    R : 3x3 numpy array
        Rotation matrix

    Returns:
    --------
    rpy : 1D array, shape = (3,)
        [roll, pitch, yaw] in radians
    """
    # Protect against numerical issues
    if abs(R[2,0]) < 1:
        pitch = np.arcsin(-R[2,0])
        roll = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock
        yaw = 0.0
        if R[2,0] <= -1:
            pitch = np.pi / 2
            roll = np.arctan2(R[0,1], R[0,2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0,1], -R[0,2])
    return np.array([roll, pitch, yaw], dtype=float)

def main():
    # Define DH parameters [a, d, alpha]
    # For example, a 7-DOF manipulator + a flange link
    dh_params = np.array([
        [0,       0.333,  0],      # Joint 1
        [0,       0,     -np.pi/2],# Joint 2
        [0,       0.316,  np.pi/2],# Joint 3
        [0.0825,  0,      np.pi/2],# Joint 4
        [-0.0825, 0.384, -np.pi/2],# Joint 5
        [0,       0,      np.pi/2],# Joint 6
        [0.088,   0,      np.pi/2],# Joint 7
        [0,       0.107,  0]       # Flange
    ])

    # Initial joint values (theta) in radians (7 joints only)
    initial_joint_positions = np.array([-0.0070,  0.3027, -0.0309, 
                                        -2.5290, -0.0224,  4.4196,  0.7622])

    # Forward Kinematics for initial pose (without the flange, then add flange)
    T = forward_kinematics(initial_joint_positions, dh_params[:7, :])
    T = T @ dh_transform(dh_params[7, 0], dh_params[7, 1], dh_params[7, 2], 0.0)

    current_position = T[0:3, 3]
    current_orientation = T[0:3, 0:3]

    # Convert current orientation to quaternion
    q_current = rotm2quat(current_orientation)

    # Define the full motion (e.g., 100 mm along Y-axis in 1 mm steps => 10 steps of 10 mm)
    # For demonstration, let's just do 10 steps from current_position(1) to current_position(1) + 0.1
    n_steps = 10
    desired_positions = np.tile(current_position, (n_steps, 1))  # replicate along rows
    desired_positions[:, 1] = np.linspace(current_position[1], current_position[1] + 0.1, n_steps)

    # Desired orientation remains the same
    q_desired = q_current

    # Initialize variables for the iterative IK
    joint_positions = initial_joint_positions.copy()
    trajectory = [current_position.copy()]  # store the trajectory of positions
    trajectory_ori = [rotation_to_rpy(current_orientation)]  # store orientation in RPY

    # Iterative IK for each step
    for i in range(n_steps):
        desired_position = desired_positions[i, :]

        # IK loop: up to 10 iterations
        for _ in range(10):
            # Forward kinematics with updated joint positions
            T_current = forward_kinematics(joint_positions, dh_params[:7, :])
            T_current = T_current @ dh_transform(dh_params[7, 0],
                                                 dh_params[7, 1],
                                                 dh_params[7, 2], 0.0)

            current_position = T_current[0:3, 3]
            current_orientation = T_current[0:3, 0:3]

            # Position error (only Y-axis is desired to move)
            position_error = desired_position - current_position
            # Zero out X and Z error to enforce strictly +Y motion
            position_error[[0, 2]] = 0.0

            # Orientation error using quaternion
            q_current = rotm2quat(current_orientation)
            q_error = quatmultiply(q_desired, quatinv(q_current))
            # The vector part of the quaternion (x, y, z)
            orientation_error = q_error[1:4]

            # Combine position and orientation error
            full_error = np.concatenate([position_error, orientation_error])

            # Check convergence
            if np.linalg.norm(full_error) < 1e-4:
                break

            # Compute Jacobian (only for the 7 joints)
            J = compute_jacobian(joint_positions, dh_params[:7, :])

            # Solve for joint updates: use pseudo-inverse
            # We only need a 6x7 Jacobian, so the result is 7x1
            delta_theta = np.linalg.pinv(J) @ full_error

            # Update joint positions
            joint_positions = joint_positions + delta_theta

        # After finishing for this step, store the current end-effector pose
        trajectory.append(current_position.copy())
        trajectory_ori.append(rotation_to_rpy(current_orientation))

    trajectory = np.array(trajectory)
    trajectory_ori = np.array(trajectory_ori)

    # Plot the trajectory in 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], '-o', linewidth=2, label='Trajectory')

    # Highlight the initial point in red
    ax.plot3D([trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
              'ro', markersize=8, markerfacecolor='r', label='Initial Point')

    # Configure labels, grid, etc.
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory (Strictly +Y Motion with Quaternion Orientation)')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    plt.show()

if __name__ == "__main__":
    main()