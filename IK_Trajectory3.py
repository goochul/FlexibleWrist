import numpy as np
import matplotlib.pyplot as plt

def dh_transform(a, d, alpha, theta):
    """
    Calculate the individual transformation matrix from DH parameters.
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0,           a],
        [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
        [0,                0,               0,                    1]
    ], dtype=float)
    return T

def forward_kinematics(joint_angles, dh_params):
    """
    Compute the forward kinematics for a given set of joint angles using DH parameters.
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
    """
    trace = np.trace(R)
    w = np.sqrt(max(0, 1 + trace)) / 2.0
    x = (R[2,1] - R[1,2]) / (4.0 * w)
    y = (R[0,2] - R[2,0]) / (4.0 * w)
    z = (R[1,0] - R[0,1]) / (4.0 * w)
    return np.array([w, x, y, z], dtype=float)

def quatmultiply(q1, q2):
    """
    Multiply two quaternions [w, x, y, z].
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
    """
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def compute_jacobian(joint_angles, dh_params):
    """
    Compute the geometric Jacobian for a manipulator given the current joint angles.
    """
    num_joints = len(joint_angles)
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
        origins[:, i]   = T[0:3, 3]

    # End-effector position
    o_n = T[0:3, 3]

    # Build the Jacobian
    J = np.zeros((6, num_joints))
    for i in range(num_joints):
        # Linear velocity part
        J[0:3, i] = np.cross(z_vectors[:, i], (o_n - origins[:, i]))
        # Angular velocity part
        J[3:6, i] = z_vectors[:, i]
    return J

def rotation_to_rpy(R):
    """
    Convert a 3x3 rotation matrix to Roll-Pitch-Yaw (RPY) angles (XYZ convention).
    """
    if abs(R[2,0]) < 1:
        pitch = np.arcsin(-R[2,0])
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
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
    dh_params = np.array([
        [0,       0.333,  0],       # Joint 1
        [0,       0,     -np.pi/2], # Joint 2
        [0,       0.316,  np.pi/2], # Joint 3
        [0.0825,  0,      np.pi/2], # Joint 4
        [-0.0825, 0.384, -np.pi/2], # Joint 5
        [0,       0,      np.pi/2], # Joint 6
        [0.088,   0,      np.pi/2], # Joint 7
        [0,       0.107,  0]        # Flange
    ])

    # Initial joint values (theta) in radians (7 joints only)
    initial_joint_positions = np.array([0.0023, 0.9816, -0.0015, -1.8090, 0.0050, 3.8799, 0.7467]
)

    # FW 0mm init
    #[0.3519, 0.6895, 0.1798, -2.0106, 0.5197, 4.3909, 0.4279]

    # Forward Kinematics for initial pose
    T = forward_kinematics(initial_joint_positions, dh_params[:7, :])
    T = T @ dh_transform(dh_params[7, 0], dh_params[7, 1], dh_params[7, 2], 0.0)

    current_position    = T[0:3, 3]
    current_orientation = T[0:3, 0:3]
    q_current = rotm2quat(current_orientation)

    # Move +x by 0.1 m in 10 steps
    # n_steps = 100
    # desired_positions = np.tile(current_position, (n_steps, 1))
    # desired_positions[:, 0] = np.linspace(current_position[0], 
    #                                       current_position[0] + 0.16, 
    #                                       n_steps)

    # # Move +Y by 0.1 m in 10 steps
    # n_steps = 100
    # desired_positions = np.tile(current_position, (n_steps, 1))
    # desired_positions[:, 1] = np.linspace(current_position[1], 
    #                                       current_position[1] - 0.35, 
    #                                       n_steps)
    
    # # Move +Y by 0.1 m in 10 steps
    # n_steps = 5
    # desired_positions = np.tile(current_position, (n_steps, 1))
    # desired_positions[:, 1] = np.linspace(current_position[1], 
    #                                       current_position[1] + 0.06311611, 
    #                                       n_steps)

    # Move +z by 0.1 m in 10 steps
    n_steps = 100
    desired_positions = np.tile(current_position, (n_steps, 1))
    desired_positions[:, 2] = np.linspace(current_position[2], 
                                          current_position[2] -0.04, 
                                          n_steps)
    # + 0.49

    # Desired orientation remains the same
    q_desired = q_current.copy()

    # Store the trajectory (position, orientation, and also joint angles)
    trajectory = [current_position.copy()]
    trajectory_ori = [rotation_to_rpy(current_orientation)]
    
    # Store the joint angles at each step
    joint_trajectory = []
    joint_trajectory.append(initial_joint_positions.copy())

    # Iterative IK variables
    joint_positions = initial_joint_positions.copy()

    for i in range(n_steps):
        desired_position = desired_positions[i, :]

        # IK loop (max 10 iterations per step)
        for _ in range(10):
            # Forward kinematics
            T_current = forward_kinematics(joint_positions, dh_params[:7, :])
            T_current = T_current @ dh_transform(dh_params[7, 0], 
                                                 dh_params[7, 1], 
                                                 dh_params[7, 2], 0.0)

            current_position    = T_current[0:3, 3]
            current_orientation = T_current[0:3, 0:3]

            # Position error (no zeroing out X,Z here)
            position_error = desired_position - current_position

            # Orientation error
            q_current = rotm2quat(current_orientation)
            q_error   = quatmultiply(q_desired, quatinv(q_current))
            orientation_error = q_error[1:4]  # x, y, z part

            # Combine errors
            full_error = np.concatenate([position_error, orientation_error])

            # Convergence check
            if np.linalg.norm(full_error) < 1e-4:
                break

            # Jacobian and update
            J = compute_jacobian(joint_positions, dh_params[:7, :])
            delta_theta = np.linalg.pinv(J) @ full_error
            joint_positions += delta_theta

        # After finishing the step, record end-effector and joint angles
        trajectory.append(current_position.copy())
        trajectory_ori.append(rotation_to_rpy(current_orientation))
        joint_trajectory.append(joint_positions.copy())

    # Convert lists to numpy arrays for plotting
    trajectory = np.array(trajectory)
    trajectory_ori = np.array(trajectory_ori)
    joint_trajectory = np.array(joint_trajectory)

    # ------------------------------
    # Plot Position and Orientation
    # ------------------------------
    fig, (ax_pos, ax_ori) = plt.subplots(1, 2, figsize=(12, 5))

    # 1) Position vs. Step
    steps = np.arange(len(trajectory))
    ax_pos.plot(steps, trajectory[:, 0], label='X')
    ax_pos.plot(steps, trajectory[:, 1], label='Y')
    ax_pos.plot(steps, trajectory[:, 2], label='Z')
    ax_pos.set_xlabel('Step')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.set_title('End-Effector Position vs. Step (not zeroed out)')
    ax_pos.grid(True)
    ax_pos.legend()

    # 2) Orientation (Roll, Pitch, Yaw) vs. Step
    roll  = trajectory_ori[:, 0]
    pitch = trajectory_ori[:, 1]
    yaw   = trajectory_ori[:, 2]
    ax_ori.plot(steps, roll,  label='Roll')
    ax_ori.plot(steps, pitch, label='Pitch')
    ax_ori.plot(steps, yaw,   label='Yaw')
    ax_ori.set_xlabel('Step')
    ax_ori.set_ylabel('Angle (rad)')
    ax_ori.set_title('End-Effector Orientation (RPY) vs. Step')
    ax_ori.grid(True)
    ax_ori.legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------
    # Print or otherwise inspect joint angles
    # --------------------------------
    # print("Joint positions per step (radians):")
    # for i, jp in enumerate(joint_trajectory):
    #     print(f"{jp}")

    # print("Joint positions per step (radians):")
    # for i, jp in enumerate(joint_trajectory):
    #     # Format each joint angle with 4 decimal places, for example
    #     jp_str = ", ".join([f"{val:.4f}" for val in jp])
    #     print(f"[{jp_str}]")

    print("Joint positions per step (radians):")
    for i, jp in enumerate(joint_trajectory):
        jp_str = ", ".join([f"{val:.4f}" for val in jp])
        print(f"[{jp_str}],")

if __name__ == "__main__":
    main()
