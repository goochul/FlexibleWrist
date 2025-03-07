#!/usr/bin/env python3
"""
Example script for using joint impedance control with end‐effector offset logging.
This script commands the robot to move from its current state to a target joint configuration.
It records the end‐effector position offsets (difference between current position and the initial position)
over time and then plots these offset values.
"""

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj, reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-impedance-controller.yml")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize robot interface and controller configuration
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_IMPEDANCE"

    # Define joint configurations for reset and target
    reset_joint_positions = [-0.0896, 0.8878, -0.0601, -2.0298, -0.1237, 4.5136, 0.8201]
    target_joint_positions = [-0.0889, 1.0518, -0.0441, -1.6214, -0.1159, 4.2690, 0.8367]

    # Wait until a robot state is available
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    # Retrieve the first available state and record the initial end-effector position.
    initial_pose = robot_interface.last_eef_pose
    initial_eef_position = initial_pose[:3, 3].copy()

    # Prepare lists to store the offsets and corresponding timestamps
    offsets = []      # List of offsets (each is a numpy array of shape (3,))
    timestamps = []   # Timestamps (in seconds)

    start_time = time.time()

    # Generate a joint trajectory from the current state to the target positions
    last_q = np.array(robot_interface.last_q)
    joint_traj = joint_interpolation_traj(start_q=last_q, end_q=target_joint_positions)

    # Execute the trajectory and record offsets
    for joint in joint_traj:
        action = joint.tolist() + [-1.0]
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        # A short sleep for stability
        time.sleep(0.01)
        current_pose = robot_interface.last_eef_pose  # 4x4 homogeneous transform
        current_pos = current_pose[:3, 3].copy()  # Current (X, Y, Z)
        # Compute offset relative to the initial position
        offset = current_pos - initial_eef_position
        offsets.append(offset)
        timestamps.append(time.time() - start_time)

    # (Optional) You may also execute a return-to-reset trajectory here.
    # last_q = np.array(robot_interface.last_q)
    # joint_traj = joint_interpolation_traj(start_q=last_q, end_q=reset_joint_positions)
    # for joint in joint_traj:
    #     action = joint.tolist() + [-1.0]
    #     robot_interface.control(
    #         controller_type=controller_type,
    #         action=action,
    #         controller_cfg=controller_cfg,
    #     )
    #     time.sleep(0.01)
    #     current_pose = robot_interface.last_eef_pose
    #     current_pos = current_pose[:3, 3].copy()
    #     offset = current_pos - initial_eef_position
    #     offsets.append(offset)
    #     timestamps.append(time.time() - start_time)

    # Close the robot interface after finishing the motion.
    robot_interface.close()

    # Convert offset list to a NumPy array for plotting
    offsets = np.array(offsets)  # shape (N, 3)

    # Plot the offsets over time (X, Y, and Z)
    plt.figure(figsize=(10, 6))
    # plt.plot(timestamps, offsets[:, 0], label="X Offset", color="red")
    # plt.plot(timestamps, offsets[:, 1], label="Y Offset", color="green")
    plt.plot(timestamps, offsets[:, 2], label="Z Offset", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("Offset (m)")
    plt.title("End-Effector Offset (from Initial Position) Over Time")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
