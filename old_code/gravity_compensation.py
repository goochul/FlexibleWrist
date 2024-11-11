"""Example script of moving robot joint positions."""
import os 
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    args = parser.parse_args()
    return args


def compute_errors(pose_1, pose_2):

    pose_a = (
        pose_1[:3]
        + transform_utils.quat2axisangle(np.array(pose_1[3:]).flatten()).tolist()
    )
    pose_b = (
        pose_2[:3]
        + transform_utils.quat2axisangle(np.array(pose_2[3:]).flatten()).tolist()
    )
    return np.abs(np.array(pose_a) - np.array(pose_b))


def osc_move(robot_interface, controller_type, controller_cfg, target_pose, num_steps):
    target_pos, target_quat = target_pose
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    current_rot, current_pos = robot_interface.last_eef_rot_and_pos

    # List to store all positions
    positions = []

    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        positions.append(current_pos.flatten())  # Save the position

        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = (target_pos - current_pos).flatten() * 10

        action_axis_angle = axis_angle_diff.flatten() * 1
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]

        #print("Starting GC mode")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0])
        print(current_pose)

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )

    return positions  # Return the list of positions


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_delta_pose,
    num_steps,
    num_additional_steps,
    interpolation_method,
    type='euler'
):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    current_ee_pose = robot_interface.last_eef_pose
    current_pos = current_ee_pose[:3, 3:]
    current_rot = current_ee_pose[:3, :3]
    
    if type == 'euler':
        target_delta_pos, target_delta_euler = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )
        target_delta_rot = R.from_euler('XYZ', target_delta_euler).as_matrix()
        target_rot = np.dot(target_delta_rot, current_rot)
        target_quat = R.from_matrix(target_rot).as_quat()
        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
        
        # Get positions during movement
        positions = osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps,
        )
        positions += osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_additional_steps,
        )
    else:
        # Implementation for other interpolation types if needed
        positions = []

    return positions  # Return all positions

def get_unique_filename(base_name, extension='png'):
    """
    Generate a unique filename by appending a number if a file already exists.
    """
    directory = "figure/Traj/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    i = 1
    while os.path.exists(f"{directory}{base_name}{i}.{extension}"):
        i += 1
    return f"{directory}{base_name}{i}.{extension}"

def save_trajectory_data(positions, filename):
    np.savetxt(filename, positions, delimiter=",")
    print(f"Trajectory data saved as {filename}")


def plot_trajectory(positions):
    # Create a larger figure for better visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions = np.array(positions)
    num_points = len(positions)

    # Create a color map for the trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))

    # Plot the trajectory with color gradient
    for i in range(num_points - 1):
        ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2], c=colors[i])

    # Set plot labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('End-Effector Trajectory with Color Gradient')

    # Ensure the plot uses the same scale for all axes
    ax.set_box_aspect([1,1,1])

    plt.show()

    # Save the figure with a unique filename
    plot_filename = get_unique_filename('Traj', 'png')
    fig.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Save the trajectory data
    data_filename = get_unique_filename('TrajData', 'csv')
    save_trajectory_data(positions, data_filename)



def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_type = args.controller_type

    controller_cfg = get_default_controller_config(controller_type)

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    reset_joint_positions = [0.00560393,  0.92695904, -0.03755379, -1.98562784, -0.01500635,  4.457537,  0.78219662]
    
    #[0.08630862187752486, -0.2026771669594614, -0.018564449599894128, -2.477335820171727, -0.007616763467786644, 2.303372703009118, 0.852915178307475]
    '''
    [
        -0.4212149815583423, 
        0.7313779730679502, 
        0.46724951838342593, 
        -2.4031471454717646, 
        0.2897538532932723, 
        3.9199282876530517, 
        1.9819704166076382
        ]
    '''
    reset_joints_to(robot_interface, reset_joint_positions)

    current_ee_pose = robot_interface.last_eef_pose
    current_rot = current_ee_pose[:3, :3]
    current_euler_angle_degrees_intr = R.from_matrix(current_rot).as_euler('XYZ', degrees=True)
    current_euler_angle_degrees_extr = R.from_matrix(current_rot).as_euler('xyz', degrees=True)
    print("Start current_euler_angle intr: ", current_euler_angle_degrees_intr)
    print("Start current_euler_angle extr: ", current_euler_angle_degrees_extr)
    
    # Collect all trajectory points during movement
    positions = move_to_target_pose(
        robot_interface,
        controller_type,
        controller_cfg,
        target_delta_pose=[0.0, 0.0, 0.0, 0.18, 0.0, 0.0],
        num_steps=80,
        num_additional_steps=100,
        interpolation_method="linear",
        type="euler"
    )

    final_ee_pose = robot_interface.last_eef_pose
    final_rot = final_ee_pose[:3, :3]
    final_euler_angle_degrees_intr = R.from_matrix(final_rot).as_euler('XYZ', degrees=True)
    final_euler_angle_degrees_extr = R.from_matrix(final_rot).as_euler('xyz', degrees=True)
    print("final_euler_angle intr: ", final_euler_angle_degrees_intr)
    print("final_euler_angle extr: ", final_euler_angle_degrees_extr)

    robot_interface.close()

    # Plot the entire trajectory with color gradient and consistent scale
    plot_trajectory(positions)
    

if __name__ == "__main__":
    main()
