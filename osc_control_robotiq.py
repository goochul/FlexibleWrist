"""Example script of moving robot joint positions."""
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
    """
    Moves the end-effector toward the target pose using an OSC controller.
    This modified version ignores any rotational command by forcing the 
    rotational component to zero. Thus, the robot only moves in translation.
    """
    target_pos, target_quat = target_pose
    # target_axis_angle is computed here only for debugging purposes.
    target_axis_angle = transform_utils.quat2axisangle(target_quat)
    current_rot, current_pos = robot_interface.last_eef_rot_and_pos

    for _ in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        current_pos = current_pose[:3, 3:]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)
        if np.dot(target_quat, current_quat) < 0.0:
            current_quat = -current_quat
        quat_diff = transform_utils.quat_distance(target_quat, current_quat)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        # Normally, one would compute the rotational error as:
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        # However, we override the rotational command to zero.
        # action_axis_angle = np.zeros(3)
        
        # Compute translational command:
        action_pos = (target_pos - current_pos).flatten() * 5
        action_pos = np.clip(action_pos, -1.0, 1.0)

        action_axis_angle = axis_angle_diff.flatten() * 5
        action_axis_angle = np.clip(action_axis_angle, -0.6, 0.6)

        # Construct the full action:
        action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
        print(np.round(action, 2))
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
    return action


def move_to_target_pose(
    robot_interface,
    controller_type,
    controller_cfg,
    target_delta_pose,
    num_steps,
    num_additional_steps,
    interpolation_method,
    type='quaternion'
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
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps,
        )
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_additional_steps,
        )
    else:
        target_delta_pos, target_delta_axis_angle = (
            target_delta_pose[:3],
            target_delta_pose[3:],
        )
        current_ee_pose = robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos
        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle
        print("target_axis_angle, current_axis_angle:", np.degrees(np.array(target_axis_angle)), np.degrees(np.array(current_axis_angle)))
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        print("TARGET QUAT: ", target_quat)
        print('TARGET AXIS: ', np.degrees(transform_utils.quat2axisangle(target_quat)))
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_steps,
        )
        osc_move(
            robot_interface,
            controller_type,
            controller_cfg,
            (target_pos, target_quat),
            num_additional_steps,
        )


def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_type = args.controller_type

    controller_cfg = get_default_controller_config(controller_type)
    # Wait until a robot state is received:
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    reset_joint_positions = [-0.0896, 0.8878, -0.0601, -2.0298, -0.1237, 4.5136, 0.8201]
    reset_joints_to(robot_interface, reset_joint_positions)

    current_ee_pose = robot_interface.last_eef_pose
    current_rot = current_ee_pose[:3, :3]
    current_euler_angle_degrees_intr = R.from_matrix(current_rot).as_euler('XYZ', degrees=True)
    current_euler_angle_degrees_extr = R.from_matrix(current_rot).as_euler('xyz', degrees=True)
    print("Start current_euler_angle intr: ", current_euler_angle_degrees_intr)
    print("Start current_euler_angle extr: ", current_euler_angle_degrees_extr)
    
    move_to_target_pose(
        robot_interface,
        controller_type,
        controller_cfg,
        target_delta_pose=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        num_steps=100,
        num_additional_steps=30,
        interpolation_method="linear",
        type="quaternion"
    )

    final_ee_pose = robot_interface.last_eef_pose
    final_rot = final_ee_pose[:3, :3]
    final_euler_angle_degrees_intr = R.from_matrix(final_rot).as_euler('XYZ', degrees=True)
    final_euler_angle_degrees_extr = R.from_matrix(final_rot).as_euler('xyz', degrees=True)
    print("final_euler_angle intr: ", final_euler_angle_degrees_intr)
    print("final_euler_angle extr: ", final_euler_angle_degrees_extr)

    move_to_target_pose(
        robot_interface,
        controller_type,
        controller_cfg,
        target_delta_pose=[-0.15, 0.0, -0.12, 0.0, 0.0, 0.0],
        num_steps=100,
        num_additional_steps=30,
        interpolation_method="linear",
        type="quaternion"
    )



    reset_joints_to(robot_interface, reset_joint_positions)

    robot_interface.close()


if __name__ == "__main__":
    main()
