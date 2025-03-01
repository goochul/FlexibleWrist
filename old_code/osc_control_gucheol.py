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
    target_pos, target_quat = target_pose
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
        axis_angle_diff = transform_utils.quat2axisangle(quat_diff)
        action_pos = (target_pos - current_pos).flatten() * 10
        action_axis_angle = axis_angle_diff.flatten() * 1
        action_pos = np.clip(action_pos, -1.0, 1.0)
        action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

        action = action_pos.tolist() + action_axis_angle.tolist() + [-1.0]
        # logger.info(f"Axis angle action {action_axis_angle.tolist()}")
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
        # print("target_delta_axis_angle: ", target_delta_axis_angle)
        # target_delta_pos, target_delta_euler_angle = (
        #     target_delta_pose[:3],
        #     target_delta_pose[3:],
        # )

        current_ee_pose = robot_interface.last_eef_pose
        current_pos = current_ee_pose[:3, 3:]
        current_rot = current_ee_pose[:3, :3]
        
        current_quat = transform_utils.mat2quat(current_rot)
        current_axis_angle = transform_utils.quat2axisangle(current_quat)
        # current_euler_angle = R.from_matrix(current_rot).as_euler('XYZ') 

        target_pos = np.array(target_delta_pos).reshape(3, 1) + current_pos

        # print(np.array(target_delta_axis_angle).shape, current_axis_angle.shape)
        # exit()
        target_axis_angle = np.array(target_delta_axis_angle) + current_axis_angle
        print("target_axis_angle, current_axis_angle:", np.degrees(np.array(target_axis_angle)), np.degrees(np.array(current_axis_angle)))
        # target_euler_angle = np.array(target_delta_euler_angle) + current_euler_angle
        # print("target_euler_angle, current_euler_angle: ", np.degrees(np.array(target_euler_angle)), np.degrees(np.array(current_euler_angle)))
        # target_quat = R.from_euler('XYZ', target_euler_angle).as_quat()

        # logger.info(f"Before conversion {target_axis_angle}")
        target_quat = transform_utils.axisangle2quat(target_axis_angle)
        
        # target_pose = target_pos.flatten().tolist() + target_quat.flatten().tolist()
        # if np.dot(target_quat, current_quat) < 0.0:
        #     current_quat = -current_quat
        # target_axis_angle = transform_utils.quat2axisangle(target_quat)
        # # logger.info(f"After conversion {target_axis_angle}")
        # current_axis_angle = transform_utils.quat2axisangle(current_quat)

        # start_pose = current_pos.flatten().tolist() + current_quat.flatten().tolist()

        print("TARGET QUAT: ", target_quat)
        print('TARGET AXIS: ', np.degrees(transform_utils.quat2axisangle(target_quat)))
        # print('TARGET EULER: ', transform_utils.quat2euler(target_quat))

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
    # print("controller_type, controller_cfg", controller_type, controller_cfg)

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    reset_joint_positions = [0.2221, 0.7754, 0.0982, -2.0070, 0.3110, 4.4065, 0.5720]

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
        target_delta_pose=[0.0, -0.2, 0.0, 0.0, 0.0, 0.0],
        num_steps=80,
        num_additional_steps=40,
        interpolation_method="linear",
        type="euler"
    )

    final_ee_pose = robot_interface.last_eef_pose
    final_rot = final_ee_pose[:3, :3]
    final_euler_angle_degrees_intr = R.from_matrix(final_rot).as_euler('XYZ', degrees=True)
    final_euler_angle_degrees_extr = R.from_matrix(final_rot).as_euler('xyz', degrees=True)
    print("final_euler_angle intr: ", final_euler_angle_degrees_intr)
    print("final_euler_angle extr: ", final_euler_angle_degrees_extr)
    # final_axis_angle_degrees = R.from_matrix(final_rot).as_rotvec(degrees=True)
    # print("================")
    # print("Final euler_angle: ", final_euler_angle_degrees)
    # print("Final axis_angle: ", final_axis_angle_degrees)

    # final_quat = transform_utils.mat2quat(final_rot)
    # print("ACtual final quat", final_quat)
    # print('Actual final axis: ', np.degrees(transform_utils.quat2axisangle(final_quat)))
    # final_axis_angle_degrees = transform_utils.quat2axisangle(final_quat)
    # print("Final axis_angle: ", np.degrees(np.array(final_axis_angle_degrees)))

    robot_interface.close()


if __name__ == "__main__":
    main()

