"""Example script for using joint impedance control."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.experimental.motion_utils import joint_interpolation_traj
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import YamlConfig, get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

import os
import sys
import cv2
import threading
import time
import pandas as pd
import pyrealsense2 as rs
from ForceSensor import ForceSensor

logger = get_deoxys_example_logger()

# Global variables
force_data = []
torque_data = []
z_positions = []
joint_positions = []
joint_velocities = []
timestamps = []
global_start_time = None
force_sensor = None
initial_z_position = None
max_samples = 1000
force_threshold = 10
torque_threshold = 3

# Event signals
stop_movement = threading.Event()
stop_monitoring = threading.Event()
movement_done = threading.Event()
recording_done = threading.Event()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-impedance-controller.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    args = parser.parse_args()
    return args


# FT Sensor Functions
def calculate_force_offset(sensor, num_samples=100, sleep_time=0.001):
    readings = []
    for _ in range(num_samples):
        force, torque = sensor.get_force_obs()
        readings.append((force, torque))
        time.sleep(sleep_time)
    force_offset = np.mean([reading[0] for reading in readings], axis=0)
    torque_offset = np.mean([reading[1] for reading in readings], axis=0)
    return force_offset, torque_offset

def initialize_force_sensor(calibrate=True, predefined_bias=np.zeros(3)):
    global force_sensor
    try:
        if calibrate:
            print("Calibrating force sensor...")
            initial_sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
            initial_sensor.force_sensor_setup()
            force_offset, torque_offset = calculate_force_offset(initial_sensor)
            print("Calculated force offset:", force_offset)
            print("Calculated torque offset:", torque_offset)
        else:
            force_offset = predefined_bias
            torque_offset = np.zeros(3)
            print("Using predefined force and torque offsets:", force_offset, torque_offset)

        force_sensor = ForceSensor("/dev/ttyUSB0", force_offset, torque_offset)
        force_sensor.force_sensor_setup()
        print("Force sensor initialized.")
    except Exception as e:
        print(f"Force sensor initialization failed: {e}")


# Monitor force and torque sensor, switch to gravity compensation if threshold is exceeded
def monitor_ft_sensor(robot_interface, joint_controller_cfg, osc_controller_type, osc_controller_cfg):
    global force_data, torque_data, global_start_time, force_sensor
    print("Starting force data reading thread...")
    while len(force_data) < max_samples and not stop_monitoring.is_set():
        try:
            force, torque = force_sensor.get_force_obs()
            elapsed_time = time.time() - global_start_time
            force_magnitude = np.linalg.norm(force)
            torque_magnitude = np.linalg.norm(torque)

            if (force_magnitude > force_threshold) or (torque_magnitude > torque_threshold):
                print("Threshold exceeded. Switching to gravity compensation.")
                stop_movement.set()  # Stop movement thread
                threading.Thread(
                    target=perform_gravity_compensation,
                    args=(robot_interface, osc_controller_type, osc_controller_cfg),
                    daemon=True
                ).start()
                return

            force_data.append((elapsed_time, force_magnitude))
            torque_data.append((elapsed_time, torque[0], torque[1], torque[2]))
            time.sleep(0.001)
        except Exception as e:
            print(f"Error reading force sensor data: {e}")
            stop_monitoring.set()
            return

# Video recording function using RealSense camera
def record_video(output_path, duration=15, fps=30):
    print("Recording")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 540))

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Flip the frame vertically
            flipped_image = cv2.flip(color_image, 0)

            # Write the flipped frame to the video file
            out.write(flipped_image)

            # Display the flipped frame (optional)
            cv2.imshow('RealSense', flipped_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

# Robot Control Functions
def get_end_effector_position(robot_interface):
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def get_joint_data(robot_interface):
    return robot_interface._state_buffer[-1].q, robot_interface._state_buffer[-1].dq

def move_to_position(robot_interface, target_positions, controller_cfg):
    """
    Move the robot to a target joint position using impedance control.

    Args:
        robot_interface: The robot interface object.
        target_positions: The desired joint positions (array-like).
        controller_cfg: The configuration for the impedance controller.
    """
    global initial_z_position
    start_time = time.time()

    # Ensure robot_interface.last_q is valid
    last_q = robot_interface.last_q
    if last_q is None:
        raise ValueError("robot_interface.last_q is None. Cannot compute trajectory!")

    # Generate trajectory using joint interpolation
    joint_traj = joint_interpolation_traj(
        start_q=np.array(last_q),  # Starting joint positions
        end_q=np.array(target_positions)  # Target joint positions
    )

    # Execute trajectory
    for joint in joint_traj:
        if stop_movement.is_set():
            break  # Stop execution if stop signal is received

        # Log current robot state and update global variables
        if len(robot_interface._state_buffer) > 0:
            current_ee_position = get_end_effector_position(robot_interface)
            joint_pos, joint_vel = get_joint_data(robot_interface)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)

            if initial_z_position is None:
                initial_z_position = current_ee_position[2]

            z_positions.append(current_ee_position[2] - initial_z_position)
            timestamps.append(time.time() - global_start_time)

        # Send control action using impedance control
        action = joint.tolist() + [-1.0]  # Add flag for impedance control
        robot_interface.control(
            controller_type="JOINT_IMPEDANCE",
            action=action,
            controller_cfg=controller_cfg,
        )
        time.sleep(0.01)  # Small delay for smooth control

    # Check if the robot converged to the target positions
    final_error = np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))
    if np.max(final_error) >= 1e-3:
        print("Warning: Robot did not converge to the target positions.")
    else:
        print("Robot successfully moved to the target positions.")

def joint_position_control(robot_interface, controller_cfg):
    """
    Control the robot to move between predefined joint positions.

    Args:
        robot_interface: The robot interface object.
        controller_cfg: The configuration for the impedance controller.
    """
    # reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    # des_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]

    # reset_joint_positions = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.30396583422025, 0.8480939705504309]
    # des_joint_positions = [0.07898767752913056, -0.1430438676897629, -0.02619894482357678, -2.3945263182089045, -0.02561498419377851, 2.5206982383419305, 0.8344181147033158]

    reset_joint_positions = [-0.009733957813861346, 0.1461651583566534, -0.029233966668647162, -2.50168771573802, -0.018096381814583295, 4.2006328868590925, 0.8446265458692733]
    des_joint_positions = [-0.00963132,  0.19916105, -0.02885208, -2.50065448, -0.01226042,  4.25112287,  0.84279109]

    # Check if the robot is already at the reset positions
    current_joint_positions, _ = get_joint_data(robot_interface)
    position_error = np.linalg.norm(np.array(current_joint_positions) - np.array(reset_joint_positions))
    
    if position_error >= 3e-3:
        # Move to the reset positions
        move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
        if stop_movement.is_set():
            return

        time.sleep(1)
    else:
        print("Robot is already at the reset positions.")

    # Move to the desired positions
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return

    # Return to the reset positions
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)

    # Signal that movement is done
    movement_done.set()

# Gravity Compensation Function
def perform_gravity_compensation(robot_interface, controller_type, controller_cfg):
    print("Starting gravity compensation at the current position...")
    osc_move(robot_interface, controller_type, controller_cfg, num_steps=200)

def osc_move(robot_interface, controller_type, controller_cfg, num_steps, time_interval=0.01):
    for step in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        z_position = current_pose[2, 3]
        print(f"Step {step}, Current z-axis position: {z_position}")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0])
        robot_interface.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
        time.sleep(time_interval)

# Save Data to CSV
def save_data_to_csv():
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    force_df = pd.DataFrame(force_data, columns=["Timestamp", "Force Magnitude"])
    force_df.to_csv(os.path.join(data_folder, "force_data.csv"), index=False)

    z_pos_df = pd.DataFrame({"Timestamp": timestamps, "Z Position": z_positions})
    z_pos_df.to_csv(os.path.join(data_folder, "z_position_data.csv"), index=False)

    if joint_positions:
        num_joints = len(joint_positions[0]) if isinstance(joint_positions[0], (list, np.ndarray)) else 1
        joint_pos_df = pd.DataFrame(joint_positions, columns=[f"Joint {i+1} Position" for i in range(num_joints)])
        joint_pos_df.to_csv(os.path.join(data_folder, "joint_positions.csv"), index=False)

    print(f"Data saved to folder: {data_folder}")
    return data_folder

# Plot Data without Display
def plot_merged_data(data_folder):
    fig, ax1 = plt.subplots()

    if force_data:
        times, force_magnitudes = zip(*force_data)
        ax1.plot(times, force_magnitudes, label="Force Magnitude", color='tab:blue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Magnitude (N)", color='tab:blue')
    ax1.set_xlim([0, max(timestamps)])
    ax1.set_ylim([-25, 25])
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if timestamps:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, z_positions, label="Z Position", color='tab:red', marker='o', markersize=4)
        ax2.set_ylabel("End-Effector Z Position (m)", color='tab:red')
        ax2.set_ylim([-0.05, 0.05])
        ax2.legend(loc="upper right")

    if torque_data:
        times, torques_x, torques_y, torques_z = zip(*torque_data)
        norm_torques = [np.linalg.norm([tx, ty, tz]) for tx, ty, tz in zip(torques_x, torques_y, torques_z)]
        ax1.plot(times, norm_torques, label="Norm Torque", color='tab:green', linestyle='--')
        ax1.legend(loc="upper left")

    plt.title("Force Magnitude, Z Position of End-Effector, and Norm Torque Over Time")
    plot_path = os.path.join(data_folder, "plot.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

def main():
    args = parse_args()

    calibration_flag = True
    predefined_bias = np.array([3, 8.5, 2.8])

    try:
        initialize_force_sensor(calibrate=calibration_flag, predefined_bias=predefined_bias)
        print("Force sensor initialized.")
    except Exception as e:
        print(f"Force sensor initialization failed: {e}")
        return

    try:
        robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
        joint_controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
        controller_type = "JOINT_IMPEDANCE"
        osc_controller_cfg = get_default_controller_config(args.controller_type)
        print("Robot interface initialized.")
    except Exception as e:
        print(f"Robot interface initialization failed: {e}")
        return

    # Initialize global_start_time here
    global global_start_time
    global_start_time = time.time()

    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    # Create data folder path with today's date and current time
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    # Start video recording thread
    video_output_path = os.path.join(data_folder, "realsense_recording.mp4")
    video_thread = threading.Thread(target=record_video, args=(video_output_path, 15, 30), daemon=True)
    video_thread.start()

    # Start force-torque sensor monitoring thread
    sensor_thread = threading.Thread(target=monitor_ft_sensor, args=(robot_interface, joint_controller_cfg, args.controller_type, osc_controller_cfg), daemon=True)
    sensor_thread.start()

    # Start movement thread
    movement_thread = threading.Thread(target=joint_position_control, args=(robot_interface, joint_controller_cfg), daemon=True)
    movement_thread.start()
    print("movement thread started")

    # Wait for threads to finish
    sensor_thread.join()
    video_thread.join()
    movement_done.wait()

    # Save data and plot merged data
    data_folder = save_data_to_csv()
    plot_merged_data(data_folder)


if __name__ == "__main__":
    main()