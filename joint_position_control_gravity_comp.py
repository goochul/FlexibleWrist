import os
import sys
import numpy as np
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from ForceSensor import ForceSensor
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import YamlConfig, get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
import argparse

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
torque_threshold = 1

# Event signals
stop_threads = threading.Event()
movement_done = threading.Event()
data_saved = threading.Event()

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    return parser.parse_args()

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
    while len(force_data) < max_samples and not stop_threads.is_set():
        try:
            force, torque = force_sensor.get_force_obs()
            elapsed_time = time.time() - global_start_time
            force_magnitude = np.linalg.norm(force)
            torque_magnitude = np.linalg.norm(torque)

            if (force_magnitude > force_threshold) or (torque_magnitude > torque_threshold):
                print("Threshold exceeded. Switching to gravity compensation.")
                stop_threads.set()  # Stop position control
                perform_gravity_compensation(robot_interface, osc_controller_type, osc_controller_cfg)
                return

            force_data.append((elapsed_time, force_magnitude))
            torque_data.append((elapsed_time, torque[0], torque[1], torque[2]))
            time.sleep(0.001)
        except Exception as e:
            print(f"Error reading force sensor data: {e}")
            stop_threads.set()
            return

# Robot Control Functions
def get_end_effector_position(robot_interface):
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def get_joint_data(robot_interface):
    return robot_interface._state_buffer[-1].q, robot_interface._state_buffer[-1].dq

def move_to_position(robot_interface, target_positions, controller_cfg):
    global initial_z_position
    action = list(target_positions) + [-1.0]
    start_time = time.time()
    while True:
        if stop_threads.is_set():
            break
        if len(robot_interface._state_buffer) > 0:
            current_ee_position = get_end_effector_position(robot_interface)
            joint_pos, joint_vel = get_joint_data(robot_interface)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)

            if initial_z_position is None:
                initial_z_position = current_ee_position[2]

            z_positions.append(current_ee_position[2] - initial_z_position)
            timestamps.append(time.time() - global_start_time)
            if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))) < 1e-3 or (time.time() - start_time > 10):
                break
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)
        time.sleep(0.05)

def joint_position_control(robot_interface, controller_cfg):
    reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    des_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]

    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_threads.is_set():
        return
    time.sleep(1)
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    if stop_threads.is_set():
        return
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
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

# Main function
def main():
    global global_start_time, force_sensor

    args = parse_args()

    calibration_flag = True
    predefined_bias = np.array([3, 8.5, 2.8])

    try:
        print("Starting force sensor initialization...")
        initialize_force_sensor(calibrate=calibration_flag, predefined_bias=predefined_bias)
        print("Force sensor initialized.")
    except Exception as e:
        print(f"Force sensor initialization failed: {e}")
        return

    try:
        print("Initializing robot interface...")
        robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
        joint_controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
        osc_controller_cfg = get_default_controller_config(args.controller_type)
        print("Robot interface initialized.")
    except Exception as e:
        print(f"Robot interface initialization failed: {e}")
        return

    global_start_time = time.time()

    sensor_thread = threading.Thread(target=monitor_ft_sensor, args=(robot_interface, joint_controller_cfg, args.controller_type, osc_controller_cfg), daemon=True)
    sensor_thread.start()

    joint_position_control(robot_interface, joint_controller_cfg)

    sensor_thread.join()
    movement_done.wait()

if __name__ == "__main__":
    main()
