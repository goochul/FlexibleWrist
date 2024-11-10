import os
import sys
import cv2
import numpy as np
import threading
import time
import subprocess
import matplotlib.pyplot as plt
from ForceSensor import ForceSensor
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger
import argparse

logger = get_deoxys_example_logger()

# Global variables
force_data = []
z_positions = []
timestamps = []
global_start_time = None
force_sensor = None
initial_z_position = None  # Store the initial Z-position for offsetting
max_samples = 1000  # Maximum number of samples to read
force_threshold = 20  # Force threshold for gravity compensation
torque_threshold = 7  # Torque threshold for gravity compensation

# Event signals
stop_threads = threading.Event()  # Control thread termination
movement_done = threading.Event()
recording_done = threading.Event()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    return parser.parse_args()

# FT Sensor Functions
def calculate_force_offset(sensor, num_samples=100, sleep_time=0.001):
    readings = []
    for _ in range(num_samples):
        force = sensor.get_force_obs()
        readings.append(force)
        time.sleep(sleep_time)
    return np.mean(readings, axis=0)

def initialize_force_sensor(calibrate=True, predefined_bias=np.zeros(3)):
    global force_sensor
    if calibrate:
        initial_sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3))
        initial_sensor.force_sensor_setup()
        force_offset = calculate_force_offset(initial_sensor)
        print("Calculated force offset:", force_offset)
    else:
        force_offset = predefined_bias
        print("Using predefined force offset:", force_offset)

    force_sensor = ForceSensor("/dev/ttyUSB0", force_offset)
    force_sensor.force_sensor_setup()

# Continuously read force sensor data with relative time
def read_ft_sensor_data():
    global force_data, global_start_time, force_sensor
    while len(force_data) < max_samples and not stop_threads.is_set():
        force = force_sensor.get_force_obs()  # Read only force data
        elapsed_time = time.time() - global_start_time

        # Calculate Euclidean norm for force
        force_magnitude = np.linalg.norm(force)  # Norm of force [Fx, Fy, Fz]

        # Check if force magnitude exceeds the threshold
        if force_magnitude > force_threshold:
            print("Force threshold exceeded. Stopping all actions and triggering gravity compensation.")
            print(f"Force magnitude: {force_magnitude:.2f} N")
            stop_threads.set()  # Stop robot movements
            movement_done.set()  # Signal that movement should stop
            recording_done.set()  # Stop video recording
            break  # Exit force monitoring loop

        force_data.append((elapsed_time, force_magnitude))  # Store force magnitude only
        time.sleep(0.01)

# Video recording function
def record_video(output_path, duration=30, fps=60):
    cap = cv2.VideoCapture(0)  # Open the default camera
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    start_time = time.time()

    while time.time() - start_time < duration and not recording_done.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    print("Video recording finished.")

# Robot Control Functions
def get_end_effector_position(robot_interface):
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def move_to_position(robot_interface, target_positions, controller_cfg):
    global initial_z_position  # Make this accessible globally to offset Z-position
    action = list(target_positions) + [-1.0]
    start_time = time.time()
    while True:
        if stop_threads.is_set():
            break  # Stop movement if stop_threads is set
        if len(robot_interface._state_buffer) > 0:
            current_ee_position = get_end_effector_position(robot_interface)

            if initial_z_position is None:
                initial_z_position = current_ee_position[2]  # Capture the initial Z-position

            z_positions.append(current_ee_position[2] - initial_z_position)  # Offset Z-position
            timestamps.append(time.time() - global_start_time)
            if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))) < 1e-3 or (time.time() - start_time > 10):
                break
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)
        time.sleep(0.05)

def control_robot_movement(robot_interface, controller_cfg):
    reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342] 
    des_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342] 
    # Movement sequence
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_threads.is_set():
        return  # Stop movement if stop_threads is set
    time.sleep(1)
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    if stop_threads.is_set():
        return
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    movement_done.set()  # Signal that movement is done

# Plot Force Magnitude and Z-Position data in a single figure at the end
def plot_merged_data():
    fig, ax1 = plt.subplots()

    # Plot Force magnitude
    if force_data:
        times, force_magnitudes = zip(*force_data)
        ax1.plot(times, force_magnitudes, label="Force Magnitude", color='tab:blue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Magnitude (N)", color='tab:blue')
    ax1.set_xlim([0, 17.5])
    ax1.set_ylim([-25, 25])  # Set y-axis limits for force magnitudes
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Plot Z-position data on secondary y-axis
    if timestamps:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, z_positions, label="Z Position", color='tab:red', marker='o', markersize=4)  # Adjusted marker size
        ax2.set_ylabel("End-Effector Z Position (m)", color='tab:red')
        ax2.set_ylim([-0.05, 0.05])  # Set y-axis limits for Z-position after offsetting
        ax2.legend(loc="upper right")

    plt.title("Force Magnitude and Z Position of End-Effector Over Time")
    plt.show()

# Main function
def main():
    global global_start_time

    # Set up calibration and bias
    calibration_flag = False
    predefined_bias = np.array([3, 8.5, 2.8])

    # Initialize force sensor
    initialize_force_sensor(calibrate=calibration_flag, predefined_bias=predefined_bias)

    # Initialize robot interface
    args = parse_args()
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    # Set global start time for all data collection
    global_start_time = time.time()

    # Start FT sensor reading thread
    sensor_thread = threading.Thread(target=read_ft_sensor_data, name="ForceSensorDataThread", daemon=True)
    sensor_thread.start()

    # Start video recording thread
    video_thread = threading.Thread(target=record_video, args=("output_video.avi",), name="VideoRecordingThread", daemon=True)
    video_thread.start()

    # Robot movement in main thread
    control_robot_movement(robot_interface, controller_cfg)

    # Wait for threads to finish
    sensor_thread.join()
    movement_done.wait()
    recording_done.set()  # Stop the video recording thread
    video_thread.join()

    # Close robot interface and plot merged data after movement
    robot_interface.close()
    plot_merged_data()  # Display merged force magnitude and Z-position plot

    # Start gravity compensation if threshold was exceeded
    if stop_threads.is_set():
        subprocess.Popen(["python3", "gravity_compensation_overload.py"])
        sys.exit()  # Immediately terminate the current script after plotting

if __name__ == "__main__":
    main()
