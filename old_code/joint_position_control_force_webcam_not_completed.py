import os
import sys
import cv2
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
import pyrealsense2 as rs

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
max_samples = 2200
force_threshold = 50
torque_threshold = 5
force_max = 20  # Set the force_max threshold here

# Event signals
stop_movement = threading.Event()
stop_monitoring = threading.Event()
movement_done = threading.Event()
recording_done = threading.Event()

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    return parser.parse_args()

# FT Sensor Functions
def calibrate_force_sensor(sensor, num_samples=100, sleep_time=0.01):
    """
    Calibrate the force sensor by calculating force and torque offsets.
    """
    try:
        readings = []
        print("Performing calibration...")
        for _ in range(num_samples):
            force, torque = sensor.get_force_obs()
            readings.append((force, torque))
            time.sleep(sleep_time)

        # Calculate offsets
        force_offset = np.mean([r[0] for r in readings], axis=0)
        torque_offset = np.mean([r[1] for r in readings], axis=0)
        print(f"Calibration complete. Force offset: {force_offset}, Torque offset: {torque_offset}")
        return force_offset, torque_offset
    except Exception as e:
        print(f"Error during calibration: {e}")
        return None, None

def initialize_force_sensor_for_calibration():
    """
    Initialize the sensor for calibration without applying offsets.
    """
    try:
        sensor = ForceSensor("/dev/ttyUSB1", np.zeros(3), np.zeros(3))
        sensor.force_sensor_setup()
        print("Sensor initialized for calibration.")
        return sensor
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        return None

def monitor_ft_sensor(robot_interface, joint_controller_cfg, osc_controller_type, osc_controller_cfg, sensor, force_offset, torque_offset):
    """
    Monitor the force-torque sensor with calibration offsets, handle thresholds, and perform gravity compensation when needed.
    """
    global force_data, torque_data, global_start_time
    print("Starting monitoring thread.")
    
    handling_threshold = False  # Flag to indicate if threshold handling is in progress

    try:
        while len(force_data) < max_samples and not stop_monitoring.is_set():
            # Read adjusted force-torque data
            raw_force, raw_torque = sensor.get_force_obs()
            adjusted_force = raw_force - force_offset
            adjusted_torque = raw_torque - torque_offset

            elapsed_time = time.time() - global_start_time
            force_magnitude = np.linalg.norm(adjusted_force)
            torque_magnitude = np.linalg.norm(adjusted_torque)

            # Log the data
            force_data.append((elapsed_time, force_magnitude))
            torque_data.append((elapsed_time, torque_magnitude))

            # Handle maximum force threshold
            if force_magnitude > force_max and not handling_threshold:
                print(f"Force exceeds maximum limit ({force_max} N). Returning to initial position.")
                handling_threshold = True  # Set flag to prevent re-triggering
                stop_movement.set()

                # Start a thread to handle returning to the initial position
                threading.Thread(
                    target=return_to_initial_position,
                    args=(robot_interface, joint_controller_cfg),
                    daemon=True
                ).start()
                stop_movement.clear()
                handling_threshold = False  # Reset flag after handling

            # Handle regular force/torque thresholds
            if not handling_threshold and (force_magnitude > force_threshold or torque_magnitude > torque_threshold):
                print("Threshold exceeded. Switching to gravity compensation.")
                stop_movement.set()
                threading.Thread(
                    target=perform_gravity_compensation,
                    args=(robot_interface, osc_controller_type, osc_controller_cfg),
                    daemon=True
                ).start()
                stop_monitoring.set()
                return

            # Short delay for smoother monitoring
            time.sleep(0.01)

    except Exception as e:
        print(f"Error in monitor_ft_sensor: {e}")


# Video recording function using RealSense camera (index 6)
def record_video_realsense(output_path, duration=30, fps=30):
    print("Recording from RealSense camera...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

    pipeline.start(config)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 540))
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            out.write(color_image)
            cv2.imshow('RealSense Camera', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

# Video recording function using Nexigo camera (index 0)
def record_video_nexigo(output_path, duration=60, fps=30, camera_index=0):
    print(f"Recording from Nexigo camera (/dev/video{camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open Nexigo camera at /dev/video{camera_index}.")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from Nexigo camera.")
                break
            out.write(frame)
            cv2.imshow('Nexigo Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# Function to return the robot to its initial position
def return_to_initial_position(robot_interface, controller_cfg):
    print("Returning to initial joint positions.")
    reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    print("Returning to initial position.")

# Robot Control Functions
def get_end_effector_position(robot_interface):
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def get_joint_data(robot_interface):
    return robot_interface._state_buffer[-1].q, robot_interface._state_buffer[-1].dq

# Add this global variable
event_markers = []

def move_to_position(robot_interface, target_positions, controller_cfg, event_label=None):
    global initial_z_position, event_markers
    action = list(target_positions) + [-1.0]
    start_time = time.time()

    if event_label:
        event_markers.append((time.time() - global_start_time, event_label))  # Log the event

    while True:
        # Check if the stop signal is set before and after sending commands
        if stop_movement.is_set():
            print("Movement stopped due to stop_movement event.")
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

            # Check if the robot is close enough to the target positions
            position_error = np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))
            if np.max(position_error) < 1e-3 or (time.time() - start_time > 30):
            # if (time.time() - start_time > 30):
                break

        # Send control action
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)

        # Check again if stop signal is set after sending commands
        if stop_movement.is_set():
            print("Movement interrupted after command.")
            break

        time.sleep(0.01)

def joint_position_control(robot_interface, controller_cfg):
    reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    # [-0.00757461,  0.47413217, -0.02512669, -2.18534287, -0.02667678,  4.2501711, 0.7698466 ]
    # [-0.00805785,  0.46225722, -0.0245614,  -2.18755885, -0.02669979,  4.24048583, 0.76958523]
    # [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    des_joint_positions = [-0.00749242,  0.54708303, -0.0248903,  -2.16802759, -0.02433914,  4.30569219,  0.76901974]

    # [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]              # Alimunum Frame origin for Panda
    # [-0.00767597,  0.51022177, -0.02485,    -2.17755938, -0.02581892,  4.27849113,  0.76947171]   # -10mm
    # [-0.00744893,  0.52245477, -0.02512409, -2.17452938, -0.02589844,  4.28777901,  0.76955813]   # -15mm
    # [-0.00764558,  0.534649,   -0.02463884, -2.17151983, -0.02343242,  4.29640372,  0.76849901]   # -20mm
    # [-0.00749242,  0.54708303, -0.0248903,  -2.16802759, -0.02433914,  4.30569219,  0.76901974]   # -25mm
    # [-0.00786796,  0.55953669, -0.0245075,  -2.16437121, -0.02514699,  4.31473024, 0.76914151]    # -30mm
    # [-0.0075991,   0.57211732, -0.02482249, -2.1605095,  -0.02561976,  4.32375554,  0.76977484]   # -35mm
    # [-0.00817004,  0.584347,   -0.02353005, -2.15728207, -0.01831063,  4.33053075,  0.76582103]   # -40mm
    # [-0.00817453,  0.58435545, -0.02352894, -2.15726601, -0.01829912,  4.33055562,  0.76575631]   # -45mm

    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(1)
    #event label = 1 => loading phase. 2 => unloading phase
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg, event_label="1")
    if stop_movement.is_set():
        return
    # move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    # move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    # move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg, event_label="2")
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

    z_pos_df = pd.DataFrame({"Timestamp": timestamps, "Z Position": z_positions, "Event": None})

    for timestamp, event in event_markers:
        closest_index = (z_pos_df["Timestamp"] - timestamp).abs().idxmin()  # Find the closest timestamp
        z_pos_df.loc[closest_index, "Event"] = event

    z_pos_df.to_csv(os.path.join(data_folder, "z_position_data.csv"), index=False)


    if joint_positions:
        num_joints = len(joint_positions[0]) if isinstance(joint_positions[0], (list, np.ndarray)) else 1
        joint_pos_df = pd.DataFrame(joint_positions, columns=[f"Joint {i+1} Position" for i in range(num_joints)])
        joint_pos_df.to_csv(os.path.join(data_folder, "joint_positions.csv"), index=False)

    if torque_data:
        torque_df = pd.DataFrame(torque_data, columns=["Timestamp", "Torque Magnitude"])
        torque_df.to_csv(os.path.join(data_folder, "torque_data.csv"), index=False)

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
    ax1.set_xlim([0, max(timestamps) if timestamps else 1])
    ax1.set_ylim([-25, 25])
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if timestamps:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, z_positions, label="Z Position", color='tab:red', marker='o', markersize=2)
        ax2.set_ylabel("End-Effector Z Position (m)", color='tab:red')
        ax2.set_ylim([-0.05, 0.05])
        ax2.legend(loc="upper right")

    if torque_data:
        times, torque_magnitudes = zip(*torque_data)  # Adjust for two elements
        ax1.plot(times, torque_magnitudes, label="Torque Magnitude", color='tab:green', linestyle='--')
        ax1.legend(loc="upper left")

    plt.title("Force Magnitude, Z Position of End-Effector, and Torque Magnitude Over Time")
    plot_path = os.path.join(data_folder, "plot.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def main():
    global global_start_time, force_sensor

    args = parse_args()

    # Initialize the sensor for calibration
    sensor = initialize_force_sensor_for_calibration()
    if sensor is None:
        print("Sensor initialization failed. Exiting...")
        return

    # Perform calibration
    force_offset, torque_offset = calibrate_force_sensor(sensor)
    if force_offset is None or torque_offset is None:
        print("Calibration failed. Exiting...")
        return

    # Begin robot interface setup
    try:
        robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
        joint_controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
        osc_controller_cfg = get_default_controller_config(args.controller_type)
        print("Robot interface initialized.")
    except Exception as e:
        print(f"Robot interface initialization failed: {e}")
        return

    global_start_time = time.time()

    # Create data folder path
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    # Start RealSense video recording thread
    realsense_video_path = os.path.join(data_folder, "realsense_recording.mp4")
    realsense_thread = threading.Thread(target=record_video_realsense, args=(realsense_video_path, 60, 30), daemon=True)
    realsense_thread.start()

    # Start Nexigo video recording thread
    nexigo_video_path = os.path.join(data_folder, "nexigo_recording.mp4")
    nexigo_thread = threading.Thread(target=record_video_nexigo, args=(nexigo_video_path, 60, 30, 0), daemon=True)
    nexigo_thread.start()

    # Start monitoring thread
    monitoring_thread = threading.Thread(
        target=monitor_ft_sensor,
        args=(robot_interface, joint_controller_cfg, args.controller_type, osc_controller_cfg, sensor, force_offset, torque_offset),
        daemon=True
    )
    monitoring_thread.start()

    # Start movement thread
    movement_thread = threading.Thread(target=joint_position_control, args=(robot_interface, joint_controller_cfg), daemon=True)
    movement_thread.start()

    # Wait for threads to finish
    monitoring_thread.join()
    movement_thread.join()
    realsense_thread.join()
    nexigo_thread.join()

    # Save and plot data after threads finish
    data_folder = save_data_to_csv()
    plot_merged_data(data_folder)

    print("Process complete.")

if __name__ == "__main__":
    main()