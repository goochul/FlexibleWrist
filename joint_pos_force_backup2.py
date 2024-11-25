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
max_samples = 1000
force_threshold = 50
torque_threshold = 5
force_max = 20  # Set the force_max threshold here
loading_flag = False
unloading_flag = False


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
        sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
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
    global force_data, torque_data, z_positions, timestamps, global_start_time
    print("Starting monitoring thread.")
    
    handling_threshold = False  # Flag to indicate if threshold handling is in progress

    try:
        while len(force_data) < max_samples and not stop_monitoring.is_set():
            # Read adjusted force-torque data
            raw_force, raw_torque = sensor.get_force_obs()
            adjusted_force = raw_force - force_offset
            adjusted_torque = raw_torque - torque_offset

            # Log time
            timestamps.append(time.time() - global_start_time)

            # Calculate magnitudes
            force_magnitude = np.linalg.norm(adjusted_force)
            torque_magnitude = np.linalg.norm(adjusted_torque)

            # Append force and torque data
            force_data.append(force_magnitude)
            torque_data.append(torque_magnitude)

            # Append Z position data (retrieve end-effector position)
            try:
                current_pose = robot_interface.last_eef_pose  # Retrieve end-effector pose
                z_positions.append(current_pose[2, 3])        # Extract Z position
            except Exception as e:
                print(f"Error retrieving Z position: {e}")
                z_positions.append(np.nan)  # Append NaN if retrieval fails

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




# Video recording function using RealSense camera
def record_video(output_path, duration=30, fps=30):
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

# Modified move_to_position to ensure tuple storage in z_positions
def move_to_position(robot_interface, target_positions, controller_cfg):
    global initial_z_position, z_positions, timestamps, loading_flag
    action = list(target_positions) + [-1.0]
    start_time = time.time()

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

            current_z_position = current_ee_position[2] - initial_z_position

            # Record Z position, timestamp, and event based on the loading flag
            timestamps.append(time.time() - global_start_time)
            z_event = 1 if loading_flag else 2
            z_positions.append((current_z_position, z_event))  # Always store tuples

            # Check if the robot is close enough to the target positions
            position_error = np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))
            # if np.max(position_error) < 1e-3 or (time.time() - start_time > 30):
            if (time.time() - start_time > 30):
                break

        # Send control action
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)

        # Check again if stop signal is set after sending commands
        if stop_movement.is_set():
            print("Movement interrupted after command.")
            break

        time.sleep(0.01)

# Updated joint_position_control to handle event setting
def joint_position_control(robot_interface, controller_cfg):
    global loading_flag
    reset_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]
    des_joint_positions = [-0.0075636, 0.486079, -0.0250772, -2.182928, -0.0263943, 4.2597242, 0.76971342]

    # Move to initial position
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(1)

    # Toggle loading flag and mark event as `1` for loading phase
    loading_flag = True
    z_positions.append((0.0, 1))  # Mark the event as loading
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return

    # Toggle loading flag and mark event as `2` for unloading phase
    loading_flag = False
    z_positions.append((0.0, 2))  # Mark the event as unloading
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

    # Synchronize list lengths
    min_length = min(len(timestamps), len(force_data))
    timestamps_synced = timestamps[:min_length]
    force_data_synced = force_data[:min_length]

    # Create DataFrame for force data
    force_df = pd.DataFrame({
        "Timestamp": timestamps_synced,
        "Force Magnitude": force_data_synced
    })
    force_df.to_csv(os.path.join(data_folder, "force_data.csv"), index=False)

    # Create DataFrame for Z-position data
    z_positions_cleaned = [(pos[0], pos[1]) if isinstance(pos, tuple) else (pos, None) for pos in z_positions]
    z_pos_df = pd.DataFrame({
        "Timestamp": timestamps[:len(z_positions_cleaned)],
        "Z Position": [pos[0] for pos in z_positions_cleaned],
        "Event": [pos[1] for pos in z_positions_cleaned]
    })
    z_pos_df.to_csv(os.path.join(data_folder, "z_position_data.csv"), index=False)

    # Save torque data
    if torque_data:
        torque_df = pd.DataFrame({
            "Timestamp": timestamps[:len(torque_data)],  # Sync length to torque_data
            "Torque Magnitude": torque_data
        })
        torque_df.to_csv(os.path.join(data_folder, "torque_data.csv"), index=False)

    print(f"Data saved to folder: {data_folder}")
    return data_folder


# Plot Data without Display
def plot_merged_data(data_folder):
    # Synchronize lengths of timestamps and force_data
    min_length = min(len(timestamps), len(force_data))
    timestamps_synced = timestamps[:min_length]
    force_data_synced = force_data[:min_length]

    fig, ax1 = plt.subplots()

    # Plot Force Magnitude
    if force_data_synced:
        ax1.plot(timestamps_synced, force_data_synced, label="Force Magnitude", color='tab:blue')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Force Magnitude (N)", color='tab:blue')
        ax1.set_xlim([0, max(timestamps_synced)])
        ax1.set_ylim([-25, 25])
        ax1.legend(loc="upper left")
        ax1.grid(True)

    # Plot Z Position
    if z_positions:
        # Synchronize z_positions to timestamps_synced
        z_positions_synced = z_positions[:min_length]
        print(len(z_positions), len(timestamps_synced), len(z_positions_synced))
        if len(z_positions_synced) == len(timestamps_synced):
            ax2 = ax1.twinx()
            ax2.plot(timestamps_synced, z_positions_synced, label="Z Position", color='tab:red', marker='o', markersize=2)
            ax2.set_ylabel("End-Effector Z Position (m)", color='tab:red')
            ax2.set_ylim([min(z_positions_synced) - 0.01, max(z_positions_synced) + 0.01])
            ax2.legend(loc="upper right")
        else:
            print(f"Skipping Z Position plot: z_positions length ({len(z_positions)}) does not match timestamps_synced.")

    # Plot Torque Magnitude
    if torque_data:
        torque_data_synced = torque_data[:len(timestamps_synced)]  # Sync with timestamps
        ax1.plot(timestamps_synced, torque_data_synced, label="Torque Magnitude", color='tab:green', linestyle='--')
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

    # Start video recording thread
    video_output_path = os.path.join(data_folder, "realsense_recording.mp4")
    video_thread = threading.Thread(target=record_video, args=(video_output_path, 60, 30), daemon=True)
    video_thread.start()

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
    video_thread.join()  # Ensure video thread finishes

    print(f"Length of timestamps: {len(timestamps)}")
    print(f"Length of force_data: {len(force_data)}")

    # Save and plot data after threads finish
    data_folder = save_data_to_csv()
    plot_merged_data(data_folder)

    print("Process complete.")

if __name__ == "__main__":
    main()
