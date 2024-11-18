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
force_threshold = 30
torque_threshold = 3
force_max = 15  # Set the force_max threshold here

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
def calculate_force_offset(sensor, num_samples=100, sleep_time=0.001):
    """
    Calculate the offset for force and torque by averaging a number of samples.
    """
    readings = []
    print("Starting offset calculation...")
    for _ in range(num_samples):
        try:
            force, torque = sensor.get_force_obs()
            readings.append((force, torque))
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error reading sensor data during calibration: {e}")
            return None, None  # Return invalid offsets on error
    
    # Calculate the mean offsets
    force_offset = np.mean([reading[0] for reading in readings], axis=0)
    torque_offset = np.mean([reading[1] for reading in readings], axis=0)
    print(f"Calibration complete. Force offset: {force_offset}, Torque offset: {torque_offset}")
    return force_offset, torque_offset

def initialize_force_sensor(calibrate=True, predefined_bias=np.zeros(3)):
    """
    Initialize the force sensor and apply offsets.
    """
    global force_sensor
    try:
        print("Initializing force sensor...")
        if calibrate:
            # Perform calibration
            initial_sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
            initial_sensor.force_sensor_setup()
            force_offset, torque_offset = calculate_force_offset(initial_sensor)
            if force_offset is None or torque_offset is None:
                print("Calibration failed. Check the sensor or connection.")
                return  # Exit if calibration fails
        else:
            # Use predefined bias if no calibration is requested
            force_offset = predefined_bias
            torque_offset = np.zeros(3)

        # Initialize the sensor with calculated offsets
        force_sensor = ForceSensor("/dev/ttyUSB0", force_offset, torque_offset)
        force_sensor.force_sensor_setup()
        print("Force sensor initialized successfully.")
    except Exception as e:
        print(f"Force sensor initialization failed: {e}")

# Monitor force and torque sensor, handle force_max by returning to initial position, or gravity compensation if threshold is exceeded
def monitor_ft_sensor(robot_interface, joint_controller_cfg, osc_controller_type, osc_controller_cfg):
    global force_data, torque_data, global_start_time, force_sensor
    print("Starting force data reading thread...")

    while len(force_data) < max_samples and not stop_monitoring.is_set():
        try:
            # Read force and torque from the sensor
            force, torque = force_sensor.get_force_obs()
            elapsed_time = time.time() - global_start_time
            force_magnitude = np.linalg.norm(force)
            torque_magnitude = np.linalg.norm(torque)

            # Log force and torque data
            force_data.append((elapsed_time, force_magnitude))
            torque_data.append((elapsed_time, torque[0], torque[1], torque[2]))

            # Handle force_max: Return to initial position but pause further monitoring
            if force_magnitude > force_max:
                print(f"Force exceeds maximum limit ({force_max} N). Returning to initial position.")
                stop_movement.set()  # Stop ongoing movements
                return_to_initial_position(robot_interface, joint_controller_cfg)
                stop_movement.clear()  # Allow further movement
                print("Monitoring paused after returning to initial position.")
                time.sleep(1)  # Pause before resuming monitoring
                continue

            # Handle regular force/torque thresholds: Switch to gravity compensation
            if force_magnitude > force_threshold or torque_magnitude > torque_threshold:
                print("Threshold exceeded. Switching to gravity compensation.")
                stop_movement.set()  # Stop movement thread
                threading.Thread(
                    target=perform_gravity_compensation,
                    args=(robot_interface, osc_controller_type, osc_controller_cfg),
                    daemon=True
                ).start()
                stop_monitoring.set()  # Stop monitoring entirely
                return

            # Short delay for smoother monitoring
            time.sleep(0.01)

        except Exception as e:
            print(f"Error reading force sensor data: {e}")
            stop_monitoring.set()
            return


# Video recording function using RealSense camera
def record_video(output_path, duration=45, fps=30):
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
    print("Returned to initial position.")

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
            if np.max(position_error) < 1e-3 or (time.time() - start_time > 20):
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
    des_joint_positions = [-0.00786796,  0.55953669, -0.0245075,  -2.16437121, -0.02514699,  4.31473024, 0.76914151]

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
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)
    if stop_movement.is_set():
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
        initialize_force_sensor(calibrate=calibration_flag, predefined_bias=predefined_bias)
        print("Force sensor initialized.")
    except Exception as e:
        print(f"Force sensor initialization failed: {e}")
        return

    try:
        robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
        joint_controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
        osc_controller_cfg = get_default_controller_config(args.controller_type)
        print("Robot interface initialized.")
    except Exception as e:
        print(f"Robot interface initialization failed: {e}")
        return

    global_start_time = time.time()

    # Create data folder path with today's date and current time
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    # Start video recording thread
    video_output_path = os.path.join(data_folder, "realsense_recording.mp4")
    video_thread = threading.Thread(target=record_video, args=(video_output_path, 45, 30), daemon=True)
    video_thread.start()

    # Start force-torque sensor monitoring thread
    sensor_thread = threading.Thread(target=monitor_ft_sensor, args=(robot_interface, joint_controller_cfg, args.controller_type, osc_controller_cfg), daemon=True)
    sensor_thread.start()

    # Start movement thread
    movement_thread = threading.Thread(target=joint_position_control, args=(robot_interface, joint_controller_cfg), daemon=True)
    movement_thread.start()

    # Wait for threads to finish
    sensor_thread.join()
    video_thread.join()
    movement_done.wait()

    # Save data and plot merged data
    data_folder = save_data_to_csv()
    plot_merged_data(data_folder)

if __name__ == "__main__":
    main()
