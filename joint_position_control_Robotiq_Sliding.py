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
from pyRobotiqGripper import RobotiqGripper


logger = get_deoxys_example_logger()

# Global variables
force_data = []
torque_data = []
y_positions = []
eef_positions = []    # <-- New global list to store full (x, y, z) end-effector positions
joint_positions = []
joint_velocities = []
timestamps = []
global_start_time = None
force_sensor = None
initial_z_position = None
initial_eef_position = None
max_samples = 1500
video_duration = 60
pressing_time_long = 12
pressing_time_short = 0.43
rs_camera_index = 6
Nexigo_camera_index = 0
force_threshold = 25
torque_threshold = 5
force_max = 25  # Set the force_max threshold here
eef_title = "End-Effector Positions (X, Y, Z), Sliding Task"
gripper_pos = 0

# max_samples = 3500 video_duration = 110  pressing_time = 5.5

# Event signals
stop_movement = threading.Event()
stop_monitoring = threading.Event()
movement_done = threading.Event()
recording_done = threading.Event()

# Parse command-line arguments (added options to enable/disable FT sensor and camera)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    # FT sensor toggle: default is disabled here. Use --enable-ft-sensor to enable.
    parser.add_argument("--enable-ft-sensor", dest="enable_ft_sensor", action="store_true", help="Enable force-torque sensor monitoring")
    parser.add_argument("--disable-ft-sensor", dest="enable_ft_sensor", action="store_false", help="Disable force-torque sensor monitoring")
    parser.set_defaults(enable_ft_sensor=True)
    # Camera toggle: default is disabled here. Use --enable-camera to enable.
    parser.add_argument("--enable-camera", dest="enable_camera", action="store_true", help="Enable camera recording")
    parser.add_argument("--disable-camera", dest="enable_camera", action="store_false", help="Disable camera recording")
    parser.set_defaults(enable_camera=True)
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

            # Log the data with extra values
            force_data.append((elapsed_time, *adjusted_force, force_magnitude))
            torque_data.append((elapsed_time, *adjusted_torque, torque_magnitude))

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
def record_video(output_path, duration, fps=30, camera_index=rs_camera_index):
    print(f"Recording video using camera index {camera_index}.")

    # Initialize VideoCapture with the specified camera index
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open the camera at index {camera_index}.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # # Flip the frame both vertically and horizontally
            # flipped_frame = cv2.flip(frame, -1)
            # flipped_frame = cv2.flip(frame, 1)
            flipped_frame = frame

            # Write the flipped frame to the video file
            out.write(flipped_frame)

            # Display the flipped frame (optional)
            cv2.imshow('Camera', flipped_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

# Function to return the robot to its initial position
def return_to_initial_position(robot_interface, controller_cfg):
    print("Returning to initial joint positions.")
    reset_joint_positions = [-0.0089260, 0.3819599, -0.0253966, -2.1973930, -0.0307321, 4.1700501, 0.7718912]
    move_to_position_long(robot_interface, np.array(reset_joint_positions), controller_cfg)
    print("Returning to initial position.")

# Robot Control Functions
def get_end_effector_position(robot_interface):
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def get_joint_data(robot_interface):
    return robot_interface._state_buffer[-1].q, robot_interface._state_buffer[-1].dq

# Global variable to mark events (used in plotting)
event_markers = []

def move_to_position_long(robot_interface, target_positions, controller_cfg, event_label=None):
    """
    Moves the robot toward the target joint positions. If the current configuration is far from the target 
    (for example, due to external interference), this function regenerates a smooth trajectory to slow down 
    the motion and avoid a sudden jump.
    """
    global initial_eef_position, eef_positions, event_markers
    action = list(target_positions) + [-1.0]
    start_time = time.time()

    if event_label:
        event_markers.append((time.time() - global_start_time, event_label))

    # Get the current joint position (if available)
    if len(robot_interface._state_buffer) > 0:
        current_joint_pos, _ = get_joint_data(robot_interface)
    else:
        current_joint_pos = target_positions  # fallback if no data is available

    while True:
        if stop_movement.is_set():
            print("Movement stopped due to stop_movement event.")
            break

        if len(robot_interface._state_buffer) > 0:
            current_ee_position = get_end_effector_position(robot_interface)  # shape: (3,)
            joint_pos, joint_vel = get_joint_data(robot_interface)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)

            # If we haven't stored the initial EEF position yet, store it now
            if initial_eef_position is None:
                initial_eef_position = current_ee_position.copy()

            # Compute the offset relative to the initial position
            offset_x = current_ee_position[0] - initial_eef_position[0]
            offset_y = current_ee_position[1] - initial_eef_position[1]
            offset_z = current_ee_position[2] - initial_eef_position[2]

            current_time = time.time() - global_start_time

            # Store offset (x, y, z) in eef_positions
            eef_positions.append((current_time, offset_x, offset_y, offset_z))

            # Also store the Z offset in y_positions, if desired
            y_positions.append(offset_y)
            timestamps.append(current_time)

            position_error = np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))
            if time.time() - start_time > pressing_time_long:
                print("Timeout reached. Breaking loop.")
                print(pressing_time_long)
                break
            if np.max(position_error) < 1e-4:
                print("Position error is small. Breaking loop.")
                break

        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)

        if stop_movement.is_set():
            print("Movement interrupted after command.")
            break

        time.sleep(0.005)


def move_to_position_short(robot_interface, target_positions, controller_cfg, event_label=None):
    """
    Moves the robot toward the target joint positions. If the current configuration is far from the target 
    (for example, due to external interference), this function regenerates a smooth trajectory to slow down 
    the motion and avoid a sudden jump.
    """
    global initial_eef_position, eef_positions, event_markers
    action = list(target_positions) + [-1.0]
    start_time = time.time()

    if event_label:
        event_markers.append((time.time() - global_start_time, event_label))

    # Get the current joint position (if available)
    if len(robot_interface._state_buffer) > 0:
        current_joint_pos, _ = get_joint_data(robot_interface)
    else:
        current_joint_pos = target_positions  # fallback if no data is available

    while True:
        if stop_movement.is_set():
            print("Movement stopped due to stop_movement event.")
            break

        if len(robot_interface._state_buffer) > 0:
            current_ee_position = get_end_effector_position(robot_interface)  # shape: (3,)
            joint_pos, joint_vel = get_joint_data(robot_interface)
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)

            # If we haven't stored the initial EEF position yet, store it now
            if initial_eef_position is None:
                initial_eef_position = current_ee_position.copy()

            # Compute the offset relative to the initial position
            offset_x = current_ee_position[0] - initial_eef_position[0]
            offset_y = current_ee_position[1] - initial_eef_position[1]
            offset_z = current_ee_position[2] - initial_eef_position[2]

            current_time = time.time() - global_start_time

            # Store offset (x, y, z) in eef_positions
            eef_positions.append((current_time, offset_x, offset_y, offset_z))

            # Also store the Z offset in y_positions, if desired
            y_positions.append(offset_y)
            timestamps.append(current_time)

            position_error = np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(target_positions))
            if time.time() - start_time > pressing_time_short:
                print("Timeout reached. Breaking loop.")
                print(pressing_time_short)
                break
            if np.max(position_error) < 3e-4:
                print("Position error is small. Breaking loop.")
                break

        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)

        if stop_movement.is_set():
            print("Movement interrupted after command.")
            break

        time.sleep(0.005)

# ---------------------------------------------------------------------------
# Discrete Trajectory: Step-by-Step Joint Position Control
# ---------------------------------------------------------------------------
def joint_position_control(robot_interface, controller_cfg):
    """
    Moves the robot through a discrete sequence of joint positions.
    
    The trajectory is split into a loading phase (step-by-step moves) and an unloading phase
    (which returns through the reverse of the loading positions).
    """
    # Define a series of discrete positions for the loading phase

    initial_positions = [0.0061, 0.2437, -0.0046, -2.5701, 0.0049, 3.7030, 0.7468]

    approach_positions = [0.0022, 1.0901, -0.0014, -1.7297, 0.0049, 3.9090, 0.7469]

    # 0mm: [0.0023, 0.9816, -0.0015, -1.8090, 0.0050, 3.8799, 0.7467]
    # 5mm: [0.0023, 0.9948, -0.0015, -1.8000, 0.0050, 3.8841, 0.7467]
    # 10mm: [0.0023, 1.0083, -0.0015, -1.7905, 0.0050, 3.8882, 0.7468]
    # 15mm: [0.0023, 1.0218, -0.0015, -1.7810, 0.0050, 3.8920, 0.7468]
    # 20mm: [0.0022, 1.0353, -0.0015, -1.7712, 0.0050, 3.8958, 0.7468]
    # 25mm: [0.0022, 1.0489, -0.0015, -1.7611, 0.0049, 3.8993, 0.7468]
    # 30mm: [0.0022, 1.0626, -0.0014, -1.7509, 0.0049, 3.9027, 0.7469]
    # 35mm: [0.0022, 1.0763, -0.0014, -1.7404, 0.0049, 3.9060, 0.7469]
    # 40mm: [0.0022, 1.0901, -0.0014, -1.7297, 0.0049, 3.9090, 0.7469]

    # 100 steps for 160mm movements. Pressing time = 0.4
    loading_positions = [ 
    [-0.0109, -0.3610, 0.0154, -2.1814, 0.0339, 3.4099, 0.7190],
    [-0.0110, -0.3560, 0.0155, -2.1768, 0.0339, 3.4103, 0.7190],
    [-0.0112, -0.3510, 0.0157, -2.1723, 0.0339, 3.4107, 0.7190],
    [-0.0113, -0.3460, 0.0158, -2.1677, 0.0339, 3.4111, 0.7190],
    [-0.0115, -0.3410, 0.0159, -2.1630, 0.0338, 3.4115, 0.7190],
    [-0.0116, -0.3361, 0.0161, -2.1584, 0.0338, 3.4118, 0.7189],
    [-0.0118, -0.3311, 0.0162, -2.1537, 0.0338, 3.4122, 0.7189],
    [-0.0120, -0.3261, 0.0164, -2.1491, 0.0338, 3.4125, 0.7189],
    [-0.0121, -0.3211, 0.0166, -2.1444, 0.0338, 3.4128, 0.7189],
    [-0.0123, -0.3162, 0.0167, -2.1397, 0.0338, 3.4130, 0.7189],
    [-0.0125, -0.3112, 0.0169, -2.1349, 0.0338, 3.4133, 0.7188],
    [-0.0127, -0.3062, 0.0171, -2.1302, 0.0338, 3.4135, 0.7188],
    [-0.0129, -0.3013, 0.0173, -2.1254, 0.0339, 3.4137, 0.7188],
    [-0.0131, -0.2963, 0.0175, -2.1206, 0.0339, 3.4139, 0.7187],
    [-0.0134, -0.2913, 0.0177, -2.1158, 0.0339, 3.4140, 0.7187],
    [-0.0136, -0.2864, 0.0180, -2.1110, 0.0339, 3.4142, 0.7187],
    [-0.0138, -0.2814, 0.0182, -2.1062, 0.0339, 3.4143, 0.7186],
    [-0.0141, -0.2765, 0.0184, -2.1013, 0.0340, 3.4144, 0.7186],
    [-0.0143, -0.2715, 0.0187, -2.0964, 0.0340, 3.4144, 0.7185],
    [-0.0146, -0.2666, 0.0190, -2.0915, 0.0341, 3.4145, 0.7185],
    [-0.0149, -0.2616, 0.0192, -2.0866, 0.0341, 3.4145, 0.7184],
    [-0.0152, -0.2567, 0.0195, -2.0817, 0.0341, 3.4145, 0.7183],
    [-0.0155, -0.2517, 0.0198, -2.0767, 0.0342, 3.4145, 0.7182],
    [-0.0158, -0.2468, 0.0201, -2.0718, 0.0343, 3.4145, 0.7182],
    [-0.0161, -0.2418, 0.0205, -2.0668, 0.0343, 3.4145, 0.7181],
    [-0.0164, -0.2369, 0.0208, -2.0618, 0.0344, 3.4144, 0.7180],
    [-0.0168, -0.2319, 0.0212, -2.0567, 0.0345, 3.4143, 0.7179],
    [-0.0172, -0.2270, 0.0216, -2.0517, 0.0346, 3.4142, 0.7178],
    [-0.0175, -0.2220, 0.0219, -2.0466, 0.0347, 3.4141, 0.7176],
    [-0.0179, -0.2171, 0.0224, -2.0415, 0.0348, 3.4139, 0.7175],
    [-0.0183, -0.2122, 0.0228, -2.0364, 0.0350, 3.4138, 0.7174],
    [-0.0188, -0.2072, 0.0232, -2.0313, 0.0351, 3.4136, 0.7172],
    [-0.0192, -0.2023, 0.0237, -2.0261, 0.0352, 3.4134, 0.7170],
    [-0.0197, -0.1973, 0.0242, -2.0209, 0.0354, 3.4131, 0.7169],
    [-0.0202, -0.1924, 0.0247, -2.0157, 0.0356, 3.4129, 0.7167],
    [-0.0207, -0.1874, 0.0252, -2.0105, 0.0358, 3.4126, 0.7165],
    [-0.0212, -0.1825, 0.0258, -2.0053, 0.0360, 3.4123, 0.7162],
    [-0.0218, -0.1775, 0.0264, -2.0000, 0.0362, 3.4120, 0.7160],
    [-0.0223, -0.1726, 0.0270, -1.9947, 0.0365, 3.4117, 0.7157],
    [-0.0230, -0.1676, 0.0277, -1.9894, 0.0368, 3.4113, 0.7154],
    [-0.0236, -0.1627, 0.0284, -1.9841, 0.0371, 3.4110, 0.7151],
    [-0.0242, -0.1577, 0.0291, -1.9788, 0.0374, 3.4106, 0.7148],
    [-0.0249, -0.1528, 0.0298, -1.9734, 0.0377, 3.4102, 0.7144],
    [-0.0257, -0.1478, 0.0306, -1.9680, 0.0381, 3.4097, 0.7140],
    [-0.0264, -0.1429, 0.0315, -1.9626, 0.0386, 3.4093, 0.7135],
    [-0.0272, -0.1379, 0.0324, -1.9571, 0.0390, 3.4088, 0.7130],
    [-0.0280, -0.1330, 0.0333, -1.9517, 0.0395, 3.4083, 0.7125],
    [-0.0289, -0.1280, 0.0343, -1.9462, 0.0401, 3.4078, 0.7119],
    [-0.0298, -0.1230, 0.0353, -1.9407, 0.0407, 3.4073, 0.7113],
    [-0.0308, -0.1180, 0.0364, -1.9352, 0.0413, 3.4067, 0.7106],
    [-0.0318, -0.1131, 0.0375, -1.9296, 0.0421, 3.4061, 0.7098],
    [-0.0328, -0.1081, 0.0388, -1.9240, 0.0429, 3.4055, 0.7090],
    [-0.0339, -0.1031, 0.0400, -1.9184, 0.0437, 3.4049, 0.7081],
    [-0.0350, -0.0981, 0.0414, -1.9128, 0.0447, 3.4043, 0.7071],
    [-0.0362, -0.0931, 0.0428, -1.9071, 0.0458, 3.4036, 0.7060],
    [-0.0374, -0.0881, 0.0443, -1.9014, 0.0469, 3.4030, 0.7047],
    [-0.0387, -0.0831, 0.0459, -1.8957, 0.0482, 3.4022, 0.7034],
    [-0.0401, -0.0781, 0.0475, -1.8900, 0.0497, 3.4015, 0.7019],
    [-0.0414, -0.0731, 0.0493, -1.8842, 0.0513, 3.4008, 0.7002],
    [-0.0429, -0.0681, 0.0511, -1.8784, 0.0530, 3.4000, 0.6984],
    [-0.0443, -0.0631, 0.0530, -1.8726, 0.0549, 3.3992, 0.6964],
    [-0.0458, -0.0581, 0.0549, -1.8667, 0.0571, 3.3984, 0.6941],
    [-0.0473, -0.0530, 0.0570, -1.8608, 0.0595, 3.3976, 0.6917],
    [-0.0488, -0.0480, 0.0591, -1.8549, 0.0621, 3.3968, 0.6889],
    [-0.0503, -0.0430, 0.0612, -1.8489, 0.0650, 3.3959, 0.6859],
    [-0.0517, -0.0379, 0.0634, -1.8429, 0.0682, 3.3950, 0.6825],
    [-0.0530, -0.0328, 0.0656, -1.8369, 0.0717, 3.3941, 0.6788],
    [-0.0543, -0.0277, 0.0677, -1.8308, 0.0756, 3.3931, 0.6747],
    [-0.0554, -0.0227, 0.0698, -1.8247, 0.0798, 3.3922, 0.6703],
    [-0.0563, -0.0176, 0.0717, -1.8185, 0.0844, 3.3912, 0.6655],
    [-0.0569, -0.0124, 0.0735, -1.8123, 0.0894, 3.3902, 0.6603],
    [-0.0572, -0.0073, 0.0750, -1.8061, 0.0947, 3.3892, 0.6548],
    [-0.0571, -0.0022, 0.0762, -1.7998, 0.1003, 3.3881, 0.6489],
    [-0.0566, 0.0030, 0.0770, -1.7934, 0.1061, 3.3871, 0.6428],
    [-0.0557, 0.0081, 0.0774, -1.7870, 0.1121, 3.3860, 0.6365],
    [-0.0543, 0.0133, 0.0774, -1.7806, 0.1183, 3.3849, 0.6300],
    [-0.0524, 0.0185, 0.0769, -1.7741, 0.1245, 3.3837, 0.6236],
    [-0.0501, 0.0236, 0.0758, -1.7676, 0.1306, 3.3825, 0.6172],
    [-0.0473, 0.0288, 0.0743, -1.7610, 0.1365, 3.3814, 0.6110],
    [-0.0442, 0.0340, 0.0724, -1.7544, 0.1422, 3.3801, 0.6051],
    [-0.0407, 0.0393, 0.0700, -1.7478, 0.1476, 3.3789, 0.5995],
    [-0.0370, 0.0445, 0.0673, -1.7411, 0.1526, 3.3776, 0.5943],
    [-0.0332, 0.0497, 0.0644, -1.7345, 0.1572, 3.3763, 0.5895],
    [-0.0292, 0.0549, 0.0612, -1.7278, 0.1615, 3.3750, 0.5852],
    [-0.0253, 0.0601, 0.0579, -1.7210, 0.1653, 3.3737, 0.5812],
    [-0.0213, 0.0654, 0.0545, -1.7143, 0.1688, 3.3723, 0.5777],
    [-0.0174, 0.0706, 0.0511, -1.7076, 0.1719, 3.3709, 0.5746],
    [-0.0136, 0.0759, 0.0477, -1.7008, 0.1746, 3.3694, 0.5719],
    [-0.0098, 0.0812, 0.0442, -1.6940, 0.1770, 3.3680, 0.5694],
    [-0.0062, 0.0864, 0.0409, -1.6871, 0.1792, 3.3665, 0.5673],
    [-0.0028, 0.0917, 0.0376, -1.6803, 0.1811, 3.3650, 0.5655],
    [0.0005, 0.0970, 0.0343, -1.6734, 0.1828, 3.3635, 0.5639],
    [0.0037, 0.1023, 0.0312, -1.6665, 0.1842, 3.3619, 0.5626],
    [0.0067, 0.1076, 0.0282, -1.6596, 0.1855, 3.3603, 0.5614],
    [0.0096, 0.1130, 0.0252, -1.6526, 0.1867, 3.3587, 0.5604],
    [0.0124, 0.1183, 0.0223, -1.6456, 0.1876, 3.3570, 0.5596],
    [0.0150, 0.1237, 0.0196, -1.6386, 0.1885, 3.3554, 0.5589],
    [0.0176, 0.1290, 0.0169, -1.6315, 0.1892, 3.3537, 0.5584],
    [0.0200, 0.1344, 0.0143, -1.6244, 0.1899, 3.3519, 0.5579],
    [0.0223, 0.1398, 0.0117, -1.6172, 0.1905, 3.3502, 0.5576],
    ]


    # For the unloading phase, simply reverse the loading positions.
    unloading_positions = list(reversed(loading_positions))

    # 1. Move to a "reset" or home position before starting (we use the first position from the loading phase)
    move_to_position_long(robot_interface, np.array(initial_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

    gripper.open()

    time.sleep(0.5)

    # 2. Approach the object with a series of discrete positions
    move_to_position_long(robot_interface, np.array(approach_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

    # gripper.close()
    gripper.goTomm(gripper_pos)
    time.sleep(0.5)

    # 3. Move to a "reset" or home position before starting (we use the first position from the loading phase)
    # middle_positions
    move_to_position_long(robot_interface, np.array(initial_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

    time.sleep(1)  # Optional delay between steps
    gripper.open()

    # Finally, signal that movement is done.
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

    # Save force data
    if force_data:
        force_df = pd.DataFrame(force_data, columns=["Timestamp", "Fx", "Fy", "Fz", "Force Magnitude"])
        force_df.to_csv(os.path.join(data_folder, "force_data.csv"), index=False)

    # Save end-effector positions (X, Y, Z) to CSV
    if eef_positions:
        eef_df = pd.DataFrame(eef_positions, columns=["Timestamp", "X_Offset", "Y_Offset", "Z_Offset"])
        eef_df.to_csv(os.path.join(data_folder, "eef_positions.csv"), index=False)

    # Save Z-position CSV
    z_pos_df = pd.DataFrame({"Timestamp": timestamps, "Y Position": y_positions, "Event": None})
    for timestamp, event in event_markers:
        closest_index = (z_pos_df["Timestamp"] - timestamp).abs().idxmin()  
        z_pos_df.loc[closest_index, "Event"] = event
    z_pos_df.to_csv(os.path.join(data_folder, "y_position_data.csv"), index=False)

    # Save joint positions
    if joint_positions:
        num_joints = len(joint_positions[0]) if isinstance(joint_positions[0], (list, np.ndarray)) else 1
        joint_pos_df = pd.DataFrame(joint_positions, columns=[f"Joint {i+1} Position" for i in range(num_joints)])
        joint_pos_df.to_csv(os.path.join(data_folder, "joint_positions.csv"), index=False)

    # Save torque data
    if torque_data:
        torque_df = pd.DataFrame(torque_data, columns=["Timestamp", "Tx", "Ty", "Tz", "Torque Magnitude"])
        torque_df.to_csv(os.path.join(data_folder, "torque_data.csv"), index=False)

    print(f"Data saved to folder: {data_folder}")
    return data_folder

def plot_merged_data(data_folder):
    # First Figure: Fx, Fy, Fz, Force Magnitude with Z-position
    fig1, ax1 = plt.subplots()

    if force_data:
        times = [entry[0] for entry in force_data]  # Extract timestamps
        Fx = [entry[1] for entry in force_data]    # Extract Fx
        Fy = [entry[2] for entry in force_data]    # Extract Fy
        Fz = [entry[3] for entry in force_data]    # Extract Fz
        force_magnitudes = [entry[4] for entry in force_data]  # Extract force magnitudes

        ax1.plot(times, Fx, label="Fx", color='tab:blue')
        ax1.plot(times, Fy, label="Fy", color='tab:orange')
        ax1.plot(times, Fz, label="Fz", color='tab:green')
        ax1.plot(times, force_magnitudes, label="Force Magnitude", color='tab:red', linestyle='--', linewidth=0.25)

        max_force_magnitude = max(abs(val) for val in force_magnitudes)
        ax1.set_ylim([-(max_force_magnitude + 5), max_force_magnitude + 5])

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force (N)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    if timestamps:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, y_positions, label="Y Position", color='tab:purple', marker='o', markersize=2)
        max_y_position = max(abs(val) for val in y_positions)
        ax2.set_ylim([-max_y_position - 0.025, max_y_position + 0.025])
        ax2.set_ylabel("Y Position (m)", color='tab:purple')
        ax2.legend(loc="upper right")

    plt.title("Forces (Fx, Fy, Fz, Magnitude) and Y-Position Over Time")
    force_plot_path = os.path.join(data_folder, "force_plot.png")
    plt.savefig(force_plot_path, dpi=1000)
    plt.show()
    plt.close(fig1)

    print(f"Force plot saved to {force_plot_path}")

    # Second Figure: Tx, Ty, Tz, Torque Magnitude with Z-position
    fig2, ax3 = plt.subplots()

    if torque_data:
        times = [entry[0] for entry in torque_data]
        Tx = [entry[1] for entry in torque_data]
        Ty = [entry[2] for entry in torque_data]
        Tz = [entry[3] for entry in torque_data]
        torque_magnitudes = [entry[4] for entry in torque_data]

        ax3.plot(times, Tx, label="Tx", color='tab:blue')
        ax3.plot(times, Ty, label="Ty", color='tab:orange')
        ax3.plot(times, Tz, label="Tz", color='tab:green')
        ax3.plot(times, torque_magnitudes, label="Torque Magnitude", color='tab:red', linestyle='--', linewidth=1)
        max_torque_magnitude = max(abs(val) for val in torque_magnitudes)
        ax3.set_ylim([-(max_torque_magnitude + 5), max_torque_magnitude + 5])

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.legend(loc="upper left")
    ax3.grid(True)

    if timestamps:
        ax4 = ax3.twinx()
        ax4.plot(timestamps, y_positions, label="Y Position", color='tab:purple', marker='o', markersize=2)
        max_z_position = max(abs(val) for val in y_positions)
        ax4.set_ylim([-max_z_position - 0.0025, max_z_position + 0.0025])
        ax4.set_ylabel("Y Position (m)", color='tab:purple')
        ax4.legend(loc="upper right")

    plt.title("Torques (Tx, Ty, Tz, Magnitude) and Y-Position Over Time")
    torque_plot_path = os.path.join(data_folder, "torque_plot.png")
    plt.savefig(torque_plot_path, dpi=1000)
    plt.show()
    plt.close(fig2)

    print(f"Torque plot saved to {torque_plot_path}")

    # New Plot for End-Effector X, Y, Z offsets
    if eef_positions:
        fig3, ax5 = plt.subplots(figsize=(10, 6))
        times_eef = [pos[0] for pos in eef_positions]
        x_off = [pos[1] for pos in eef_positions]
        y_off = [pos[2] for pos in eef_positions]
        z_off = [pos[3] for pos in eef_positions]

        ax5.plot(times_eef, x_off, label='X Offset', color='blue')
        ax5.plot(times_eef, y_off, label='Y Offset', color='orange')
        ax5.plot(times_eef, z_off, label='Z Offset', color='green')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Position Offset (m)')
        ax5.legend(loc='upper left')
        ax5.grid(True)
        ax5.set_title(eef_title)

        eef_plot_path = os.path.join(data_folder, "eef_offset_plot.png")
        plt.savefig(eef_plot_path, dpi=300)
        plt.show()
        plt.close(fig3)

        print(f"Offset end-effector position plot saved to {eef_plot_path}")

    print("All plots generated.")

def main():
    global global_start_time, force_sensor, gripper

    args = parse_args()

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

    gripper = RobotiqGripper()

    gripper.activate()
    gripper.calibrate(0, 40)
    print("Calibrated")

    # Start video recording thread if camera is enabled
    if args.enable_camera:
        video_output_path = os.path.join(data_folder, "realsense_recording.mp4")
        video_thread = threading.Thread(target=record_video, args=(video_output_path, video_duration, 30, rs_camera_index), daemon=True)
        video_thread.start()
    else:
        print("Camera recording disabled.")

    # If FT sensor is enabled, initialize and calibrate, then start monitoring thread.
    if args.enable_ft_sensor:
        sensor = initialize_force_sensor_for_calibration()
        if sensor is None:
            print("Sensor initialization failed. Exiting...")
            return

        force_offset, torque_offset = calibrate_force_sensor(sensor)
        if force_offset is None or torque_offset is None:
            print("Calibration failed. Exiting...")
            return

        monitoring_thread = threading.Thread(
            target=monitor_ft_sensor,
            args=(robot_interface, joint_controller_cfg, args.controller_type, osc_controller_cfg, sensor, force_offset, torque_offset),
            daemon=True
        )
        monitoring_thread.start()
    else:
        print("FT sensor monitoring disabled.")

    # Start movement thread using the discrete trajectory
    movement_thread = threading.Thread(target=joint_position_control, args=(robot_interface, joint_controller_cfg), daemon=True)
    movement_thread.start()

    # Wait for threads to finish
    if args.enable_ft_sensor:
        monitoring_thread.join()
    movement_thread.join()
    if args.enable_camera:
        video_thread.join()

    # Save and plot data after threads finish
    data_folder = save_data_to_csv()
    plot_merged_data(data_folder)

    print("Process complete.")

if __name__ == "__main__":
    main()
