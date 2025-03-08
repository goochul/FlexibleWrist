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
max_samples = 1000
video_duration = 20
pressing_time_long = 5
pressing_time_short = 0.34
rs_camera_index = 6
Nexigo_camera_index = 0
force_threshold = 15
torque_threshold = 5
force_max = 50  # Set the force_max threshold here
eef_title = "End-Effector Positions (X, Y, Z), 450mm to Y-dir"

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
    parser.set_defaults(enable_ft_sensor=False)
    # Camera toggle: default is disabled here. Use --enable-camera to enable.
    parser.add_argument("--enable-camera", dest="enable_camera", action="store_true", help="Enable camera recording")
    parser.add_argument("--disable-camera", dest="enable_camera", action="store_false", help="Disable camera recording")
    parser.set_defaults(enable_camera=False)
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

            # Flip the frame both vertically and horizontally
            flipped_frame = cv2.flip(frame, -1)

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

    initial_positions = [-0.0879, 0.5412, -0.0809, -2.3352, -0.1555, 4.4739, 0.8284]

    approach_positions = [-0.0875, 0.8052, -0.0637, -2.0776, -0.1288, 4.4792, 0.8250]

    # For the unloading phase, simply reverse the loading positions.
    lift_positions = [-0.0843, 0.6186, -0.0712, -2.1620, -0.1440, 4.3780, 0.8382]

    target_positions = [-0.0836, 0.8093, -0.0537, -1.7614, -0.1337, 4.1681, 0.8572]


    # 100 steps for 100mm movements. Pressing time = 0.28
    loading_positions = [
    [-0.0843, 0.6186, -0.0712, -2.1620, -0.1440, 4.3780, 0.8382],
    [-0.0843, 0.6203, -0.0710, -2.1584, -0.1438, 4.3760, 0.8384],
    [-0.0842, 0.6219, -0.0708, -2.1547, -0.1436, 4.3740, 0.8386],
    [-0.0842, 0.6236, -0.0706, -2.1510, -0.1434, 4.3721, 0.8388],
    [-0.0841, 0.6253, -0.0705, -2.1474, -0.1433, 4.3701, 0.8390],
    [-0.0841, 0.6270, -0.0703, -2.1437, -0.1431, 4.3681, 0.8391],
    [-0.0840, 0.6287, -0.0701, -2.1400, -0.1429, 4.3661, 0.8393],
    [-0.0840, 0.6304, -0.0699, -2.1364, -0.1427, 4.3641, 0.8395],
    [-0.0839, 0.6321, -0.0697, -2.1327, -0.1426, 4.3621, 0.8397],
    [-0.0839, 0.6338, -0.0695, -2.1290, -0.1424, 4.3601, 0.8399],
    [-0.0839, 0.6355, -0.0693, -2.1253, -0.1422, 4.3581, 0.8401],
    [-0.0838, 0.6372, -0.0692, -2.1215, -0.1420, 4.3561, 0.8403],
    [-0.0838, 0.6389, -0.0690, -2.1178, -0.1419, 4.3541, 0.8405],
    [-0.0837, 0.6407, -0.0688, -2.1141, -0.1417, 4.3521, 0.8406],
    [-0.0837, 0.6424, -0.0686, -2.1104, -0.1415, 4.3501, 0.8408],
    [-0.0837, 0.6441, -0.0684, -2.1066, -0.1414, 4.3481, 0.8410],
    [-0.0836, 0.6459, -0.0682, -2.1029, -0.1412, 4.3461, 0.8412],
    [-0.0836, 0.6476, -0.0681, -2.0991, -0.1411, 4.3441, 0.8414],
    [-0.0836, 0.6493, -0.0679, -2.0954, -0.1409, 4.3421, 0.8416],
    [-0.0835, 0.6511, -0.0677, -2.0916, -0.1407, 4.3401, 0.8418],
    [-0.0835, 0.6528, -0.0675, -2.0878, -0.1406, 4.3381, 0.8420],
    [-0.0835, 0.6546, -0.0673, -2.0841, -0.1404, 4.3360, 0.8421],
    [-0.0834, 0.6564, -0.0672, -2.0803, -0.1403, 4.3340, 0.8423],
    [-0.0834, 0.6581, -0.0670, -2.0765, -0.1401, 4.3320, 0.8425],
    [-0.0834, 0.6599, -0.0668, -2.0727, -0.1400, 4.3300, 0.8427],
    [-0.0833, 0.6617, -0.0666, -2.0689, -0.1398, 4.3279, 0.8429],
    [-0.0833, 0.6635, -0.0664, -2.0650, -0.1397, 4.3259, 0.8431],
    [-0.0833, 0.6653, -0.0663, -2.0612, -0.1396, 4.3239, 0.8433],
    [-0.0833, 0.6671, -0.0661, -2.0574, -0.1394, 4.3218, 0.8434],
    [-0.0832, 0.6689, -0.0659, -2.0535, -0.1393, 4.3198, 0.8436],
    [-0.0832, 0.6707, -0.0657, -2.0497, -0.1391, 4.3177, 0.8438],
    [-0.0832, 0.6725, -0.0656, -2.0458, -0.1390, 4.3157, 0.8440],
    [-0.0832, 0.6743, -0.0654, -2.0420, -0.1389, 4.3136, 0.8442],
    [-0.0831, 0.6761, -0.0652, -2.0381, -0.1387, 4.3116, 0.8444],
    [-0.0831, 0.6780, -0.0650, -2.0342, -0.1386, 4.3095, 0.8446],
    [-0.0831, 0.6798, -0.0648, -2.0303, -0.1385, 4.3075, 0.8447],
    [-0.0831, 0.6816, -0.0647, -2.0264, -0.1383, 4.3054, 0.8449],
    [-0.0831, 0.6835, -0.0645, -2.0225, -0.1382, 4.3034, 0.8451],
    [-0.0830, 0.6853, -0.0643, -2.0186, -0.1381, 4.3013, 0.8453],
    [-0.0830, 0.6872, -0.0641, -2.0147, -0.1380, 4.2992, 0.8455],
    [-0.0830, 0.6890, -0.0640, -2.0108, -0.1378, 4.2971, 0.8457],
    [-0.0830, 0.6909, -0.0638, -2.0068, -0.1377, 4.2951, 0.8459],
    [-0.0830, 0.6928, -0.0636, -2.0029, -0.1376, 4.2930, 0.8461],
    [-0.0830, 0.6946, -0.0634, -1.9989, -0.1375, 4.2909, 0.8462],
    [-0.0829, 0.6965, -0.0633, -1.9950, -0.1374, 4.2888, 0.8464],
    [-0.0829, 0.6984, -0.0631, -1.9910, -0.1372, 4.2867, 0.8466],
    [-0.0829, 0.7003, -0.0629, -1.9870, -0.1371, 4.2846, 0.8468],
    [-0.0829, 0.7022, -0.0627, -1.9830, -0.1370, 4.2825, 0.8470],
    [-0.0829, 0.7041, -0.0626, -1.9790, -0.1369, 4.2804, 0.8472],
    [-0.0829, 0.7060, -0.0624, -1.9750, -0.1368, 4.2783, 0.8474],
    [-0.0829, 0.7079, -0.0622, -1.9710, -0.1367, 4.2762, 0.8476],
    [-0.0829, 0.7098, -0.0620, -1.9670, -0.1366, 4.2741, 0.8478],
    [-0.0829, 0.7117, -0.0619, -1.9629, -0.1365, 4.2720, 0.8479],
    [-0.0829, 0.7137, -0.0617, -1.9589, -0.1364, 4.2699, 0.8481],
    [-0.0828, 0.7156, -0.0615, -1.9548, -0.1363, 4.2678, 0.8483],
    [-0.0828, 0.7175, -0.0613, -1.9508, -0.1362, 4.2657, 0.8485],
    [-0.0828, 0.7195, -0.0612, -1.9467, -0.1361, 4.2635, 0.8487],
    [-0.0828, 0.7214, -0.0610, -1.9426, -0.1360, 4.2614, 0.8489],
    [-0.0828, 0.7234, -0.0608, -1.9385, -0.1359, 4.2593, 0.8491],
    [-0.0828, 0.7253, -0.0607, -1.9344, -0.1358, 4.2571, 0.8493],
    [-0.0828, 0.7273, -0.0605, -1.9303, -0.1357, 4.2550, 0.8495],
    [-0.0828, 0.7293, -0.0603, -1.9262, -0.1357, 4.2529, 0.8497],
    [-0.0828, 0.7313, -0.0601, -1.9221, -0.1356, 4.2507, 0.8498],
    [-0.0828, 0.7333, -0.0600, -1.9179, -0.1355, 4.2486, 0.8500],
    [-0.0828, 0.7353, -0.0598, -1.9138, -0.1354, 4.2464, 0.8502],
    [-0.0828, 0.7373, -0.0596, -1.9096, -0.1353, 4.2442, 0.8504],
    [-0.0829, 0.7393, -0.0594, -1.9055, -0.1352, 4.2421, 0.8506],
    [-0.0829, 0.7413, -0.0593, -1.9013, -0.1352, 4.2399, 0.8508],
    [-0.0829, 0.7433, -0.0591, -1.8971, -0.1351, 4.2377, 0.8510],
    [-0.0829, 0.7453, -0.0589, -1.8929, -0.1350, 4.2356, 0.8512],
    [-0.0829, 0.7473, -0.0587, -1.8887, -0.1349, 4.2334, 0.8514],
    [-0.0829, 0.7494, -0.0586, -1.8844, -0.1349, 4.2312, 0.8516],
    [-0.0829, 0.7514, -0.0584, -1.8802, -0.1348, 4.2290, 0.8518],
    [-0.0829, 0.7535, -0.0582, -1.8760, -0.1347, 4.2268, 0.8520],
    [-0.0829, 0.7555, -0.0581, -1.8717, -0.1347, 4.2246, 0.8522],
    [-0.0829, 0.7576, -0.0579, -1.8674, -0.1346, 4.2224, 0.8524],
    [-0.0830, 0.7597, -0.0577, -1.8632, -0.1346, 4.2202, 0.8526],
    [-0.0830, 0.7617, -0.0575, -1.8589, -0.1345, 4.2180, 0.8528],
    [-0.0830, 0.7638, -0.0574, -1.8546, -0.1344, 4.2158, 0.8530],
    [-0.0830, 0.7659, -0.0572, -1.8503, -0.1344, 4.2135, 0.8532],
    [-0.0830, 0.7680, -0.0570, -1.8459, -0.1343, 4.2113, 0.8533],
    [-0.0830, 0.7701, -0.0568, -1.8416, -0.1343, 4.2091, 0.8535],
    [-0.0831, 0.7722, -0.0567, -1.8373, -0.1342, 4.2069, 0.8537],
    [-0.0831, 0.7743, -0.0565, -1.8329, -0.1342, 4.2046, 0.8539],
    [-0.0831, 0.7765, -0.0563, -1.8285, -0.1341, 4.2024, 0.8541],
    [-0.0831, 0.7786, -0.0561, -1.8241, -0.1341, 4.2001, 0.8543],
    [-0.0832, 0.7807, -0.0560, -1.8197, -0.1341, 4.1979, 0.8545],
    [-0.0832, 0.7829, -0.0558, -1.8153, -0.1340, 4.1956, 0.8547],
    [-0.0832, 0.7851, -0.0556, -1.8109, -0.1340, 4.1933, 0.8550],
    [-0.0832, 0.7872, -0.0554, -1.8065, -0.1340, 4.1911, 0.8552],
    [-0.0833, 0.7894, -0.0553, -1.8020, -0.1339, 4.1888, 0.8554],
    [-0.0833, 0.7916, -0.0551, -1.7976, -0.1339, 4.1865, 0.8556],
    [-0.0833, 0.7937, -0.0549, -1.7931, -0.1339, 4.1842, 0.8558],
    [-0.0833, 0.7959, -0.0547, -1.7886, -0.1338, 4.1819, 0.8560],
    [-0.0834, 0.7981, -0.0545, -1.7841, -0.1338, 4.1796, 0.8562],
    [-0.0834, 0.8003, -0.0544, -1.7796, -0.1338, 4.1773, 0.8564],
    [-0.0834, 0.8026, -0.0542, -1.7751, -0.1338, 4.1750, 0.8566],
    [-0.0835, 0.8048, -0.0540, -1.7705, -0.1338, 4.1727, 0.8568],
    [-0.0835, 0.8070, -0.0538, -1.7660, -0.1337, 4.1704, 0.8570],
    [-0.0836, 0.8093, -0.0537, -1.7614, -0.1337, 4.1681, 0.8572],

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
    gripper.goTomm(10)
    time.sleep(0.5)

    # 3. Lift the object
    move_to_position_long(robot_interface, np.array(lift_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

    # # 4. Move to a target position
    # move_to_position_long(robot_interface, np.array(target_positions), controller_cfg)
    # if stop_movement.is_set():
    #     return
    # time.sleep(0.5)

    # # 5. come back to the lift position
    # move_to_position_long(robot_interface, np.array(lift_positions), controller_cfg)
    # if stop_movement.is_set():
    #     return
    # time.sleep(0.5)

    # 4. Loading Phase: move through each discrete position.
    for pos in loading_positions:
        move_to_position_short(robot_interface, np.array(pos), controller_cfg, event_label="1")
        if stop_movement.is_set():
            return
        # time.sleep(0.5)  # Optional delay between steps

    # 3. Unloading Phase (event label "2"): return along the reversed trajectory.
    for pos in unloading_positions:
        move_to_position_short(robot_interface, np.array(pos), controller_cfg, event_label="2")
        if stop_movement.is_set():
            return
        # time.sleep(0.5)


    gripper.open()

    time.sleep(0.5)

    # 6. Return to the initial position
    move_to_position_long(robot_interface, np.array(initial_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

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
