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
y_positions = []
eef_positions = []    # <-- New global list to store full (x, y, z) end-effector positions
joint_positions = []
joint_velocities = []
timestamps = []
global_start_time = None
force_sensor = None
initial_z_position = None
initial_eef_position = None
max_samples = 20000
video_duration = 600
pressing_time = 8
rs_camera_index = 6
Nexigo_camera_index = 0
force_threshold = 15
torque_threshold = 5
force_max = 20  # Set the force_max threshold here
eef_title = "Offset End-Effector Positions (X, Y, Z) Over Time with 2x Kp"

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
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
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

def smooth_move_to_position(robot_interface, current_pos, target_pos, controller_cfg, duration=3.0, steps=100):
    """
    Interpolates a smooth trajectory between the current joint positions and the target joint positions.
    The interpolation is done in 'steps' steps over 'duration' seconds.
    """
    print("Regenerating a smooth trajectory to avoid sudden movement...")
    for i in range(1, steps + 1):
        # Linear interpolation for each joint:
        interp_pos = current_pos + (target_pos - current_pos) * (i / steps)
        action = list(interp_pos) + [-1.0]
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)
        time.sleep(duration / steps)

def move_to_position(robot_interface, target_positions, controller_cfg, event_label=None):
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

    # If the difference is large, regenerate a smooth trajectory.
    # (Threshold here can be tuned – here we check if any joint difference exceeds 0.05 rad.)
    # if np.any(np.abs(np.array(target_positions) - np.array(current_joint_pos)) > 0.05):
    #     smooth_move_to_position(robot_interface, np.array(current_joint_pos), np.array(target_positions), controller_cfg, duration=3.0, steps=100)
    #     return

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
            if time.time() - start_time > pressing_time:
                print("Timeout reached. Breaking loop.")
                break
            if np.max(position_error) < 1e-4:
                print("Position error is small. Breaking loop.")
                break

        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)

        if stop_movement.is_set():
            print("Movement interrupted after command.")
            break

        time.sleep(0.01)

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
    loading_positions = [

    [0.1737, 0.3923, 0.0417, -2.1585, 0.2565, 4.1599, 0.5885],
    [0.1622, 0.3891, 0.0378, -2.1644, 0.2386, 4.1599, 0.6004],
    [0.1506, 0.3862, 0.0337, -2.1697, 0.2206, 4.1598, 0.6122],
    [0.1389, 0.3837, 0.0296, -2.1743, 0.2025, 4.1597, 0.6240],
    [0.1273, 0.3812, 0.0255, -2.1789, 0.1843, 4.1597, 0.6357],
    [0.1156, 0.3791, 0.0214, -2.1827, 0.1660, 4.1596, 0.6474],
    [0.1040, 0.3771, 0.0172, -2.1864, 0.1478, 4.1597, 0.6592],
    [0.0923, 0.3756, 0.0130, -2.1893, 0.1294, 4.1596, 0.6708],
    [0.0806, 0.3740, 0.0087, -2.1922, 0.1110, 4.1596, 0.6825],
    [0.0688, 0.3729, 0.0044, -2.1942, 0.0925, 4.1596, 0.6941],
    [0.0571, 0.3718, 0.0001, -2.1962, 0.0740, 4.1596, 0.7058],
    [0.0453, 0.3712, -0.0042, -2.1974, 0.0555, 4.1596, 0.7174],
    [0.0336, 0.3706, -0.0085, -2.1986, 0.0370, 4.1597, 0.7290],
    [0.0218, 0.3704, -0.0129, -2.1989, 0.0184, 4.1596, 0.7406],
    [0.0100, 0.3703, -0.0172, -2.1991, -0.0001, 4.1597, 0.7522],
    [-0.0018, 0.3706, -0.0215, -2.1986, -0.0186, 4.1597, 0.7639],
    [-0.0135, 0.3710, -0.0259, -2.1980, -0.0372, 4.1598, 0.7755],
    [-0.0253, 0.3718, -0.0302, -2.1965, -0.0557, 4.1598, 0.7871],
    [-0.0371, 0.3726, -0.0345, -2.1951, -0.0742, 4.1599, 0.7987],
    [-0.0488, 0.3739, -0.0388, -2.1928, -0.0926, 4.1598, 0.8104],
    [-0.0606, 0.3752, -0.0430, -2.1904, -0.1110, 4.1600, 0.8221],
    [-0.0723, 0.3769, -0.0472, -2.1873, -0.1294, 4.1600, 0.8338],
    [-0.0841, 0.3787, -0.0514, -2.1841, -0.1477, 4.1601, 0.8455],
    [-0.0958, 0.3809, -0.0556, -2.1800, -0.1659, 4.1601, 0.8572],
    [-0.1075, 0.3832, -0.0597, -2.1760, -0.1841, 4.1602, 0.8690],
    [-0.1191, 0.3859, -0.0638, -2.1711, -0.2022, 4.1603, 0.8808],
    [-0.1308, 0.3886, -0.0678, -2.1662, -0.2202, 4.1604, 0.8926],
    [-0.1424, 0.3917, -0.0718, -2.1604, -0.2381, 4.1604, 0.9045],
    [-0.1540, 0.3949, -0.0757, -2.1546, -0.2560, 4.1606, 0.9164],
    [-0.1656, 0.3985, -0.0795, -2.1480, -0.2737, 4.1606, 0.9284],
    [-0.1772, 0.4022, -0.0833, -2.1413, -0.2914, 4.1608, 0.9403],
    [-0.1887, 0.4062, -0.0870, -2.1339, -0.3089, 4.1608, 0.9524],
    [-0.2002, 0.4103, -0.0907, -2.1264, -0.3263, 4.1610, 0.9645],
    [-0.2116, 0.4148, -0.0942, -2.1180, -0.3436, 4.1610, 0.9767],
    [-0.2231, 0.4194, -0.0978, -2.1097, -0.3608, 4.1612, 0.9888],
    [-0.2345, 0.4244, -0.1012, -2.1005, -0.3778, 4.1612, 1.0011],
    [-0.2459, 0.4294, -0.1045, -2.0913, -0.3948, 4.1614, 1.0134],
    [-0.2572, 0.4348, -0.1078, -2.0812, -0.4115, 4.1614, 1.0258],
    [-0.2685, 0.4404, -0.1109, -2.0709, -0.4281, 4.1615, 1.0383],
    [-0.2798, 0.4460, -0.1140, -2.0604, -0.4447, 4.1617, 1.0508],


    # [0.1737, 0.3923, 0.0417, -2.1585, 0.2565, 4.1599, 0.5885],
    # [0.1499, 0.3861, 0.0335, -2.1698, 0.2196, 4.1597, 0.6128],
    # [0.1261, 0.3810, 0.0251, -2.1792, 0.1824, 4.1596, 0.6370],
    # [0.1021, 0.3769, 0.0165, -2.1868, 0.1448, 4.1596, 0.6610],
    # [0.0781, 0.3738, 0.0078, -2.1925, 0.1071, 4.1595, 0.6849],
    # [0.0540, 0.3717, -0.0010, -2.1965, 0.0692, 4.1595, 0.7088],
    # [0.0298, 0.3706, -0.0099, -2.1986, 0.0311, 4.1596, 0.7327],
    # [0.0057, 0.3704, -0.0188, -2.1989, -0.0069, 4.1596, 0.7565],
    # [-0.0185, 0.3713, -0.0277, -2.1973, -0.0450, 4.1597, 0.7804],
    # [-0.0427, 0.3732, -0.0365, -2.1939, -0.0829, 4.1598, 0.8043],
    # [-0.0668, 0.3761, -0.0452, -2.1887, -0.1207, 4.1599, 0.8282],
    # [-0.0908, 0.3800, -0.0538, -2.1817, -0.1583, 4.1600, 0.8523],
    # [-0.1148, 0.3849, -0.0623, -2.1729, -0.1955, 4.1602, 0.8764],
    # [-0.1387, 0.3907, -0.0705, -2.1622, -0.2325, 4.1604, 0.9007],
    # [-0.1626, 0.3975, -0.0785, -2.1497, -0.2691, 4.1606, 0.9252],
    # [-0.1863, 0.4053, -0.0862, -2.1354, -0.3052, 4.1608, 0.9499],
    # [-0.2098, 0.4141, -0.0937, -2.1194, -0.3409, 4.1610, 0.9747],
    # [-0.2333, 0.4238, -0.1008, -2.1015, -0.3760, 4.1612, 0.9998],
    # [-0.2566, 0.4345, -0.1076, -2.0817, -0.4106, 4.1614, 1.0252],
    # [-0.2798, 0.4461, -0.1140, -2.0602, -0.4447, 4.1616, 1.0508],


## 60 steps for 0.1m movement
    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0119, 0.3711, -0.0253, -2.1979, -0.0346, 4.1598, 0.7739],
    # [-0.0150, 0.3713, -0.0264, -2.1976, -0.0395, 4.1598, 0.7770],
    # [-0.0181, 0.3714, -0.0276, -2.1973, -0.0444, 4.1598, 0.7801],
    # [-0.0212, 0.3716, -0.0287, -2.1970, -0.0493, 4.1598, 0.7831],
    # [-0.0243, 0.3718, -0.0298, -2.1966, -0.0542, 4.1598, 0.7862],
    # [-0.0275, 0.3720, -0.0310, -2.1962, -0.0591, 4.1598, 0.7893],
    # [-0.0306, 0.3723, -0.0321, -2.1958, -0.0640, 4.1599, 0.7924],
    # [-0.0337, 0.3725, -0.0333, -2.1954, -0.0689, 4.1599, 0.7955],
    # [-0.0368, 0.3728, -0.0344, -2.1949, -0.0737, 4.1599, 0.7985],
    # [-0.0399, 0.3731, -0.0355, -2.1944, -0.0786, 4.1599, 0.8016],
    # [-0.0430, 0.3734, -0.0367, -2.1938, -0.0835, 4.1599, 0.8047],
    # [-0.0461, 0.3737, -0.0378, -2.1933, -0.0884, 4.1599, 0.8078],
    # [-0.0492, 0.3740, -0.0389, -2.1927, -0.0932, 4.1599, 0.8109],
    # [-0.0523, 0.3744, -0.0400, -2.1920, -0.0981, 4.1599, 0.8139],
    # [-0.0554, 0.3747, -0.0412, -2.1914, -0.1030, 4.1600, 0.8170],
    # [-0.0585, 0.3751, -0.0423, -2.1907, -0.1079, 4.1600, 0.8201],
    # [-0.0617, 0.3755, -0.0434, -2.1899, -0.1127, 4.1600, 0.8232],
    # [-0.0648, 0.3759, -0.0445, -2.1892, -0.1176, 4.1600, 0.8263],
    # [-0.0679, 0.3764, -0.0457, -2.1884, -0.1224, 4.1600, 0.8294],
    # [-0.0710, 0.3768, -0.0468, -2.1876, -0.1273, 4.1600, 0.8325],
    # [-0.0741, 0.3773, -0.0479, -2.1867, -0.1321, 4.1601, 0.8356],
    # [-0.0772, 0.3778, -0.0490, -2.1859, -0.1370, 4.1601, 0.8387],
    # [-0.0803, 0.3783, -0.0501, -2.1850, -0.1418, 4.1601, 0.8418],
    # [-0.0834, 0.3788, -0.0512, -2.1840, -0.1466, 4.1601, 0.8449],
    # [-0.0865, 0.3793, -0.0523, -2.1831, -0.1515, 4.1601, 0.8480],
    # [-0.0896, 0.3799, -0.0534, -2.1821, -0.1563, 4.1601, 0.8511],
    # [-0.0927, 0.3804, -0.0545, -2.1811, -0.1611, 4.1602, 0.8542],
    # [-0.0958, 0.3810, -0.0556, -2.1800, -0.1659, 4.1602, 0.8573],
    # [-0.0988, 0.3816, -0.0567, -2.1789, -0.1708, 4.1602, 0.8604],
    # [-0.1019, 0.3822, -0.0578, -2.1778, -0.1756, 4.1602, 0.8635],
    # [-0.1050, 0.3829, -0.0589, -2.1767, -0.1804, 4.1602, 0.8666],
    # [-0.1081, 0.3835, -0.0599, -2.1755, -0.1852, 4.1603, 0.8697],
    # [-0.1112, 0.3842, -0.0610, -2.1743, -0.1899, 4.1603, 0.8728],
    # [-0.1143, 0.3848, -0.0621, -2.1730, -0.1947, 4.1603, 0.8760],
    # [-0.1174, 0.3855, -0.0632, -2.1718, -0.1995, 4.1603, 0.8791],
    # [-0.1205, 0.3863, -0.0642, -2.1705, -0.2043, 4.1604, 0.8822],
    # [-0.1235, 0.3870, -0.0653, -2.1691, -0.2091, 4.1604, 0.8853],
    # [-0.1266, 0.3877, -0.0664, -2.1678, -0.2138, 4.1604, 0.8885],
    # [-0.1297, 0.3885, -0.0674, -2.1664, -0.2186, 4.1604, 0.8916],
    # [-0.1328, 0.3893, -0.0685, -2.1650, -0.2233, 4.1605, 0.8947],
    # [-0.1359, 0.3901, -0.0695, -2.1635, -0.2281, 4.1605, 0.8979],
    # [-0.1389, 0.3909, -0.0706, -2.1621, -0.2328, 4.1605, 0.9010],
    # [-0.1420, 0.3917, -0.0716, -2.1605, -0.2376, 4.1605, 0.9042],
    # [-0.1451, 0.3925, -0.0727, -2.1590, -0.2423, 4.1605, 0.9073],
    # [-0.1482, 0.3934, -0.0737, -2.1574, -0.2470, 4.1606, 0.9105],
    # [-0.1512, 0.3943, -0.0747, -2.1558, -0.2517, 4.1606, 0.9136],
    # [-0.1543, 0.3952, -0.0758, -2.1542, -0.2564, 4.1606, 0.9168],
    # [-0.1574, 0.3961, -0.0768, -2.1526, -0.2611, 4.1606, 0.9199],
    # [-0.1604, 0.3970, -0.0778, -2.1509, -0.2658, 4.1607, 0.9231],
    # [-0.1635, 0.3979, -0.0788, -2.1492, -0.2705, 4.1607, 0.9262],
    # [-0.1665, 0.3989, -0.0798, -2.1474, -0.2752, 4.1607, 0.9294],
    # [-0.1696, 0.3999, -0.0808, -2.1456, -0.2798, 4.1608, 0.9326],
    # [-0.1726, 0.4009, -0.0818, -2.1438, -0.2845, 4.1608, 0.9357],
    # [-0.1757, 0.4019, -0.0828, -2.1420, -0.2892, 4.1608, 0.9389],
    # [-0.1788, 0.4029, -0.0838, -2.1401, -0.2938, 4.1608, 0.9421],
    # [-0.1818, 0.4039, -0.0848, -2.1382, -0.2984, 4.1609, 0.9453],
    # [-0.1849, 0.4050, -0.0858, -2.1363, -0.3031, 4.1609, 0.9485],
    # [-0.1879, 0.4060, -0.0868, -2.1343, -0.3077, 4.1609, 0.9517],
    # [-0.1909, 0.4071, -0.0877, -2.1323, -0.3123, 4.1609, 0.9548],

## 30 steps for 0.05m movement
    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0119, 0.3711, -0.0253, -2.1979, -0.0347, 4.1598, 0.7740],
    # [-0.0151, 0.3713, -0.0265, -2.1976, -0.0397, 4.1598, 0.7771],
    # [-0.0183, 0.3714, -0.0276, -2.1973, -0.0447, 4.1598, 0.7802],
    # [-0.0214, 0.3716, -0.0288, -2.1969, -0.0496, 4.1598, 0.7834],
    # [-0.0246, 0.3718, -0.0299, -2.1966, -0.0546, 4.1598, 0.7865],
    # [-0.0278, 0.3720, -0.0311, -2.1962, -0.0596, 4.1598, 0.7896],
    # [-0.0309, 0.3723, -0.0323, -2.1958, -0.0646, 4.1599, 0.7927],
    # [-0.0341, 0.3725, -0.0334, -2.1953, -0.0695, 4.1599, 0.7959],
    # [-0.0373, 0.3728, -0.0346, -2.1948, -0.0745, 4.1599, 0.7990],
    # [-0.0404, 0.3731, -0.0357, -2.1943, -0.0795, 4.1599, 0.8021],
    # [-0.0436, 0.3734, -0.0369, -2.1937, -0.0844, 4.1599, 0.8053],
    # [-0.0468, 0.3738, -0.0380, -2.1931, -0.0894, 4.1599, 0.8084],
    # [-0.0499, 0.3741, -0.0392, -2.1925, -0.0943, 4.1599, 0.8115],
    # [-0.0531, 0.3745, -0.0403, -2.1919, -0.0993, 4.1599, 0.8147],
    # [-0.0562, 0.3748, -0.0415, -2.1912, -0.1042, 4.1600, 0.8178],
    # [-0.0594, 0.3752, -0.0426, -2.1905, -0.1092, 4.1600, 0.8210],
    # [-0.0626, 0.3756, -0.0437, -2.1897, -0.1141, 4.1600, 0.8241],
    # [-0.0657, 0.3761, -0.0449, -2.1890, -0.1191, 4.1600, 0.8273],
    # [-0.0689, 0.3765, -0.0460, -2.1881, -0.1240, 4.1600, 0.8304],
    # [-0.0720, 0.3770, -0.0472, -2.1873, -0.1290, 4.1600, 0.8335],
    # [-0.0752, 0.3775, -0.0483, -2.1864, -0.1339, 4.1601, 0.8367],
    # [-0.0783, 0.3780, -0.0494, -2.1855, -0.1388, 4.1601, 0.8398],
    # [-0.0815, 0.3785, -0.0505, -2.1846, -0.1437, 4.1601, 0.8430],
    # [-0.0847, 0.3790, -0.0517, -2.1836, -0.1486, 4.1601, 0.8461],
    # [-0.0878, 0.3796, -0.0528, -2.1826, -0.1536, 4.1601, 0.8493],
    # [-0.0910, 0.3801, -0.0539, -2.1816, -0.1585, 4.1602, 0.8525],
    # [-0.0941, 0.3807, -0.0550, -2.1806, -0.1634, 4.1602, 0.8556],
    # [-0.0972, 0.3813, -0.0561, -2.1795, -0.1683, 4.1602, 0.8588],
    # [-0.1004, 0.3819, -0.0572, -2.1784, -0.1732, 4.1602, 0.8619],


    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0088, 0.3710, -0.0241, -2.1981, -0.0297, 4.1598, 0.7708],
    # [-0.0190, 0.3714, -0.0279, -2.1974, -0.0458, 4.1598, 0.7809],
    # [-0.0292, 0.3721, -0.0316, -2.1962, -0.0618, 4.1598, 0.7910],
    # [-0.0394, 0.3729, -0.0353, -2.1946, -0.0778, 4.1598, 0.8011],
    # [-0.0496, 0.3739, -0.0391, -2.1928, -0.0938, 4.1599, 0.8112],
    # [-0.0598, 0.3752, -0.0427, -2.1906, -0.1098, 4.1599, 0.8213],
    # [-0.0699, 0.3765, -0.0464, -2.1880, -0.1257, 4.1599, 0.8314],
    # [-0.0801, 0.3781, -0.0501, -2.1852, -0.1416, 4.1600, 0.8416],
    # [-0.0903, 0.3799, -0.0537, -2.1820, -0.1574, 4.1601, 0.8518],
    # [-0.1004, 0.3818, -0.0573, -2.1785, -0.1732, 4.1601, 0.8619],
    


        # [-0.0088,  0.3710, -0.0241, -2.1981, -0.0297,  4.1598, 0.7708],
        # [-0.0088,  0.3710, -0.0241, -2.1981, -0.0297,  4.1598, 0.7708],
        # [ 0.0014,  0.3706, -0.0204, -2.1988, -0.0137,  4.1598, 0.7608],
        # [ 0.0116,  0.3704, -0.0166, -2.1991,  0.0024,  4.1597, 0.7507],
        # [ 0.0218,  0.3704, -0.0129, -2.1990,  0.0185,  4.1596, 0.7406],
        # [ 0.0320,  0.3706, -0.0091, -2.1986,  0.0346,  4.1596, 0.7306],
        # [ 0.0422,  0.3710, -0.0054, -2.1978,  0.0506,  4.1596, 0.7205],
        # [ 0.0524,  0.3716, -0.0016, -2.1968,  0.0667,  4.1596, 0.7104],
        # [ 0.0626,  0.3723,  0.0021, -2.1954,  0.0827,  4.1596, 0.7004],
        # [ 0.0728,  0.3732,  0.0058, -2.1937,  0.0987,  4.1596, 0.6903],
        # [ 0.0829,  0.3743,  0.0095, -2.1916,  0.1147,  4.1596, 0.6802],
    ]

    # For the unloading phase, simply reverse the loading positions.
    unloading_positions = list(reversed(loading_positions))

    # 1. Move to a "reset" or home position before starting (we use the first position from the loading phase)
    reset_joint_positions = loading_positions[0]
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return
    time.sleep(0.5)

    # 2. Loading Phase (event label "1"): move through each discrete position.
    for pos in loading_positions:
        move_to_position(robot_interface, np.array(pos), controller_cfg, event_label="1")
        if stop_movement.is_set():
            return
        time.sleep(0.5)  # Optional delay between steps

    # 3. Unloading Phase (event label "2"): return along the reversed trajectory.
    for pos in unloading_positions:
        move_to_position(robot_interface, np.array(pos), controller_cfg, event_label="2")
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
        max_z_position = max(abs(val) for val in y_positions)
        ax2.set_ylim([-max_z_position - 0.0025, max_z_position + 0.0025])
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
    global global_start_time, force_sensor

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
