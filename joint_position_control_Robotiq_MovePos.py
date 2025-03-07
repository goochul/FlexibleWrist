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
pressing_time = 0.26
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
                print(pressing_time)
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
    
    # 30 steps for 100mm movement. Pressing time = 0.46s
    '''
    loading_positions = [
    [-0.0896, 0.8878, -0.0601, -2.0298, -0.1237, 4.5136, 0.8201],
    [-0.0895, 0.8922, -0.0595, -2.0175, -0.1232, 4.5056, 0.8207],
    [-0.0894, 0.8967, -0.0589, -2.0050, -0.1227, 4.4977, 0.8212],
    [-0.0893, 0.9013, -0.0583, -1.9924, -0.1222, 4.4897, 0.8218],
    [-0.0892, 0.9059, -0.0578, -1.9797, -0.1218, 4.4817, 0.8223],
    [-0.0891, 0.9107, -0.0572, -1.9670, -0.1213, 4.4736, 0.8229],
    [-0.0890, 0.9155, -0.0566, -1.9541, -0.1209, 4.4655, 0.8234],
    [-0.0889, 0.9204, -0.0560, -1.9411, -0.1205, 4.4575, 0.8239],
    [-0.0888, 0.9254, -0.0555, -1.9280, -0.1201, 4.4493, 0.8245],
    [-0.0887, 0.9304, -0.0549, -1.9148, -0.1197, 4.4412, 0.8250],
    [-0.0887, 0.9356, -0.0544, -1.9015, -0.1194, 4.4330, 0.8256],
    [-0.0886, 0.9408, -0.0538, -1.8880, -0.1190, 4.4248, 0.8261],
    [-0.0886, 0.9461, -0.0533, -1.8745, -0.1187, 4.4166, 0.8267],
    [-0.0885, 0.9516, -0.0527, -1.8608, -0.1184, 4.4083, 0.8273],
    [-0.0885, 0.9571, -0.0522, -1.8470, -0.1181, 4.4000, 0.8278],
    [-0.0885, 0.9627, -0.0516, -1.8330, -0.1178, 4.3916, 0.8284],
    [-0.0884, 0.9683, -0.0511, -1.8190, -0.1176, 4.3832, 0.8289],
    [-0.0884, 0.9741, -0.0505, -1.8047, -0.1173, 4.3747, 0.8295],
    [-0.0884, 0.9800, -0.0500, -1.7904, -0.1171, 4.3663, 0.8301],
    [-0.0884, 0.9860, -0.0495, -1.7759, -0.1169, 4.3577, 0.8306],
    [-0.0884, 0.9920, -0.0489, -1.7612, -0.1167, 4.3491, 0.8312],
    [-0.0884, 0.9982, -0.0484, -1.7464, -0.1165, 4.3405, 0.8318],
    [-0.0885, 1.0045, -0.0479, -1.7314, -0.1164, 4.3318, 0.8324],
    [-0.0885, 1.0109, -0.0473, -1.7162, -0.1162, 4.3230, 0.8330],
    [-0.0885, 1.0174, -0.0468, -1.7009, -0.1161, 4.3142, 0.8336],
    [-0.0886, 1.0240, -0.0463, -1.6854, -0.1160, 4.3053, 0.8342],
    [-0.0887, 1.0308, -0.0457, -1.6697, -0.1160, 4.2963, 0.8348],
    [-0.0887, 1.0376, -0.0452, -1.6538, -0.1159, 4.2873, 0.8354],
    [-0.0888, 1.0446, -0.0447, -1.6377, -0.1159, 4.2782, 0.8361],
    [-0.0889, 1.0518, -0.0441, -1.6214, -0.1159, 4.2690, 0.8367],

    ]
    '''

    # 100 steps for 100mm movements. Pressing time = 0.28
    loading_positions = [
    [-0.0896, 0.8878, -0.0601, -2.0298, -0.1237, 4.5136, 0.8201],
    [-0.0896, 0.8891, -0.0599, -2.0262, -0.1236, 4.5113, 0.8203],
    [-0.0895, 0.8904, -0.0598, -2.0226, -0.1234, 4.5089, 0.8204],
    [-0.0895, 0.8917, -0.0596, -2.0189, -0.1233, 4.5066, 0.8206],
    [-0.0895, 0.8930, -0.0594, -2.0153, -0.1231, 4.5043, 0.8207],
    [-0.0894, 0.8943, -0.0592, -2.0116, -0.1230, 4.5019, 0.8209],
    [-0.0894, 0.8956, -0.0591, -2.0080, -0.1228, 4.4996, 0.8211],
    [-0.0894, 0.8970, -0.0589, -2.0043, -0.1227, 4.4973, 0.8212],
    [-0.0893, 0.8983, -0.0587, -2.0006, -0.1225, 4.4949, 0.8214],
    [-0.0893, 0.8996, -0.0585, -1.9970, -0.1224, 4.4926, 0.8216],
    [-0.0893, 0.9010, -0.0584, -1.9933, -0.1223, 4.4902, 0.8217],
    [-0.0892, 0.9023, -0.0582, -1.9896, -0.1221, 4.4879, 0.8219],
    [-0.0892, 0.9037, -0.0580, -1.9859, -0.1220, 4.4855, 0.8220],
    [-0.0892, 0.9051, -0.0579, -1.9821, -0.1219, 4.4832, 0.8222],
    [-0.0892, 0.9064, -0.0577, -1.9784, -0.1217, 4.4808, 0.8224],
    [-0.0891, 0.9078, -0.0575, -1.9747, -0.1216, 4.4785, 0.8225],
    [-0.0891, 0.9092, -0.0574, -1.9709, -0.1215, 4.4761, 0.8227],
    [-0.0891, 0.9106, -0.0572, -1.9672, -0.1214, 4.4738, 0.8228],
    [-0.0890, 0.9120, -0.0570, -1.9634, -0.1212, 4.4714, 0.8230],
    [-0.0890, 0.9134, -0.0569, -1.9597, -0.1211, 4.4690, 0.8232],
    [-0.0890, 0.9148, -0.0567, -1.9559, -0.1210, 4.4667, 0.8233],
    [-0.0890, 0.9163, -0.0565, -1.9521, -0.1209, 4.4643, 0.8235],
    [-0.0889, 0.9177, -0.0564, -1.9483, -0.1207, 4.4619, 0.8236],
    [-0.0889, 0.9191, -0.0562, -1.9445, -0.1206, 4.4596, 0.8238],
    [-0.0889, 0.9206, -0.0560, -1.9407, -0.1205, 4.4572, 0.8240],
    [-0.0889, 0.9220, -0.0559, -1.9368, -0.1204, 4.4548, 0.8241],
    [-0.0888, 0.9235, -0.0557, -1.9330, -0.1203, 4.4524, 0.8243],
    [-0.0888, 0.9250, -0.0555, -1.9292, -0.1202, 4.4501, 0.8244],
    [-0.0888, 0.9264, -0.0554, -1.9253, -0.1200, 4.4477, 0.8246],
    [-0.0888, 0.9279, -0.0552, -1.9214, -0.1199, 4.4453, 0.8248],
    [-0.0888, 0.9294, -0.0550, -1.9176, -0.1198, 4.4429, 0.8249],
    [-0.0887, 0.9309, -0.0549, -1.9137, -0.1197, 4.4405, 0.8251],
    [-0.0887, 0.9324, -0.0547, -1.9098, -0.1196, 4.4381, 0.8252],
    [-0.0887, 0.9339, -0.0546, -1.9059, -0.1195, 4.4357, 0.8254],
    [-0.0887, 0.9354, -0.0544, -1.9020, -0.1194, 4.4333, 0.8256],
    [-0.0887, 0.9369, -0.0542, -1.8980, -0.1193, 4.4309, 0.8257],
    [-0.0887, 0.9385, -0.0541, -1.8941, -0.1192, 4.4285, 0.8259],
    [-0.0886, 0.9400, -0.0539, -1.8902, -0.1191, 4.4261, 0.8261],
    [-0.0886, 0.9416, -0.0537, -1.8862, -0.1190, 4.4237, 0.8262],
    [-0.0886, 0.9431, -0.0536, -1.8822, -0.1189, 4.4213, 0.8264],
    [-0.0886, 0.9447, -0.0534, -1.8783, -0.1188, 4.4189, 0.8265],
    [-0.0886, 0.9462, -0.0533, -1.8743, -0.1187, 4.4165, 0.8267],
    [-0.0886, 0.9478, -0.0531, -1.8703, -0.1186, 4.4140, 0.8269],
    [-0.0886, 0.9494, -0.0529, -1.8663, -0.1185, 4.4116, 0.8270],
    [-0.0885, 0.9510, -0.0528, -1.8623, -0.1184, 4.4092, 0.8272],
    [-0.0885, 0.9526, -0.0526, -1.8582, -0.1184, 4.4067, 0.8274],
    [-0.0885, 0.9542, -0.0525, -1.8542, -0.1183, 4.4043, 0.8275],
    [-0.0885, 0.9558, -0.0523, -1.8501, -0.1182, 4.4019, 0.8277],
    [-0.0885, 0.9574, -0.0521, -1.8461, -0.1181, 4.3994, 0.8278],
    [-0.0885, 0.9591, -0.0520, -1.8420, -0.1180, 4.3970, 0.8280],
    [-0.0885, 0.9607, -0.0518, -1.8379, -0.1179, 4.3945, 0.8282],
    [-0.0885, 0.9624, -0.0517, -1.8338, -0.1179, 4.3921, 0.8283],
    [-0.0885, 0.9640, -0.0515, -1.8297, -0.1178, 4.3896, 0.8285],
    [-0.0885, 0.9657, -0.0513, -1.8256, -0.1177, 4.3872, 0.8287],
    [-0.0885, 0.9673, -0.0512, -1.8215, -0.1176, 4.3847, 0.8288],
    [-0.0885, 0.9690, -0.0510, -1.8173, -0.1175, 4.3822, 0.8290],
    [-0.0884, 0.9707, -0.0509, -1.8132, -0.1175, 4.3798, 0.8292],
    [-0.0884, 0.9724, -0.0507, -1.8090, -0.1174, 4.3773, 0.8293],
    [-0.0884, 0.9741, -0.0506, -1.8048, -0.1173, 4.3748, 0.8295],
    [-0.0884, 0.9758, -0.0504, -1.8006, -0.1173, 4.3723, 0.8297],
    [-0.0884, 0.9775, -0.0502, -1.7964, -0.1172, 4.3698, 0.8298],
    [-0.0884, 0.9793, -0.0501, -1.7922, -0.1171, 4.3674, 0.8300],
    [-0.0884, 0.9810, -0.0499, -1.7880, -0.1171, 4.3649, 0.8302],
    [-0.0884, 0.9827, -0.0498, -1.7837, -0.1170, 4.3624, 0.8303],
    [-0.0884, 0.9845, -0.0496, -1.7795, -0.1170, 4.3599, 0.8305],
    [-0.0884, 0.9863, -0.0495, -1.7752, -0.1169, 4.3573, 0.8307],
    [-0.0884, 0.9880, -0.0493, -1.7709, -0.1168, 4.3548, 0.8308],
    [-0.0884, 0.9898, -0.0491, -1.7666, -0.1168, 4.3523, 0.8310],
    [-0.0884, 0.9916, -0.0490, -1.7623, -0.1167, 4.3498, 0.8312],
    [-0.0884, 0.9934, -0.0488, -1.7580, -0.1167, 4.3473, 0.8313],
    [-0.0884, 0.9952, -0.0487, -1.7536, -0.1166, 4.3447, 0.8315],
    [-0.0885, 0.9970, -0.0485, -1.7493, -0.1166, 4.3422, 0.8317],
    [-0.0885, 0.9988, -0.0484, -1.7449, -0.1165, 4.3397, 0.8319],
    [-0.0885, 1.0007, -0.0482, -1.7406, -0.1165, 4.3371, 0.8320],
    [-0.0885, 1.0025, -0.0480, -1.7362, -0.1164, 4.3346, 0.8322],
    [-0.0885, 1.0044, -0.0479, -1.7318, -0.1164, 4.3320, 0.8324],
    [-0.0885, 1.0062, -0.0477, -1.7273, -0.1164, 4.3295, 0.8325],
    [-0.0885, 1.0081, -0.0476, -1.7229, -0.1163, 4.3269, 0.8327],
    [-0.0885, 1.0100, -0.0474, -1.7184, -0.1163, 4.3243, 0.8329],
    [-0.0885, 1.0119, -0.0473, -1.7140, -0.1162, 4.3217, 0.8331],
    [-0.0885, 1.0138, -0.0471, -1.7095, -0.1162, 4.3192, 0.8332],
    [-0.0885, 1.0157, -0.0469, -1.7050, -0.1162, 4.3166, 0.8334],
    [-0.0886, 1.0176, -0.0468, -1.7005, -0.1162, 4.3140, 0.8336],
    [-0.0886, 1.0195, -0.0466, -1.6960, -0.1161, 4.3114, 0.8338],
    [-0.0886, 1.0215, -0.0465, -1.6914, -0.1161, 4.3088, 0.8340],
    [-0.0886, 1.0234, -0.0463, -1.6869, -0.1161, 4.3062, 0.8341],
    [-0.0886, 1.0254, -0.0462, -1.6823, -0.1161, 4.3036, 0.8343],
    [-0.0886, 1.0273, -0.0460, -1.6777, -0.1160, 4.3009, 0.8345],
    [-0.0886, 1.0293, -0.0458, -1.6731, -0.1160, 4.2983, 0.8347],
    [-0.0887, 1.0313, -0.0457, -1.6685, -0.1160, 4.2957, 0.8349],
    [-0.0887, 1.0333, -0.0455, -1.6638, -0.1160, 4.2930, 0.8350],
    [-0.0887, 1.0353, -0.0454, -1.6592, -0.1160, 4.2904, 0.8352],
    [-0.0887, 1.0374, -0.0452, -1.6545, -0.1160, 4.2877, 0.8354],
    [-0.0887, 1.0394, -0.0451, -1.6498, -0.1160, 4.2851, 0.8356],
    [-0.0888, 1.0414, -0.0449, -1.6451, -0.1159, 4.2824, 0.8358],
    [-0.0888, 1.0435, -0.0447, -1.6404, -0.1159, 4.2797, 0.8360],
    [-0.0888, 1.0456, -0.0446, -1.6356, -0.1159, 4.2770, 0.8361],
    [-0.0888, 1.0476, -0.0444, -1.6308, -0.1159, 4.2744, 0.8363],
    [-0.0889, 1.0497, -0.0443, -1.6261, -0.1159, 4.2717, 0.8365],
    [-0.0889, 1.0518, -0.0441, -1.6213, -0.1159, 4.2690, 0.8367],

    ]

    # For the unloading phase, simply reverse the loading positions.
    unloading_positions = list(reversed(loading_positions))

    # 1. Move to a "reset" or home position before starting (we use the first position from the loading phase)
    reset_joint_positions = loading_positions[0]
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    if stop_movement.is_set():
        return
    # time.sleep(0.5)

    # 2. Loading Phase (event label "1"): move through each discrete position.
    for pos in loading_positions:
        move_to_position(robot_interface, np.array(pos), controller_cfg, event_label="1")
        if stop_movement.is_set():
            return
        # time.sleep(0.5)  # Optional delay between steps

    # 3. Unloading Phase (event label "2"): return along the reversed trajectory.
    for pos in unloading_positions:
        move_to_position(robot_interface, np.array(pos), controller_cfg, event_label="2")
        if stop_movement.is_set():
            return
        # time.sleep(0.5)


    # # 1. Move to a "reset" or home position before starting (we use the first position from the loading phase)
    # reset_joint_positions = loading_positions[0]
    # move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)
    # if stop_movement.is_set():
    #     return
    # time.sleep(0.5)

    # # 2. Loading Phase (event label "1"): move through each discrete position.
    # for pos in loading_positions:
    #     move_to_position(robot_interface, np.array(pos), controller_cfg, event_label="1")
    #     if stop_movement.is_set():
    #         return
    #     time.sleep(0.5)  # Optional delay between steps

    # gripper.open()

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

    # gripper.activate()
    # gripper.calibrate(0, 40)
    # print("Calibrated")

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
