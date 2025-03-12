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
pressing_time = 0.43
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
            if np.max(position_error) < 1e-4:
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

    # 150 steps for 160mm movements. Pressing time = 0.34
    loading_positions = [

    [-0.0383, -0.3224, 0.0336, -1.8473, 0.0301, 3.1144, 0.7270],
    [-0.0387, -0.3175, 0.0340, -1.8427, 0.0298, 3.1147, 0.7272],
    [-0.0391, -0.3126, 0.0344, -1.8381, 0.0295, 3.1149, 0.7275],
    [-0.0395, -0.3078, 0.0347, -1.8334, 0.0292, 3.1152, 0.7277],
    [-0.0399, -0.3029, 0.0351, -1.8287, 0.0289, 3.1154, 0.7280],
    [-0.0403, -0.2980, 0.0355, -1.8240, 0.0286, 3.1156, 0.7283],
    [-0.0408, -0.2931, 0.0360, -1.8193, 0.0283, 3.1157, 0.7286],
    [-0.0412, -0.2882, 0.0364, -1.8146, 0.0279, 3.1159, 0.7289],
    [-0.0417, -0.2833, 0.0368, -1.8098, 0.0276, 3.1160, 0.7292],
    [-0.0421, -0.2784, 0.0373, -1.8050, 0.0272, 3.1161, 0.7295],
    [-0.0426, -0.2734, 0.0377, -1.8001, 0.0268, 3.1162, 0.7298],
    [-0.0431, -0.2685, 0.0382, -1.7953, 0.0264, 3.1163, 0.7301],
    [-0.0436, -0.2636, 0.0386, -1.7904, 0.0260, 3.1163, 0.7305],
    [-0.0441, -0.2586, 0.0391, -1.7855, 0.0256, 3.1164, 0.7309],
    [-0.0446, -0.2537, 0.0396, -1.7805, 0.0252, 3.1164, 0.7312],
    [-0.0451, -0.2487, 0.0401, -1.7755, 0.0247, 3.1163, 0.7316],
    [-0.0456, -0.2438, 0.0406, -1.7706, 0.0242, 3.1163, 0.7320],
    [-0.0461, -0.2388, 0.0411, -1.7655, 0.0237, 3.1162, 0.7324],
    [-0.0467, -0.2338, 0.0416, -1.7605, 0.0232, 3.1162, 0.7329],
    [-0.0472, -0.2289, 0.0422, -1.7554, 0.0227, 3.1161, 0.7333],
    [-0.0478, -0.2239, 0.0427, -1.7503, 0.0221, 3.1159, 0.7338],
    [-0.0484, -0.2189, 0.0432, -1.7451, 0.0216, 3.1158, 0.7343],
    [-0.0489, -0.2139, 0.0438, -1.7400, 0.0210, 3.1156, 0.7348],
    [-0.0495, -0.2089, 0.0444, -1.7348, 0.0203, 3.1154, 0.7353],
    [-0.0501, -0.2038, 0.0449, -1.7295, 0.0197, 3.1152, 0.7359],
    [-0.0507, -0.1988, 0.0455, -1.7243, 0.0190, 3.1150, 0.7364],
    [-0.0513, -0.1938, 0.0461, -1.7190, 0.0183, 3.1147, 0.7370],
    [-0.0519, -0.1887, 0.0467, -1.7137, 0.0176, 3.1145, 0.7376],
    [-0.0525, -0.1837, 0.0473, -1.7083, 0.0169, 3.1142, 0.7382],
    [-0.0532, -0.1786, 0.0479, -1.7029, 0.0161, 3.1139, 0.7389],
    [-0.0538, -0.1736, 0.0485, -1.6975, 0.0153, 3.1135, 0.7395],
    [-0.0544, -0.1685, 0.0491, -1.6921, 0.0145, 3.1131, 0.7402],
    [-0.0550, -0.1634, 0.0497, -1.6866, 0.0136, 3.1128, 0.7409],
    [-0.0557, -0.1583, 0.0503, -1.6811, 0.0127, 3.1123, 0.7417],
    [-0.0563, -0.1532, 0.0510, -1.6755, 0.0118, 3.1119, 0.7425],
    [-0.0569, -0.1481, 0.0516, -1.6700, 0.0108, 3.1114, 0.7432],
    [-0.0575, -0.1429, 0.0522, -1.6644, 0.0098, 3.1110, 0.7441],
    [-0.0581, -0.1378, 0.0528, -1.6587, 0.0088, 3.1105, 0.7449],
    [-0.0588, -0.1326, 0.0534, -1.6530, 0.0077, 3.1099, 0.7458],
    [-0.0594, -0.1275, 0.0540, -1.6473, 0.0066, 3.1094, 0.7467],
    [-0.0600, -0.1223, 0.0546, -1.6416, 0.0055, 3.1088, 0.7476],
    [-0.0605, -0.1171, 0.0552, -1.6358, 0.0044, 3.1082, 0.7485],
    [-0.0611, -0.1119, 0.0558, -1.6299, 0.0032, 3.1076, 0.7495],
    [-0.0617, -0.1067, 0.0564, -1.6241, 0.0020, 3.1069, 0.7505],
    [-0.0622, -0.1014, 0.0570, -1.6182, 0.0007, 3.1062, 0.7515],
    [-0.0627, -0.0962, 0.0575, -1.6122, -0.0006, 3.1055, 0.7525],
    [-0.0632, -0.0910, 0.0580, -1.6063, -0.0019, 3.1048, 0.7536],
    [-0.0637, -0.0857, 0.0586, -1.6002, -0.0032, 3.1040, 0.7547],
    [-0.0642, -0.0804, 0.0591, -1.5942, -0.0046, 3.1033, 0.7558],
    [-0.0646, -0.0751, 0.0595, -1.5881, -0.0060, 3.1024, 0.7569],
    [-0.0650, -0.0698, 0.0600, -1.5819, -0.0074, 3.1016, 0.7580],
    [-0.0654, -0.0645, 0.0604, -1.5758, -0.0089, 3.1007, 0.7592],
    [-0.0658, -0.0591, 0.0608, -1.5695, -0.0103, 3.0999, 0.7603],
    [-0.0661, -0.0538, 0.0612, -1.5633, -0.0118, 3.0989, 0.7615],
    [-0.0664, -0.0484, 0.0616, -1.5570, -0.0133, 3.0980, 0.7627],
    [-0.0666, -0.0430, 0.0619, -1.5506, -0.0148, 3.0970, 0.7639],
    [-0.0669, -0.0376, 0.0622, -1.5442, -0.0164, 3.0960, 0.7651],
    [-0.0670, -0.0322, 0.0624, -1.5378, -0.0179, 3.0950, 0.7663],
    [-0.0672, -0.0267, 0.0627, -1.5313, -0.0194, 3.0939, 0.7675],
    [-0.0673, -0.0213, 0.0629, -1.5247, -0.0210, 3.0928, 0.7687],
    [-0.0674, -0.0158, 0.0631, -1.5181, -0.0225, 3.0917, 0.7699],
    [-0.0674, -0.0103, 0.0632, -1.5115, -0.0240, 3.0906, 0.7710],
    [-0.0674, -0.0047, 0.0633, -1.5048, -0.0255, 3.0894, 0.7722],
    [-0.0674, 0.0008, 0.0634, -1.4981, -0.0271, 3.0882, 0.7734],
    [-0.0673, 0.0064, 0.0635, -1.4913, -0.0286, 3.0869, 0.7745],
    [-0.0672, 0.0119, 0.0635, -1.4845, -0.0300, 3.0857, 0.7756],
    [-0.0671, 0.0175, 0.0635, -1.4776, -0.0315, 3.0844, 0.7767],
    [-0.0669, 0.0232, 0.0635, -1.4706, -0.0330, 3.0830, 0.7778],
    [-0.0667, 0.0288, 0.0634, -1.4636, -0.0344, 3.0817, 0.7789],
    [-0.0665, 0.0345, 0.0633, -1.4565, -0.0358, 3.0803, 0.7799],
    [-0.0662, 0.0402, 0.0633, -1.4494, -0.0372, 3.0788, 0.7809],
    [-0.0660, 0.0459, 0.0631, -1.4422, -0.0385, 3.0774, 0.7819],
    [-0.0657, 0.0517, 0.0630, -1.4350, -0.0398, 3.0759, 0.7829],
    [-0.0653, 0.0574, 0.0628, -1.4277, -0.0411, 3.0743, 0.7838],
    [-0.0650, 0.0632, 0.0627, -1.4204, -0.0424, 3.0727, 0.7847],
    [-0.0646, 0.0691, 0.0625, -1.4129, -0.0436, 3.0711, 0.7856],
    [-0.0642, 0.0749, 0.0623, -1.4054, -0.0448, 3.0695, 0.7864],
    [-0.0638, 0.0808, 0.0621, -1.3979, -0.0460, 3.0678, 0.7872],
    [-0.0634, 0.0867, 0.0618, -1.3903, -0.0471, 3.0661, 0.7880],
    [-0.0629, 0.0926, 0.0616, -1.3826, -0.0482, 3.0643, 0.7887],
    [-0.0625, 0.0986, 0.0614, -1.3748, -0.0493, 3.0625, 0.7894],
    [-0.0620, 0.1046, 0.0611, -1.3670, -0.0503, 3.0607, 0.7901],
    [-0.0616, 0.1107, 0.0609, -1.3591, -0.0513, 3.0588, 0.7908],
    [-0.0611, 0.1167, 0.0606, -1.3511, -0.0523, 3.0569, 0.7914],
    [-0.0606, 0.1229, 0.0604, -1.3430, -0.0532, 3.0549, 0.7919],
    [-0.0601, 0.1290, 0.0601, -1.3349, -0.0541, 3.0529, 0.7925],
    [-0.0596, 0.1352, 0.0598, -1.3266, -0.0550, 3.0508, 0.7930],
    [-0.0591, 0.1414, 0.0596, -1.3183, -0.0558, 3.0487, 0.7935],
    [-0.0586, 0.1477, 0.0593, -1.3099, -0.0566, 3.0466, 0.7940],
    [-0.0580, 0.1540, 0.0591, -1.3014, -0.0574, 3.0444, 0.7944],
    [-0.0575, 0.1604, 0.0588, -1.2928, -0.0581, 3.0421, 0.7948],
    [-0.0570, 0.1668, 0.0585, -1.2841, -0.0589, 3.0398, 0.7951],
    [-0.0565, 0.1732, 0.0583, -1.2754, -0.0595, 3.0375, 0.7955],
    [-0.0559, 0.1797, 0.0580, -1.2665, -0.0602, 3.0351, 0.7958],
    [-0.0554, 0.1862, 0.0578, -1.2575, -0.0608, 3.0326, 0.7961],
    [-0.0549, 0.1928, 0.0576, -1.2484, -0.0615, 3.0301, 0.7963],
    [-0.0544, 0.1995, 0.0573, -1.2392, -0.0620, 3.0275, 0.7966],
    [-0.0538, 0.2062, 0.0571, -1.2298, -0.0626, 3.0249, 0.7968],
    [-0.0533, 0.2130, 0.0569, -1.2204, -0.0631, 3.0222, 0.7970],
    [-0.0528, 0.2198, 0.0567, -1.2108, -0.0637, 3.0195, 0.7971],
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

    # gripper = RobotiqGripper()

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
