import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger
logger = get_deoxys_example_logger()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    return parser.parse_args()
def osc_move(robot_interface, controller_type, controller_cfg, num_steps, time_interval=0.01):
    z_positions = []  # List to store z-axis positions
    time_stamps = []  # List to store time stamps
    start_time = time.time()  # Record start time
    for step in range(num_steps):
        current_pose = robot_interface.last_eef_pose
        z_position = current_pose[2, 3]  # Extract z-axis (vertical position)
        current_time = time.time() - start_time  # Calculate elapsed time
        # Store z-axis position and the corresponding time stamp
        z_positions.append(z_position)
        time_stamps.append(current_time)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0])
        print(f"Step {step}, Current z-axis position: {z_position}")
        robot_interface.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
        # Sleep for the time interval (optional)
        time.sleep(time_interval)
    return time_stamps, z_positions
def start_gravity_compensation(robot_interface, controller_type, controller_cfg):
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)
    print("Starting gravity compensation at the current position...")
    time_stamps, z_positions = osc_move(robot_interface, controller_type, controller_cfg, num_steps=500)
    # Plot the z-axis with time
    plt.figure()
    plt.plot(time_stamps, z_positions, label="Z-axis position")
    plt.xlabel("Time (s)")
    plt.ylabel("Z-axis position (m)")
    plt.title("Z-axis position over time")
    plt.legend()
    plt.show()
def main():
    args = parse_args()
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type)
    while robot_interface.state_buffer_size == 0:
        logger.warning("Robot state not received")
        time.sleep(0.5)
    # Start gravity compensation without resetting joint positions
    start_gravity_compensation(robot_interface, controller_type, controller_cfg)
    robot_interface.close()
if __name__ == "__main__":
    main()