import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

# Global variables to store z position and time for plotting
z_positions = []
timestamps = []
global_start_time = None  # Global start time for consistent timestamps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    args = parser.parse_args()
    return args

def get_end_effector_position(robot_interface):
    """Get the current end-effector position (x, y, z)."""
    ee_pose = robot_interface.last_eef_pose
    position = ee_pose[:3, 3]
    return position

def move_to_position(robot_interface, target_positions, controller_cfg):
    """Move the robot to the target joint positions."""
    action = list(target_positions) + [-1.0]
    start_time = time.time()  # Start the timer for timeout
    while True:
        if len(robot_interface._state_buffer) > 0:
            # Log the current and desired joint positions
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            # Log the current end-effector position
            current_ee_position = get_end_effector_position(robot_interface)
            logger.info(f"Current End-Effector Position (x, y, z): {np.round(current_ee_position, 3)}")

            # Store the z position and corresponding time
            z_positions.append(current_ee_position[2])
            timestamps.append(time.time() - global_start_time)  # Use global start time

            # Check if the robot has reached the target position
            if (
                np.max(
                    np.abs(
                        np.array(robot_interface._state_buffer[-1].q)
                        - np.array(target_positions)
                    )
                )
                < 1e-3
            ) or (time.time() - start_time > 3):  # Break if the position is reached
                break

        # Send the control command
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=controller_cfg,
        )
        time.sleep(0.05)  # Add a small delay to ensure smooth control

def plot_z_position():
    """Plot the z position over time."""
    plt.figure()
    plt.plot(timestamps, z_positions, marker='o')
    plt.xlabel('Time (s)')
    plt.ylabel('End-Effector Z Position (m)')
    plt.title('Z Position of End-Effector Over Time')
    plt.grid(True)
    plt.show()

def main():
    global global_start_time  # Use the global variable for consistent timestamps
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
    print(f"Time Fraction: {controller_cfg.traj_interpolator_cfg.time_fraction}")

    controller_type = "JOINT_POSITION"

    #'''
    # My resetting joints
    reset_joint_positions = [0.00704433,  0.79005939, -0.03927315, -2.06660569, -0.01351208,  4.40001906,  0.78233447]

    # [0.00963172,  0.43557522, -0.04227658, -2.18943231, -0.01642555,  4.16616049,  0.77614201]
    # [0.00783327163909093,    0.5558480613472636, -0.04036106289390794, -2.163110868612272, -0.015487256765292425, 4.260755619281773, 0.7779242274028955]
    # [0.00699113,  0.68371062, -0.03936263, -2.11721285, -0.01475596,  4.34364621,  0.78012584]
    # [0.00704433,  0.79005939, -0.03927315, -2.06660569, -0.01351208,  4.40001906,  0.78233447]  # box + aluminum frame height origin
    # [0.00596583,  0.81711726, -0.03810693, -2.05191146, -0.01553237,  4.41286068,  0.7810791 ]  # -10mm
    # [0.00719648,  0.8443429,  -0.03947203, -2.03661865, -0.01318968,  4.42484039,  0.78348306] # -20mm
    # [0.00689032,  0.87169665, -0.03901226, -2.02049281, -0.01335772,  4.4361352,  0.78334327]   # -30mm
    # [0.00675194,  0.89923439, -0.03876085, -2.00355735, -0.01229395,  4.44688982, 0.78401951]  # -40mm
    # [0.00560393,  0.92695904, -0.03755379, -1.98562784, -0.01500635,  4.457537,  0.78219662]    $ -50mm
    # [0.00611077,  0.95472431, -0.03799757, -1.96735045, -0.01266424,  4.46640307,  0.78294461]
    # [0.00537377,  0.98278969, -0.03724054, -1.9477574 , -0.01489819,  4.47587418,  0.78296324]

    #[0.00598084,  0.60881906, -0.03928992, -2.29356142, -0.01350361,  4.44381729,  0.77569607]
    #[0.00553582,  0.75293517, -0.03834142, -2.22669967, -0.01299479,  4.5223186,  0.77857714]

    #[0.00783327163909093, 0.5558480613472636, -0.04036106289390794, -2.163110868612272, -0.015487256765292425, 4.260755619281773, 0.7779242274028955]
    #'''

    des_joint_positions = [0.00719648,  0.8443429,  -0.03947203, -2.03661865, -0.01318968,  4.42484039,  0.78348306]

    '''
    # Golden resetting joints
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]
    #'''

    # This is for varying initialization of joints a little bit to
    # increase data variation.
    reset_joint_positions = [
        e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
        for e in reset_joint_positions
    ]

    # Set global start time for consistent timestamps
    global_start_time = time.time()

    logger.info("Moving to reset joint positions...")
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)

    logger.info("Reset joint positions reached.")
    time.sleep(1)  # Wait for a moment before moving to the desired position

    logger.info("Moving to desired joint positions...")
    move_to_position(robot_interface, np.array(des_joint_positions), controller_cfg)

    logger.info("Desired joint positions reached.")
    time.sleep(3)  # Wait for a moment before moving to the desired position

    logger.info("Moving to reset joint positions...")
    move_to_position(robot_interface, np.array(reset_joint_positions), controller_cfg)

    logger.info("Reset joint positions reached.")
    
    # Plot the Z position of the end-effector over time
    plot_z_position()

    robot_interface.close()

if __name__ == "__main__":
    main()
