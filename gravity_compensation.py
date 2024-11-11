import os 
import argparse
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    args = parser.parse_args()
    return args

def turn_on_gravity_compensation(robot_interface, controller_type, controller_cfg):
    while robot_interface.state_buffer_size == 0:
        logger.warn("Robot state not received")
        time.sleep(0.5)

    # Send gravity compensation command
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0])
    robot_interface.control(
        controller_type=controller_type,
        action=action,
        controller_cfg=controller_cfg,
    )
    print("Gravity compensation mode activated.")

def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_type = args.controller_type

    controller_cfg = get_default_controller_config(controller_type)

    turn_on_gravity_compensation(robot_interface, controller_type, controller_cfg)

    # Close the interface after a delay to observe gravity compensation
    time.sleep(10)  # Adjust the sleep duration as needed to keep gravity compensation active
    robot_interface.close()

if __name__ == "__main__":
    main()
