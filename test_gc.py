import os
import sys
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
force_threshold = 10
torque_threshold = 1


parser = argparse.ArgumentParser()
parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
parser.add_argument("--controller-type", type=str, default="OSC_POSE")
args = parser.parse_args()

robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
joint_controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
joint_impedance_controller_cfg = get_default_controller_config("JOINT_IMPEDANCE")
osc_controller_cfg = get_default_controller_config("OSC_POSE")


while robot_interface.state_buffer_size == 0:
    logger.warning("Robot state not received")
    time.sleep(0.5)

# joint pos
# og_qpos = np.array(robot_interface.last_q)
# action = np.zeros(8)
# action[:7] = og_qpos
# action[-1] = -1

# for _ in range(40):
#     robot_interface.control(controller_type="JOINT_POSITION", 
#                             action=action, 
#                             controller_cfg=joint_controller_cfg)
#     time.sleep(0.05)

# osc
# og_eef_rot, og_eef_pos = robot_interface.last_eef_rot_and_pos
# for i in range(40):
#     action = np.zeros(7)
#     cur_rot, cur_pos = robot_interface.last_eef_rot_and_pos
#     action[:3] = og_eef_pos.reshape(3) - cur_pos.reshape(3)
#     action[-1] = -1
#     robot_interface.control(controller_type="OSC_POSE", 
#                             action=action, 
#                             controller_cfg=osc_controller_cfg)
#     time.sleep(0.05)
#     print(i)

# joint impedance
og_qpos = np.array(robot_interface.last_q)
action = np.zeros(8)
action[:7] = og_qpos
action[-1] = -1
for _ in range(40):
    robot_interface.control(controller_type="JOINT_IMPEDANCE", 
                            action=action, 
                            controller_cfg=joint_controller_cfg)
    time.sleep(0.05)


print('osc start')
osc_controller_cfg.Kp.translation = 0.0
osc_controller_cfg.Kp.rotation = 0.0
action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0])
for _ in range(200):
    robot_interface.control(controller_type="OSC_POSE", 
                            action=action, 
                            controller_cfg=osc_controller_cfg)
    print('osc run')


