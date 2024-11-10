"""Example script of moving robot joint positions."""
import time
from pathlib import Path

import numpy as np
import torch
import time

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.apriltag_utils import AprilTagDetector
import cv2
from ForceSensor import ForceSensor
import torch
import torch.nn as nn
import threading
from multiprocessing import Process
import sys

# sys.setswitchinterval(1e-6)



class DummyPolicy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DummyPolicy, self).__init__()
        self.policy = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.policy(x)


class OSC_POSE_variable_kp:

    def __init__(self,):

        # policy parameters
        self.device = "cuda:0"
        self.state_dim = 28
        self.action_dim = 13
        self.checkpoint = "agent_0.9787234042553191.pt"
        self.epi_length = 200
        
        # policy frequency in Hz
        self.control_freq = 10
        self.control_rate = 1/self.control_freq

        # for converting eef position in robot base frame to world frame
        self.robot_base_pos = [-0.5,  -0.06,  0.75 ]
        self.controller_type = "OSC_POSE"
        self.interface_cfg = "charmander.yml"
        self.robot_interface, self.controller_cfg = self.setup_controller()
        self.agent = self.setup_policy()
        self.force_sensor = ForceSensor("/dev/ttyUSB0", np.array([-10.982798427581788, -144.29237174987793, 9.55335311985016]))
        self.force_sensor.force_sensor_setup()

        self.too_much_force = False
        # print("!!!!")
        # exit()
        
    
    # take a look at the yml file before deploy your policy
    # make sure you understand the meaning of each parameter
    def setup_controller(self,):
        robot_interface = FrankaInterface(
            config_root + "/" +self.interface_cfg, 
            use_visualizer=False,
            control_freq=self.control_freq,
        )
        controller_cfg = get_default_controller_config("OSC_POSE")
        return robot_interface, controller_cfg


    def setup_policy(self,):
        agent = DummyPolicy(in_dim=self.state_dim,out_dim=self.action_dim).to(self.device)
        # load in checkpoint for the policy
        # agent = torch.load(self.checkpoint, map_location=self.device)
        agent.eval()
        return agent

    def get_robot_obs(self,):
        # joint angle
        joint_q = self.robot_interface.last_q
        # joint angle cosine/sine
        joint_q_cos = np.cos(joint_q)
        joint_q_sin = np.sin(joint_q)
        # joint angle vel
        joint_vel = self.robot_interface.last_dq
        # end-effector pose
        # eef position in robosuite is in world frame
        # eff position in FrankaInterface is in robot base frame
        # convert to world frame in here
        eff_pos = self.robot_interface.last_eef_pose[:3, 3:].flatten() + self.robot_base_pos
        eff_ori = transform_utils.mat2quat(self.robot_interface.last_eef_pose[:3, :3]).flatten()

        #get force reading from the force sensor
        force = self.force_sensor.get_force_obs()

        return joint_q_cos, joint_q_sin, joint_vel, eff_pos, eff_ori

    # get robot states, camera obs, force/torque sensor obs
    def get_obs(self,):
        # 7 + 7 + 7 + 3 + 4 = 28
        joint_q_cos, joint_q_sin, joint_vel, eff_pos, eff_ori = self.get_robot_obs()
        obs = np.concatenate((joint_q_cos, joint_q_sin, joint_vel, eff_pos, eff_ori))
        return obs

    # post process value from neural network
    # normal/denormalize the values
    def post_action(self, action):
        # do similar thing as in robosuite here
        action = np.clip(action, -1.0, 1.0)
        stiffness = action[:6]
        stiffness = (stiffness + 1) * 150
        translation_stiffness = stiffness[:3]
        orientation_stiffness = stiffness[3:]

        # eff pose
        delta_translation_orientation_gripper = action[6:]
        return translation_stiffness, orientation_stiffness, delta_translation_orientation_gripper


    def run_policy(self,):
        # your policy's initial position
        # be careful with the values here !!!!!!!!!!
        # check the value by trying it in robosuite !!!!!!!!!!!!!
        # reset_joint_positions = [
        #         0.09162008114028396,
        #         -0.19826458111314524,
        #         -0.01990020486871322,
        #         -2.4732269941140346,
        #         -0.01307073642274261,
        #         2.30396583422025,
        #         0.8480939705504309,
        #         ]
        
        reset_joint_positions = [0.49952223, 0.83049418, 0.35152248, -1.76062884, -0.96152792, 1.04167855, 0.48556356]

        # reset to your policy's initial position
        reset_joints_to(self.robot_interface, reset_joint_positions)

        
        # be careful before you run the policy in real robot!
        # always start with small actions like small_action = action * 0.001
        # exit()


        # timestamp for policy frequency
        prev_time = time.time()
        step_num = 0

        # you may want to start with a small number of time steps
        # never trust your policy, it can destroy our robot
        while step_num < self.epi_length:
            # get robot state
            obs = self.get_obs()

            # get action from agent
            with torch.no_grad():
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action = self.agent(obs)
            action = action.cpu().numpy()

            # normal/denormalize value from nn to action for robot
            translation_stiffness, orientation_stiffness, delta_translation_orientation_gripper_action = self.post_action(action)

            
            
            if self.too_much_force == True:
                self.controller_cfg.translation = [150, 150, 150]
                self.controller_cfg.rotation = [150, 150, 150]
                delta_translation_orientation_gripper_action = np.zeros(delta_translation_orientation_gripper_action.shape)
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=delta_translation_orientation_gripper_action,
                    controller_cfg=self.controller_cfg,
                )
                print("gravity compensation")
            else:
                self.controller_cfg.translation = translation_stiffness
                self.controller_cfg.rotation = orientation_stiffness

                self.controller_cfg.translation = [150, 150, 150]
                self.controller_cfg.rotation = [150, 150, 150]
                delta_translation_orientation_gripper_action = np.zeros(delta_translation_orientation_gripper_action.shape)
                # delta_translation_orientation_gripper_action[1] = -1
                # delta_translation_orientation_gripper_action[2] = 1
                print(delta_translation_orientation_gripper_action)
                # send action to controller
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=delta_translation_orientation_gripper_action,
                    controller_cfg=self.controller_cfg,
                )

            # ensure the policy frequency
            overhead_time = time.time() - prev_time
            if overhead_time < self.control_rate:
                time.sleep(self.control_rate - overhead_time)
            
            print("overhead_time: ",overhead_time)
            prev_time = time.time()
            step_num+=1


def check_force_limit(force_sensor):
    force_sensor.force_sensor_setup()
    while True:
        # get force reading from the force sensor
        force = force_sensor.get_force_obs()
        print(force)
        if np.linalg.norm(force) > 1:
            exit()

            
if __name__ == "__main__":
    controller = OSC_POSE_variable_kp()
    controller.run_policy()
    controller_thread = threading.Thread(target=controller.run_policy)
    force_checker_thread = threading.Thread(target=check_force_limit, args=(controller,))
    force_checker_process = Process(target=check_force_limit, args=(ForceSensor("/dev/ttyUSB0", np.array([-10.982798427581788, -144.29237174987793, 9.55335311985016])),))
    force_checker_process.start()
    controller_thread.run()
    force_checker_process.join()
    controller_thread.too_much_force = True
    
    # time.sleep(0.01)
    

    # exit()
    
    controller.robot_interface.close()