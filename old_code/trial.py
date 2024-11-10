from deoxys.franka_interface import FrankaInterface

robot_interface = FrankaInterface("/home/robin/deoxys_control/deoxys/config/charmander.yml")
while True:
    if len(robot_interface._state_buffer) == 0:
        continue
    last_state = robot_interface._state_buffer[-1]
    print(last_state.q)
    print("")