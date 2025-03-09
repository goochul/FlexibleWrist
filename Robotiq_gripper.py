# Option 1: If your __init__.py exposes RobotiqGripper:
from pyRobotiqGripper import RobotiqGripper

# Option 2: If not, import directly from the module:
# from pyRobotiqGripper.pyrobotgripper import RobotiqGripper

gripper = RobotiqGripper()

gripper.activate()
gripper.calibrate(0, 40)
print("Calibrated")

gripper.open()
print("Opened")
gripper.close()
print("Closed")
# gripper.goTo(100)
# print("Go to 100")
# position_in_bit = gripper.getPosition()
# print(position_in_bit)

gripper.goTomm(5)
position_in_mm = gripper.getPositionmm()
print(position_in_mm)

gripper.goTomm(15)
position_in_mm = gripper.getPositionmm()
print(position_in_mm)

gripper.goTomm(25)
# print("Go to 25mm")
position_in_mm = gripper.getPositionmm()
print(position_in_mm)

# gripper.goTomm(35)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)

# gripper.goTomm(40)
# position_in_mm = gripper.getPositionmm()
# print(position_in_mm)


# gripper.close()

gripper.printInfo()
